"""Validation-only review policy, then one pass over the held-out test images.

    python scripts/evaluate_review_policy.py validate   # pool only; writes the frozen policy
    python scripts/evaluate_review_policy.py test        # reads the frozen policy; scores the test set once

Phase `validate` never opens a test image. It runs 5-fold cross-validation over the 80-image
labeled pool (the same ResNet18 recipe, trained on 64 and predicting the other 16 each time, fixed
epoch count so no fold is used for early stopping), then from the 80 out-of-fold predictions:
  - fits temperature leave-one-fold-out and adopts it only if pooled ECE improves,
  - picks the review threshold that maximises auto-decision coverage subject to auto-decided
    accuracy >= 0.90 (or records that none exists),
  - picks an anomaly-score threshold from the known-good pool images and evaluates the anomaly gate
    as implemented (anomaly score >= threshold sends the image to review),
  - decides whether the gate is enabled, by a criterion written down here before any test data:
    enabled only if the combined policy reaches auto-accuracy >= 0.90 at coverage >= 0.50 on the pool.
Phase `test` refuses to run without the frozen file, checks the pool and test sets are disjoint, and
scores the existing final checkpoint and PatchCore once on the 35 held-out images.
The CV models are not the deployed checkpoint; they exist only to produce out-of-fold predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from automotive_vision.data.augmentation import build_train_transform
from automotive_vision.data.dataset import (
    AnomalyDetectionDataset,
    ClassificationDataset,
    scan_mvtec_category,
)
from automotive_vision.data.preprocessing import build_eval_transform
from automotive_vision.data.splitting import labeled_pool_split
from automotive_vision.models.anomaly_detector import PatchCoreLite
from automotive_vision.models.classifier import DefectClassifier
from automotive_vision.training.calibration import (
    expected_calibration_error,
    fit_temperature,
)
from automotive_vision.training.evaluate import classification_metrics
from automotive_vision.training.review_policy import (
    DEFAULT_TARGET_ACCURACY,
    assert_disjoint,
    assert_same_ids,
    base_rate,
    binary_decision_metrics,
    paired_auroc_difference_ci,
    policy_outcome,
    select_anomaly_threshold,
    select_review_threshold,
    stratified_bootstrap_ci,
    wilson_interval,
)
from automotive_vision.training.train_classifier import (
    collect_logits,
    compute_class_weights,
)
from automotive_vision.utils.config import load_config
from automotive_vision.utils.seed import set_seed

ROOT = Path(__file__).resolve().parent.parent
FROZEN_PATH = ROOT / "reports" / "metrics" / "review_policy_frozen.json"
RESULT_PATH = ROOT / "reports" / "metrics" / "review_policy_evaluation.json"
N_FOLDS = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.7  # the shipped, unvalidated default in configs/inference.yaml
GATE_MIN_AUTO_ACCURACY, GATE_MIN_COVERAGE = 0.90, 0.50
ANOMALY_QUANTILE = 0.95
DEVICE = "cpu"


def _setup():
    config = load_config("configs/base.yaml", "configs/training.yaml")
    with open(ROOT / "configs" / "inference.yaml") as handle:
        inference_cfg = yaml.safe_load(handle)
    inference_cfg = inference_cfg.get("inference", inference_cfg)
    records = scan_mvtec_category(config["dataset"]["category_dir"])
    labeled = [r for r in records if r.official_split == "test"]
    pool, test = labeled_pool_split(
        labeled, test_size=config["training"]["held_out_fraction"], seed=config["seed"]
    )
    class_names = sorted({r.defect_type for r in labeled} | {"good"})
    return config, inference_cfg, pool, test, class_names


def _rid(record) -> str:
    return "/".join(Path(record.path).parts[-2:])


def _sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _transforms(config):
    d = config["dataset"]
    args = (d["image_size"], d["normalize_mean"], d["normalize_std"])
    return build_train_transform(*args), build_eval_transform(*args)


def _train_fixed_epochs(train_ds, class_names, train_cfg, epochs: int, seed: int):
    """Same recipe as train_classifier (model, loss weights, AdamW), fixed epochs, no val use."""
    set_seed(seed)
    model = DefectClassifier(
        num_classes=len(class_names),
        freeze_until_layer4=train_cfg["freeze_until_layer4"],
        dropout_p=train_cfg["dropout_p"],
    ).to(DEVICE)
    loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    labels = [train_ds.label_to_idx[r.defect_type] for r in train_ds.records]
    weights = compute_class_weights(labels, len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights) if train_cfg["use_class_weights"] else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    for _ in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            criterion(model(batch["image"].to(DEVICE)), batch["label"].to(DEVICE)).backward()
            optimizer.step()
    return model


def _probs(logits: torch.Tensor, temperature: float) -> np.ndarray:
    return torch.softmax(logits / temperature, dim=-1).numpy()


def _anomaly_scores(detector, records, eval_transform, image_size, batch_size):
    ds = AnomalyDetectionDataset(records, eval_transform, mask_size=image_size)
    scores = detector.score_dataset(DataLoader(ds, batch_size=batch_size, shuffle=False))
    return np.asarray(scores.image_scores, dtype=float), [Path(p).as_posix() for p in scores.paths]


def validate() -> None:
    config, inference_cfg, pool, test, class_names = _setup()
    train_cfg = config["training"]
    label_to_idx = {n: i for i, n in enumerate(class_names)}
    pool_ids = [_rid(r) for r in pool]
    test_ids_sha = _sha([_rid(r) for r in test])  # ids only: no test image is opened in this phase
    assert_disjoint(pool_ids, [_rid(r) for r in test])
    final_meta = json.loads((ROOT / "reports/metrics/supervised_classification.json").read_text())
    epochs = int(final_meta["best_epoch"]) + 1  # epochs up to the final run's best (chosen on its val)

    # Groups: identical image bytes must share a fold (MVTec has none; this checks, not assumes).
    digests = [hashlib.md5(Path(r.path).read_bytes()).hexdigest() for r in pool]
    y = np.array([label_to_idx[r.defect_type] for r in pool])
    folds = list(
        StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=config["seed"]).split(
            np.zeros(len(pool)), y, groups=digests
        )
    )
    train_tf, eval_tf = _transforms(config)
    oof_logits = torch.zeros(len(pool), len(class_names))
    fold_of = np.zeros(len(pool), dtype=int)
    for k, (tr, va) in enumerate(folds):
        model = _train_fixed_epochs(
            ClassificationDataset([pool[i] for i in tr], label_to_idx, train_tf),
            class_names, train_cfg, epochs, seed=config["seed"] + k,
        )
        loader = DataLoader(
            ClassificationDataset([pool[i] for i in va], label_to_idx, eval_tf),
            batch_size=train_cfg["batch_size"], shuffle=False,
        )
        logits, _ = collect_logits(model, loader, DEVICE)
        oof_logits[va] = logits
        fold_of[va] = k
        print(f"fold {k}: trained on {len(tr)}, predicted {len(va)}")
    labels_t = torch.tensor(y)

    # Temperature: leave-one-fold-out, adopted only if pooled ECE improves.
    raw_conf = _probs(oof_logits, 1.0).max(axis=1)
    raw_correct = (oof_logits.argmax(1).numpy() == y)
    lofo_conf = np.zeros(len(pool))
    for k in range(N_FOLDS):
        held = fold_of == k
        t_k = fit_temperature(oof_logits[~torch.tensor(held)], labels_t[~torch.tensor(held)])
        lofo_conf[held] = _probs(oof_logits[torch.tensor(held)], t_k).max(axis=1)
    ece_raw = expected_calibration_error(raw_conf, raw_correct)
    ece_lofo = expected_calibration_error(lofo_conf, raw_correct)
    t_all = fit_temperature(oof_logits, labels_t)
    adopt_t = bool(ece_lofo < ece_raw)
    temperature = t_all if adopt_t else 1.0
    conf = _probs(oof_logits, temperature).max(axis=1)

    selection = select_review_threshold(conf, raw_correct, DEFAULT_TARGET_ACCURACY)
    clf_threshold = selection["threshold"] if selection["selected"] else DEFAULT_CONFIDENCE_THRESHOLD

    # Anomaly gate, as implemented: needs_review = classifier flag, else anomaly_score >= threshold.
    detector = PatchCoreLite.load(inference_cfg["anomaly_checkpoint_path"], device=DEVICE)
    scores, _ = _anomaly_scores(
        detector, pool, eval_tf, config["dataset"]["image_size"],
        config.get("anomaly_detection", {}).get("batch_size", 8),
    )
    is_defect = np.array([int(r.is_anomalous) for r in pool])
    anomaly_threshold = select_anomaly_threshold(scores, is_defect, ANOMALY_QUANTILE)
    flag_clf = conf < clf_threshold
    flag_all = flag_clf | (scores >= anomaly_threshold)
    combined = policy_outcome((~flag_all).astype(float), raw_correct, 0.5)  # confidence 1 = auto
    gate_enabled = bool(
        combined["auto_accuracy"] is not None
        and combined["auto_accuracy"] >= GATE_MIN_AUTO_ACCURACY
        and combined["coverage"] >= GATE_MIN_COVERAGE
    )

    pred = oof_logits.argmax(1).numpy()
    oof_metrics = classification_metrics(y, pred, class_names)
    frozen = {
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": __doc__.strip().split("\n\n")[1] if __doc__ else "",
        "pool_ids": pool_ids,
        "test_ids_sha256": test_ids_sha,
        "cv": {
            "n_folds": N_FOLDS, "folds": [int(f) for f in fold_of], "epochs_fixed": epochs,
            "grouped_by_identical_image_bytes": True,
            "n_duplicate_image_groups": int(len(digests) - len(set(digests))),
            "note": "CV models are not the deployed checkpoint; fixed epochs, no fold used for early stopping",
        },
        "temperature": {
            "ece_raw_oof": ece_raw, "ece_leave_one_fold_out": ece_lofo, "fitted_on_all_oof": t_all,
            "adopted": adopt_t, "value_used": temperature,
        },
        "review_threshold_selection": {
            "rule": "maximise auto-decision coverage subject to out-of-fold auto-accuracy >= 0.90",
            **{k: v for k, v in selection.items()},
        },
        "classifier_policy": {
            "threshold": clf_threshold,
            "validated": bool(selection["selected"]),
            "note": "validated on out-of-fold pool" if selection["selected"]
            else "no threshold met the target; 0.7 is the unvalidated shipped default",
        },
        "anomaly_gate": {
            "rule": f"threshold = {ANOMALY_QUANTILE} quantile of known-good pool anomaly scores",
            "threshold": anomaly_threshold,
            "enable_criterion": f"combined policy auto-accuracy >= {GATE_MIN_AUTO_ACCURACY} and coverage >= {GATE_MIN_COVERAGE} on the pool",
            "pool_combined_policy": combined,
            "pool_anomaly_flag_rate_good": float((scores[is_defect == 0] >= anomaly_threshold).mean()),
            "pool_anomaly_flag_rate_defective": float((scores[is_defect == 1] >= anomaly_threshold).mean()),
            "enabled": gate_enabled,
        },
        "pool_oof": {
            "n": len(pool),
            "accuracy": float(raw_correct.mean()),
            "accuracy_ci95": wilson_interval(int(raw_correct.sum()), len(pool)),
            "classifier_only_policy": policy_outcome(conf, raw_correct, clf_threshold),
            "classification_metrics": oof_metrics,
        },
    }
    FROZEN_PATH.write_text(json.dumps(frozen, indent=2))
    print(f"wrote {FROZEN_PATH}")
    print(json.dumps({k: frozen[k] for k in ("temperature", "classifier_policy")}, indent=1))
    print("threshold selection feasible:", selection["selected"])
    print("gate enabled:", gate_enabled, "| pool combined:", combined)


def test() -> None:
    if not FROZEN_PATH.exists():
        raise SystemExit("run `validate` first: the policy must be frozen before the test set is scored")
    frozen = json.loads(FROZEN_PATH.read_text())
    config, inference_cfg, _pool, test_recs, class_names = _setup()
    test_ids = [_rid(r) for r in test_recs]
    if _sha(test_ids) != frozen["test_ids_sha256"]:
        raise SystemExit("the test set changed since the policy was frozen")
    assert_disjoint(frozen["pool_ids"], test_ids)
    label_to_idx = {n: i for i, n in enumerate(class_names)}
    train_tf, eval_tf = _transforms(config)
    del train_tf

    ckpt = torch.load(inference_cfg["checkpoint_path"], map_location=DEVICE, weights_only=False)
    model = DefectClassifier(
        num_classes=len(ckpt["class_names"]),
        freeze_until_layer4=ckpt["config"]["freeze_until_layer4"],
        dropout_p=ckpt["config"]["dropout_p"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    assert ckpt["class_names"] == class_names
    loader = DataLoader(
        ClassificationDataset(test_recs, label_to_idx, eval_tf),
        batch_size=config["training"]["batch_size"], shuffle=False,
    )
    logits, labels = collect_logits(model, loader, DEVICE)
    probs = _probs(logits, frozen["temperature"]["value_used"])
    y = labels.numpy()
    pred, conf = probs.argmax(1), probs.max(1)
    correct = pred == y

    detector = PatchCoreLite.load(inference_cfg["anomaly_checkpoint_path"], device=DEVICE)
    scores, score_paths = _anomaly_scores(
        detector, test_recs, eval_tf, config["dataset"]["image_size"],
        config.get("anomaly_detection", {}).get("batch_size", 8),
    )
    assert_same_ids([_rid(r) for r in test_recs], test_ids, "classifier vs PatchCore")
    assert len(score_paths) == len(test_ids) == len(y)

    good = class_names.index("good")
    defective = (y != good).astype(int)  # 1 = defective, from the same labels for both models
    clf_score = 1.0 - probs[:, good]
    n_pos, n_neg = int(defective.sum()), int((defective == 0).sum())

    def auc_block(score: np.ndarray) -> dict:
        return {
            "auroc": float(roc_auc_score(defective, score)),
            "auroc_ci95_stratified_bootstrap": stratified_bootstrap_ci(defective, score, roc_auc_score),
            "auprc": float(average_precision_score(defective, score)),
            "auprc_ci95_stratified_bootstrap": stratified_bootstrap_ci(defective, score, average_precision_score),
            "auprc_of_uninformative_scorer": base_rate(defective),
        }

    thr_a = frozen["anomaly_gate"]["threshold"]
    cls_metrics = classification_metrics(y, pred, class_names)
    per_class = {
        name: {**cls_metrics["per_class"][name],
               "recall_ci95_wilson": wilson_interval(round(cls_metrics["per_class"][name]["recall"] * cls_metrics["per_class"][name]["support"]), cls_metrics["per_class"][name]["support"])}
        for name in class_names
    }
    clf_threshold = frozen["classifier_policy"]["threshold"]
    flag_clf = conf < clf_threshold
    gate_on = frozen["anomaly_gate"]["enabled"]
    flag_all = flag_clf | ((scores >= thr_a) if gate_on else False)
    gate_if_enabled = flag_clf | (scores >= thr_a)
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "frozen_policy_created_utc": frozen["created_utc"],
        "test_set": {"n": len(y), "n_defective": n_pos, "n_good": n_neg, "ids": test_ids},
        "five_class_classifier_only": {
            "accuracy": cls_metrics["accuracy"],
            "accuracy_ci95_wilson": wilson_interval(int(correct.sum()), len(y)),
            "macro_f1": cls_metrics["macro_f1"],
            "per_class": per_class,
            "confusion_matrix": cls_metrics["confusion_matrix"],
            "class_names": class_names,
        },
        "binary_defect_detection_same_images": {
            "task": "defective (4 defect types) vs good, same 35 image ids, same labels",
            "classifier_score": "1 - P(good)",
            "patchcore_score": "max-patch nearest-neighbour distance (final memory bank, 220 train normals)",
            "classifier": auc_block(clf_score),
            "patchcore": auc_block(scores),
            "auroc_difference_patchcore_minus_classifier_ci95_paired": paired_auroc_difference_ci(
                defective, scores, clf_score, roc_auc_score
            ),
            "classifier_decision_argmax_not_good": binary_decision_metrics(defective, pred != good),
            "patchcore_decision_at_frozen_threshold": {
                "threshold": thr_a, **binary_decision_metrics(defective, scores >= thr_a)
            },
        },
        "review_policy_frozen": {
            "classifier_threshold": clf_threshold,
            "validated": frozen["classifier_policy"]["validated"],
            "note": frozen["classifier_policy"]["note"],
            "outcome": policy_outcome(conf, correct, clf_threshold),
        },
        "anomaly_gate": {
            "enabled_in_policy": gate_on,
            "enable_criterion": frozen["anomaly_gate"]["enable_criterion"],
            "evaluated_as_if_enabled": policy_outcome((~gate_if_enabled).astype(float), correct, 0.5),
            "anomaly_flag_rate_good": float((scores[defective == 0] >= thr_a).mean()),
            "anomaly_flag_rate_defective": float((scores[defective == 1] >= thr_a).mean()),
        },
        "per_image": [
            {"id": test_ids[i], "true": class_names[y[i]], "pred": class_names[pred[i]],
             "confidence": float(conf[i]), "anomaly_score": float(scores[i])}
            for i in range(len(y))
        ],
        "policy_in_effect_outcome": policy_outcome((~flag_all).astype(float), correct, 0.5),
    }
    RESULT_PATH.write_text(json.dumps({"frozen": frozen, "test": result}, indent=2))
    print(f"wrote {RESULT_PATH}")
    summary = {
        "five_class": {k: result["five_class_classifier_only"][k] for k in ("accuracy", "accuracy_ci95_wilson", "macro_f1")},
        "binary_clf": result["binary_defect_detection_same_images"]["classifier"],
        "binary_pc": result["binary_defect_detection_same_images"]["patchcore"],
        "diff_ci": result["binary_defect_detection_same_images"]["auroc_difference_patchcore_minus_classifier_ci95_paired"],
        "review": result["review_policy_frozen"],
        "gate": result["anomaly_gate"],
    }
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["validate", "test"])
    {"validate": validate, "test": test}[parser.parse_args().phase]()
