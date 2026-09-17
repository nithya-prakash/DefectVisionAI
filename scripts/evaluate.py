#!/usr/bin/env python3
"""Calibration evaluation + human-review threshold analysis for the supervised classifier.

Usage:
    python scripts/evaluate.py

Loads the checkpoint trained by scripts/train.py, recomputes the identical
(seeded) val/test split, fits temperature scaling on the validation set, and reports
ECE before/after on the held-out test set plus a confidence-threshold sweep connecting
calibration to the human-review decision.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from automotive_vision.data.dataset import ClassificationDataset, scan_mvtec_category
from automotive_vision.data.preprocessing import build_eval_transform
from automotive_vision.data.splitting import labeled_pool_split, stratified_split
from automotive_vision.models.classifier import DefectClassifier
from automotive_vision.training.calibration import (
    expected_calibration_error,
    fit_temperature,
    reliability_diagram_data,
)
from automotive_vision.training.train_classifier import collect_logits
from automotive_vision.utils.config import load_config
from automotive_vision.utils.logging import get_logger
from automotive_vision.utils.seed import set_seed

logger = get_logger(__name__)


def resolve_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return requested


def confidences_and_correctness(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0):
    probs = F.softmax(logits / temperature, dim=-1).numpy()
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (preds == labels.numpy()).astype(int)
    return confidences, correct, preds


def threshold_sweep(confidences: np.ndarray, correct: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    rows = []
    for t in thresholds:
        auto_mask = confidences >= t
        review_mask = ~auto_mask
        rows.append(
            {
                "threshold": float(t),
                "review_rate": float(review_mask.mean()),
                "n_auto_decided": int(auto_mask.sum()),
                "auto_decided_accuracy": float(correct[auto_mask].mean()) if auto_mask.sum() > 0 else None,
                "n_reviewed": int(review_mask.sum()),
                "reviewed_accuracy": float(correct[review_mask].mean()) if review_mask.sum() > 0 else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    args = parser.parse_args()

    config = load_config(args.base_config, args.training_config)
    set_seed(config["seed"])
    train_cfg = config["training"]
    dataset_cfg = config["dataset"]
    device = resolve_device(train_cfg["device"])

    checkpoint_path = Path(train_cfg["checkpoint_dir"]) / "defect_classifier.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found — run scripts/train.py first.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]

    model = DefectClassifier(
        num_classes=len(class_names),
        freeze_until_layer4=train_cfg["freeze_until_layer4"],
        dropout_p=train_cfg["dropout_p"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    records = scan_mvtec_category(dataset_cfg["category_dir"])
    labeled_records = [r for r in records if r.official_split == "test"]
    pool, held_out_test = labeled_pool_split(
        labeled_records, test_size=train_cfg["held_out_fraction"], seed=config["seed"]
    )
    _, val_records = stratified_split(pool, test_size=train_cfg["val_fraction"], seed=config["seed"])
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}

    eval_transform = build_eval_transform(
        dataset_cfg["image_size"], dataset_cfg["normalize_mean"], dataset_cfg["normalize_std"]
    )
    val_ds = ClassificationDataset(val_records, label_to_idx, eval_transform)
    test_ds = ClassificationDataset(held_out_test, label_to_idx, eval_transform)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    val_logits, val_labels = collect_logits(model, val_loader, device)
    test_logits, test_labels = collect_logits(model, test_loader, device)

    temperature = fit_temperature(val_logits, val_labels)
    logger.info(f"Fitted temperature T={temperature:.3f} on validation set (n={len(val_labels)})")

    raw_conf, raw_correct, _ = confidences_and_correctness(test_logits, test_labels, temperature=1.0)
    cal_conf, cal_correct, _ = confidences_and_correctness(test_logits, test_labels, temperature=temperature)

    raw_ece = expected_calibration_error(raw_conf, raw_correct, n_bins=10)
    cal_ece = expected_calibration_error(cal_conf, cal_correct, n_bins=10)
    logger.info(f"Test ECE: raw={raw_ece:.4f}, temperature-scaled={cal_ece:.4f}")

    raw_diagram = reliability_diagram_data(raw_conf, raw_correct)
    cal_diagram = reliability_diagram_data(cal_conf, cal_correct)

    thresholds = np.arange(0.3, 1.0, 0.05)
    sweep = threshold_sweep(cal_conf, cal_correct, thresholds)

    # Selection rule (stated explicitly, not claimed optimal): the smallest threshold
    # for which auto-decided accuracy is >= 0.9 on this held-out test set, so most
    # predictions are automated while the automated subset stays reliable. With n=35
    # this is a noisy estimate, not a guarantee.
    candidate = next(
        (row for row in sweep if row["auto_decided_accuracy"] is not None and row["auto_decided_accuracy"] >= 0.9),
        None,
    )
    selected_threshold = candidate["threshold"] if candidate else 0.7
    logger.info(f"Selected review threshold: {selected_threshold} (rule: see scripts/evaluate.py comment)")

    reports_dir = Path("reports")
    (reports_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    results = {
        "n_val": len(val_labels),
        "n_test": len(test_labels),
        "temperature": temperature,
        "raw_ece": raw_ece,
        "calibrated_ece": cal_ece,
        "raw_reliability_diagram": raw_diagram,
        "calibrated_reliability_diagram": cal_diagram,
        "threshold_sweep": sweep,
        "selected_review_threshold": selected_threshold,
        "selection_rule": "smallest threshold with auto-decided accuracy >= 0.9 on held-out test (n=35); falls back to 0.7 if none qualify",
    }
    out_json = reports_dir / "metrics" / "calibration.json"
    out_json.write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote {out_json}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, diagram, title in [
        (axes[0], raw_diagram, f"Raw (ECE={raw_ece:.3f})"),
        (axes[1], cal_diagram, f"Temperature-scaled T={temperature:.2f} (ECE={cal_ece:.3f})"),
    ]:
        centers = np.array(diagram["bin_centers"])
        accs = np.array(diagram["bin_accuracies"])
        valid = ~np.isnan(accs)
        ax.bar(centers[valid], accs[valid], width=0.08, edgecolor="black", label="Accuracy")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
    fig.suptitle(f"Reliability diagram — held-out test (n={len(test_labels)})")
    fig.tight_layout()
    fig.savefig(reports_dir / "figures" / "reliability_diagram.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote reports/figures/reliability_diagram.png")


if __name__ == "__main__":
    main()
