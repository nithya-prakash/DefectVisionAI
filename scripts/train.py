#!/usr/bin/env python3
"""Train and evaluate the supervised defect classifier.

Usage:
    python scripts/train.py --config configs/training.yaml

Trains on a stratified split of the officially-labeled pool (see data/README.md for
why this differs from the anomaly-detection track's protocol), evaluates on a held-out
test set never touched during training/validation, and writes real metrics + a
confusion matrix figure.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from automotive_vision.data.dataset import ClassificationDataset, scan_mvtec_category  # noqa: E402
from automotive_vision.data.augmentation import build_train_transform  # noqa: E402
from automotive_vision.data.preprocessing import build_eval_transform  # noqa: E402
from automotive_vision.data.splitting import labeled_pool_split, stratified_split  # noqa: E402
from automotive_vision.training.train_classifier import evaluate_model, train_classifier  # noqa: E402
from automotive_vision.utils.config import load_config  # noqa: E402
from automotive_vision.utils.logging import get_logger  # noqa: E402
from automotive_vision.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)


def resolve_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available; falling back to cpu.")
        return "cpu"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()

    config = load_config(args.base_config, args.config)
    set_seed(config["seed"])
    train_cfg = config["training"]
    dataset_cfg = config["dataset"]
    device = resolve_device(train_cfg["device"])

    records = scan_mvtec_category(dataset_cfg["category_dir"])
    labeled_records = [r for r in records if r.official_split == "test"]
    pool, held_out_test = labeled_pool_split(
        labeled_records, test_size=train_cfg["held_out_fraction"], seed=config["seed"]
    )
    train_records, val_records = stratified_split(
        pool, test_size=train_cfg["val_fraction"], seed=config["seed"]
    )
    logger.info(
        f"Split: {len(train_records)} train, {len(val_records)} val, "
        f"{len(held_out_test)} held-out test"
    )

    class_names = sorted({r.defect_type for r in records if r.official_split == "test"})
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    logger.info(f"Classes: {class_names}")

    train_transform = build_train_transform(
        dataset_cfg["image_size"], dataset_cfg["normalize_mean"], dataset_cfg["normalize_std"]
    )
    eval_transform = build_eval_transform(
        dataset_cfg["image_size"], dataset_cfg["normalize_mean"], dataset_cfg["normalize_std"]
    )

    train_ds = ClassificationDataset(train_records, label_to_idx, train_transform)
    val_ds = ClassificationDataset(val_records, label_to_idx, eval_transform)
    test_ds = ClassificationDataset(held_out_test, label_to_idx, eval_transform)

    model, history = train_classifier(train_ds, val_ds, class_names, train_cfg, device)
    logger.info(
        f"Best epoch {history.best_epoch}: val macro-F1={history.best_val_macro_f1:.4f}"
    )

    test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"], shuffle=False)
    test_loss, test_metrics = evaluate_model(model, test_loader, device, class_names)
    logger.info(f"Held-out test: accuracy={test_metrics['accuracy']:.4f}, macro-F1={test_metrics['macro_f1']:.4f}")

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "defect_classifier.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "config": train_cfg,
        },
        checkpoint_path,
    )
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    reports_dir = Path("reports")
    (reports_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    results = {
        "class_names": class_names,
        "n_train": len(train_records),
        "n_val": len(val_records),
        "n_held_out_test": len(held_out_test),
        "best_epoch": history.best_epoch,
        "epochs_run": len(history.train_loss),
        "history": {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_macro_f1": history.val_macro_f1,
        },
        "test_metrics": test_metrics,
    }
    out_json = reports_dir / "metrics" / "supervised_classification.json"
    out_json.write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote {out_json}")

    cm = np.array(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — held-out test (n={len(held_out_test)})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(reports_dir / "figures" / "classifier_confusion_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote reports/figures/classifier_confusion_matrix.png")


if __name__ == "__main__":
    main()
