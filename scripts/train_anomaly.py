#!/usr/bin/env python3
"""Fit and evaluate the anomaly-detection track: classical baseline + PatchCore-lite.

Usage:
    python scripts/train_anomaly.py [--config configs/base.yaml]

Follows the official MVTec AD protocol: fit on normal-only training images, evaluate
on the official test split (good + all defect types). Writes real metrics to
reports/metrics/anomaly_detection.json and a couple of figures to reports/figures/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from automotive_vision.data.dataset import (
    AnomalyDetectionDataset,
    scan_mvtec_category,
)
from automotive_vision.data.preprocessing import build_eval_transform
from automotive_vision.data.splitting import anomaly_detection_split
from automotive_vision.models.anomaly_detector import PatchCoreLite
from automotive_vision.models.classical_baseline import ClassicalAnomalyBaseline
from automotive_vision.training.evaluate import (
    image_level_anomaly_metrics,
    pixel_level_anomaly_metrics,
)
from automotive_vision.utils.config import load_config
from automotive_vision.utils.device import resolve_device
from automotive_vision.utils.logging import get_logger
from automotive_vision.utils.seed import set_seed

logger = get_logger(__name__)


def run_classical_baseline(train_records, test_records, feature_image_size: int) -> dict:
    logger.info("Fitting classical (non-deep-learning) baseline...")
    baseline = ClassicalAnomalyBaseline(feature_size=feature_image_size)
    start = time.time()
    baseline.fit([str(r.path) for r in train_records])
    scores = baseline.score_dataset(test_records)
    elapsed = time.time() - start

    metrics = image_level_anomaly_metrics(scores.labels, scores.image_scores)
    metrics["inference_seconds_total"] = elapsed
    metrics["inference_seconds_per_image"] = elapsed / len(test_records)
    logger.info(f"Classical baseline: image AUROC={metrics['image_auroc']:.4f}")
    return metrics, scores


def run_patchcore(config, train_records, test_records) -> dict:
    logger.info("Fitting PatchCore-lite anomaly detector...")
    dataset_cfg = config["dataset"]
    ad_cfg = config["anomaly_detection"]
    device = resolve_device(ad_cfg["device"])

    transform = build_eval_transform(
        dataset_cfg["image_size"], dataset_cfg["normalize_mean"], dataset_cfg["normalize_std"]
    )
    train_ds = AnomalyDetectionDataset(train_records, transform, mask_size=dataset_cfg["image_size"])
    test_ds = AnomalyDetectionDataset(test_records, transform, mask_size=dataset_cfg["image_size"])
    train_loader = DataLoader(train_ds, batch_size=ad_cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=ad_cfg["batch_size"], shuffle=False)

    detector = PatchCoreLite(
        image_size=dataset_cfg["image_size"],
        memory_bank_size=ad_cfg["memory_bank_size"],
        device=device,
        seed=config["seed"],
        nn_chunk_size=ad_cfg["nn_chunk_size"],
    )

    start = time.time()
    detector.fit(train_loader)
    fit_elapsed = time.time() - start
    logger.info(f"Memory bank built: {detector.memory_bank.shape[0]} patches in {fit_elapsed:.1f}s")

    start = time.time()
    scores = detector.score_dataset(test_loader)
    infer_elapsed = time.time() - start

    metrics = image_level_anomaly_metrics(scores.labels, scores.image_scores)
    metrics.update(pixel_level_anomaly_metrics(scores.masks, scores.pixel_maps))
    metrics["memory_bank_patches"] = int(detector.memory_bank.shape[0])
    metrics["fit_seconds"] = fit_elapsed
    metrics["inference_seconds_total"] = infer_elapsed
    metrics["inference_seconds_per_image"] = infer_elapsed / len(test_records)
    metrics["device"] = device
    logger.info(
        f"PatchCore-lite: image AUROC={metrics['image_auroc']:.4f}, "
        f"pixel AUROC={metrics['pixel_auroc']:.4f}"
    )

    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    detector.save(checkpoint_dir / "patchcore_memory_bank.pt")
    logger.info(f"Saved memory bank to {checkpoint_dir / 'patchcore_memory_bank.pt'}")

    return metrics, scores


def save_roc_figure(classical_scores, patchcore_scores, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, scores in [("Classical baseline", classical_scores), ("PatchCore-lite", patchcore_scores)]:
        fpr, tpr, _ = roc_curve(scores.labels, scores.image_scores)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Image-level ROC — MVTec AD metal_nut (test set, n=115)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_example_heatmaps(scores, out_path: Path, n_examples: int = 4) -> None:
    anomalous_idx = np.where(scores.labels == 1)[0][:n_examples]
    fig, axes = plt.subplots(2, len(anomalous_idx), figsize=(4 * len(anomalous_idx), 8))
    for col, idx in enumerate(anomalous_idx):
        from PIL import Image as PILImage

        img = PILImage.open(scores.paths[idx]).convert("RGB")
        axes[0, col].imshow(img)
        axes[0, col].set_title(scores.defect_types[idx])
        axes[0, col].axis("off")
        axes[1, col].imshow(scores.pixel_maps[idx], cmap="inferno")
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("Input")
    axes[1, 0].set_ylabel("Anomaly heatmap")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    category_dir = config["dataset"]["category_dir"]
    logger.info(f"Scanning {category_dir} ...")
    records = scan_mvtec_category(category_dir)
    train_records, test_records = anomaly_detection_split(records)
    logger.info(f"{len(train_records)} normal train images, {len(test_records)} test images")

    classical_metrics, classical_scores = run_classical_baseline(
        train_records, test_records, config["classical_baseline"]["feature_image_size"]
    )
    patchcore_metrics, patchcore_scores = run_patchcore(config, train_records, test_records)

    reports_dir = Path("reports")
    (reports_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    results = {
        "dataset": config["dataset"]["category"],
        "n_train_normal": len(train_records),
        "n_test": len(test_records),
        "seed": config["seed"],
        "classical_baseline": classical_metrics,
        "patchcore_lite": patchcore_metrics,
    }
    out_json = reports_dir / "metrics" / "anomaly_detection.json"
    out_json.write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote {out_json}")

    save_roc_figure(classical_scores, patchcore_scores, reports_dir / "figures" / "anomaly_roc.png")
    save_example_heatmaps(patchcore_scores, reports_dir / "figures" / "anomaly_heatmap_examples.png")
    logger.info("Wrote figures to reports/figures/")


if __name__ == "__main__":
    main()
