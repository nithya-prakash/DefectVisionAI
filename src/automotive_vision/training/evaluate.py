"""Reusable evaluation metrics for both the classical baseline and the deep models.

Every number here is computed from actual predictions passed in by the caller — no
metric in this module is ever hard-coded or estimated.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for a metric_fn(y_true, y_score) -> float."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # AUROC/AUPRC undefined with a single class in the resample
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    if not scores:
        return (float("nan"), float("nan"))
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return float(lower), float(upper)


def image_level_anomaly_metrics(
    labels: np.ndarray, scores: np.ndarray, n_bootstrap: int = 1000, seed: int = 42
) -> dict:
    """AUROC + AUPRC for binary (normal vs. anomalous) image-level anomaly scores."""
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    auroc_ci = bootstrap_ci(labels, scores, roc_auc_score, n_bootstrap, seed)
    auprc_ci = bootstrap_ci(labels, scores, average_precision_score, n_bootstrap, seed)
    return {
        "n_samples": int(len(labels)),
        "n_anomalous": int(labels.sum()),
        "n_normal": int(len(labels) - labels.sum()),
        "image_auroc": float(auroc),
        "image_auroc_ci95": auroc_ci,
        "image_auprc": float(auprc),
        "image_auprc_ci95": auprc_ci,
    }


def pixel_level_anomaly_metrics(masks: np.ndarray, pixel_maps: np.ndarray) -> dict:
    """Pixel-level AUROC over every pixel of every test image (flattened)."""
    y_true = masks.flatten().astype(int)
    y_score = pixel_maps.flatten()
    return {
        "n_pixels": int(len(y_true)),
        "n_positive_pixels": int(y_true.sum()),
        "pixel_auroc": float(roc_auc_score(y_true, y_score)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Accuracy/precision/recall/F1 (macro + per-class) and confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_class = {}
    for idx, name in enumerate(class_names):
        binary_true = (y_true == idx).astype(int)
        binary_pred = (y_pred == idx).astype(int)
        per_class[name] = {
            "precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
            "recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
            "f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
            "support": int(binary_true.sum()),
        }
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
