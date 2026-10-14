"""Validation-only review-policy selection and small-sample interval estimates.

Everything here takes plain arrays, and nothing takes a test-set argument: the review threshold
is chosen from out-of-fold predictions on the labeled pool, written to a frozen file, and only
then applied once to the held-out test images (scripts/evaluate_review_policy.py).

Intervals for a handful of images are wide; they are reported, not hidden:
- proportions (accuracy, recall, false-alarm rate): Wilson score interval,
- AUROC / AUPRC: percentile bootstrap that resamples within each class, so every resample keeps
  the real class counts (a plain bootstrap on 7 good images often drops to 0-2 of them).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

DEFAULT_TARGET_ACCURACY = 0.90
THRESHOLD_GRID = tuple(round(float(t), 2) for t in np.arange(0.30, 1.0001, 0.01))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile 95% bootstrap interval, resampling within each class."""
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y_true == 1)
    negatives = np.flatnonzero(y_true == 0)
    values = []
    for _ in range(n_bootstrap):
        idx = np.concatenate(
            [
                rng.choice(positives, len(positives), replace=True),
                rng.choice(negatives, len(negatives), replace=True),
            ]
        )
        values.append(metric_fn(y_true[idx], y_score[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_auroc_difference_ci(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% interval for metric(a) - metric(b) on the same images (paired, class-stratified)."""
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y_true == 1)
    negatives = np.flatnonzero(y_true == 0)
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.concatenate(
            [
                rng.choice(positives, len(positives), replace=True),
                rng.choice(negatives, len(negatives), replace=True),
            ]
        )
        diffs.append(metric_fn(y_true[idx], score_a[idx]) - metric_fn(y_true[idx], score_b[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def review_sweep(confidences: np.ndarray, correct: np.ndarray, thresholds: Sequence[float]) -> list[dict]:
    """Risk-coverage table: predictions with confidence >= t are auto-decided, the rest reviewed."""
    n = len(confidences)
    rows = []
    for t in thresholds:
        auto = confidences >= t
        n_auto = int(auto.sum())
        rows.append(
            {
                "threshold": float(t),
                "n_auto_decided": n_auto,
                "coverage": n_auto / n if n else float("nan"),
                "review_rate": 1 - n_auto / n if n else float("nan"),
                "auto_accuracy": float(correct[auto].mean()) if n_auto else None,
            }
        )
    return rows


def select_review_threshold(
    confidences: np.ndarray,
    correct: np.ndarray,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    thresholds: Sequence[float] = THRESHOLD_GRID,
) -> dict:
    """Maximise auto-decision coverage subject to auto-decided accuracy >= target.

    Coverage never increases with the threshold, so this is the smallest feasible threshold.
    Returns selected=False (and no threshold) when no threshold meets the target; the caller must
    then keep its simpler policy and say so. Inputs are validation / out-of-fold values only.
    """
    sweep = review_sweep(np.asarray(confidences), np.asarray(correct), thresholds)
    feasible = [
        r for r in sweep if r["auto_accuracy"] is not None and r["auto_accuracy"] >= target_accuracy
    ]
    if not feasible:
        return {"selected": False, "target_accuracy": target_accuracy, "sweep": sweep}
    best = max(feasible, key=lambda r: (r["coverage"], -r["threshold"]))
    return {
        "selected": True,
        "target_accuracy": target_accuracy,
        "threshold": best["threshold"],
        "coverage": best["coverage"],
        "review_rate": best["review_rate"],
        "auto_accuracy": best["auto_accuracy"],
        "sweep": sweep,
    }


def policy_outcome(confidences: np.ndarray, correct: np.ndarray, threshold: float) -> dict:
    """Review rate, coverage and auto-decided accuracy (with Wilson interval) at one threshold."""
    auto = np.asarray(confidences) >= threshold
    n, n_auto = len(auto), int(auto.sum())
    n_auto_correct = int(np.asarray(correct)[auto].sum())
    return {
        "threshold": float(threshold),
        "n": n,
        "n_reviewed": n - n_auto,
        "review_rate": (n - n_auto) / n,
        "n_auto_decided": n_auto,
        "coverage": n_auto / n,
        "auto_accuracy": n_auto_correct / n_auto if n_auto else None,
        "auto_accuracy_ci95": wilson_interval(n_auto_correct, n_auto) if n_auto else None,
        "overall_accuracy": float(np.asarray(correct).mean()),
    }


def binary_decision_metrics(y_true: np.ndarray, flagged: np.ndarray) -> dict:
    """Defect-detection decision quality: recall on defects and false-alarm rate on good images."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(flagged).astype(bool)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    tp = int((flagged & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    return {
        "n_defective": n_pos,
        "n_good": n_neg,
        "defects_caught": tp,
        "recall": tp / n_pos if n_pos else None,
        "recall_ci95": wilson_interval(tp, n_pos) if n_pos else None,
        "good_flagged": fp,
        "false_alarm_rate": fp / n_neg if n_neg else None,
        "false_alarm_ci95": wilson_interval(fp, n_neg) if n_neg else None,
        "accuracy": (tp + (n_neg - fp)) / (n_pos + n_neg),
    }


def base_rate(y_true: np.ndarray) -> float:
    """Share of positive (defective) images: the AUPRC of an uninformative scorer."""
    return float(np.mean(np.asarray(y_true) == 1))


def select_anomaly_threshold(scores: np.ndarray, labels: np.ndarray, quantile: float = 0.95) -> float:
    """Anomaly-score threshold from known-good *validation* images only (their `quantile` score)."""
    normal = np.asarray(scores)[np.asarray(labels) == 0]
    if len(normal) == 0:
        raise ValueError("need at least one known-good validation image")
    return float(np.quantile(normal, quantile))


def assert_same_ids(ids_a: Sequence[str], ids_b: Sequence[str], what: str = "images") -> None:
    if list(ids_a) != list(ids_b):
        raise ValueError(f"{what}: the two models were not scored on identical image IDs, in order")


def assert_disjoint(pool_ids: Sequence[str], test_ids: Sequence[str]) -> None:
    overlap = set(pool_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"{len(overlap)} pool images are also in the held-out test set")
