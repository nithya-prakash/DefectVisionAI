import inspect

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from automotive_vision.training import review_policy as rp


def test_wilson_interval_known_values() -> None:
    low, high = rp.wilson_interval(8, 10)
    assert low == pytest.approx(0.490, abs=0.005)
    assert high == pytest.approx(0.943, abs=0.005)
    assert rp.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert all(np.isnan(v) for v in rp.wilson_interval(0, 0))


def test_selection_maximises_coverage_subject_to_the_accuracy_target() -> None:
    conf = np.array([0.95, 0.9, 0.85, 0.8, 0.6, 0.55])
    correct = np.array([1, 1, 1, 1, 0, 0])
    result = rp.select_review_threshold(conf, correct, target_accuracy=0.9)
    assert result["selected"]
    assert result["threshold"] == pytest.approx(0.61)  # smallest threshold that drops both errors
    assert result["coverage"] == pytest.approx(4 / 6)
    assert result["auto_accuracy"] == 1.0


def test_selection_reports_infeasible_instead_of_inventing_a_threshold() -> None:
    result = rp.select_review_threshold(np.array([0.9, 0.8, 0.7, 0.6]), np.array([0, 0, 0, 0]))
    assert result["selected"] is False
    assert "threshold" not in result


def test_selection_function_accepts_no_test_set_argument() -> None:
    params = list(inspect.signature(rp.select_review_threshold).parameters)
    assert params == ["confidences", "correct", "target_accuracy", "thresholds"]


def test_coverage_never_increases_with_the_threshold() -> None:
    rng = np.random.default_rng(0)
    sweep = rp.review_sweep(rng.uniform(0.3, 1, 80), rng.integers(0, 2, 80).astype(bool), rp.THRESHOLD_GRID)
    coverage = [row["coverage"] for row in sweep]
    assert coverage == sorted(coverage, reverse=True)


def test_stratified_bootstrap_keeps_class_counts_and_is_deterministic() -> None:
    y = np.array([0] * 7 + [1] * 28)
    score = np.random.default_rng(1).uniform(size=35)
    counts: set[tuple[int, int]] = set()

    def metric(yt: np.ndarray, ys: np.ndarray) -> float:
        counts.add((int((yt == 0).sum()), int((yt == 1).sum())))
        return roc_auc_score(yt, ys)

    first = rp.stratified_bootstrap_ci(y, score, metric, n_bootstrap=200, seed=3)
    assert counts == {(7, 28)}
    assert first == rp.stratified_bootstrap_ci(y, score, metric, n_bootstrap=200, seed=3)


def test_paired_difference_is_zero_for_identical_scores() -> None:
    y = np.array([0] * 5 + [1] * 10)
    score = np.random.default_rng(2).uniform(size=15)
    assert rp.paired_auroc_difference_ci(y, score, score, roc_auc_score, n_bootstrap=100) == (0.0, 0.0)


def test_binary_decision_metrics_and_base_rate() -> None:
    result = rp.binary_decision_metrics(np.array([1, 1, 1, 0, 0]), np.array([1, 1, 0, 1, 0]))
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["false_alarm_rate"] == pytest.approx(1 / 2)
    assert result["accuracy"] == pytest.approx(3 / 5)
    assert rp.base_rate(np.array([1] * 28 + [0] * 7)) == pytest.approx(0.8)


def test_anomaly_threshold_comes_from_known_good_images_only() -> None:
    scores = np.array([1.0, 2.0, 3.0, 100.0, 200.0])
    labels = np.array([0, 0, 0, 1, 1])
    assert rp.select_anomaly_threshold(scores, labels) == pytest.approx(2.9)
    with pytest.raises(ValueError):
        rp.select_anomaly_threshold(scores, np.ones(5))


def test_models_must_be_scored_on_identical_ids_and_pool_and_test_must_be_disjoint() -> None:
    rp.assert_same_ids(["a", "b"], ["a", "b"])
    with pytest.raises(ValueError):
        rp.assert_same_ids(["a", "b"], ["b", "a"])
    with pytest.raises(ValueError):
        rp.assert_disjoint(["a", "b"], ["b", "c"])
    rp.assert_disjoint(["a"], ["b"])
