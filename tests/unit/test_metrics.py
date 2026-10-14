"""Metric calculations on tiny, hand-checkable inputs."""

import numpy as np
import pytest

from automotive_vision.training.evaluate import (
    bootstrap_ci,
    classification_metrics,
    image_level_anomaly_metrics,
    pixel_level_anomaly_metrics,
)

NAMES = ["a", "b", "c", "d"]


def test_classification_metrics_match_a_hand_computed_example() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    m = classification_metrics(y_true, y_pred, NAMES)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["confusion_matrix"][0] == [1, 1, 0, 0]
    assert m["per_class"]["a"]["precision"] == pytest.approx(1 / 2)
    assert m["per_class"]["a"]["recall"] == pytest.approx(1 / 2)
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["b"]["recall"] == pytest.approx(1.0)
    assert m["per_class"]["c"]["support"] == 2


def test_a_class_with_no_samples_does_not_crash_and_scores_zero() -> None:
    m = classification_metrics(np.array([0, 1]), np.array([0, 1]), NAMES)
    assert m["per_class"]["d"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


def test_image_level_metrics_for_perfect_and_uninformative_scores() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    perfect = image_level_anomaly_metrics(labels, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), n_bootstrap=50)
    assert perfect["image_auroc"] == 1.0 and perfect["image_auprc"] == 1.0
    assert perfect["n_anomalous"] == 3 and perfect["n_normal"] == 3
    flat = image_level_anomaly_metrics(labels, np.full(6, 0.5), n_bootstrap=50)
    assert flat["image_auroc"] == pytest.approx(0.5)


def test_image_level_metrics_with_a_single_class_are_undefined_not_a_fake_number() -> None:
    from sklearn.exceptions import UndefinedMetricWarning

    with pytest.warns(UndefinedMetricWarning):
        m = image_level_anomaly_metrics(np.ones(4), np.array([0.1, 0.2, 0.3, 0.4]), n_bootstrap=10)
    assert np.isnan(m["image_auroc"])
    assert np.isnan(m["image_auroc_ci95"][0])


def test_pixel_level_metrics_count_pixels_and_score_a_perfect_map() -> None:
    masks = np.zeros((2, 4, 4))
    masks[1, :2, :2] = 1
    maps = masks * 5.0 + 0.1
    m = pixel_level_anomaly_metrics(masks, maps)
    assert m["n_pixels"] == 32 and m["n_positive_pixels"] == 4
    assert m["pixel_auroc"] == 1.0


def test_bootstrap_interval_is_deterministic_ordered_and_nan_for_one_class() -> None:
    from sklearn.metrics import roc_auc_score

    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.6, 0.9, 0.2, 0.3])
    first = bootstrap_ci(y, s, roc_auc_score, n_bootstrap=200, seed=1)
    assert first == bootstrap_ci(y, s, roc_auc_score, n_bootstrap=200, seed=1)
    assert first[0] <= first[1]
    low, high = bootstrap_ci(np.ones(5), np.arange(5.0), roc_auc_score, n_bootstrap=20)
    assert np.isnan(low) and np.isnan(high)
