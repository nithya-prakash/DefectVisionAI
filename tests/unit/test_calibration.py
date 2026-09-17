import numpy as np

from automotive_vision.training.calibration import expected_calibration_error, reliability_diagram_data


def test_perfectly_calibrated_predictions_have_near_zero_ece():
    rng = np.random.default_rng(0)
    confidences = rng.uniform(0.5, 1.0, size=2000)
    # Correctness sampled to exactly match confidence => perfect calibration by construction.
    correct = (rng.uniform(0, 1, size=2000) < confidences).astype(int)
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece < 0.05


def test_badly_calibrated_predictions_have_high_ece():
    # Always 99% confident but only right half the time.
    confidences = np.full(200, 0.99)
    correct = np.array([1, 0] * 100)
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece > 0.4


def test_reliability_diagram_bin_counts_sum_to_total():
    confidences = np.linspace(0, 1, 50)
    correct = np.ones(50, dtype=int)
    diagram = reliability_diagram_data(confidences, correct, n_bins=10)
    assert sum(diagram["bin_counts"]) == 50
