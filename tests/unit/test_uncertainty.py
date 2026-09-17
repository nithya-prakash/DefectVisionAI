import numpy as np

from automotive_vision.models.uncertainty import softmax_uncertainty

CLASS_NAMES = ["bent", "color", "flip", "good", "scratch"]


def test_confident_prediction_has_low_entropy_and_no_review():
    logits = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
    result = softmax_uncertainty(logits, CLASS_NAMES, confidence_threshold=0.7)
    assert result.label == "bent"
    assert result.probability > 0.99
    assert result.entropy < 0.05
    assert result.needs_review is False


def test_uniform_prediction_has_max_entropy_and_needs_review():
    logits = np.zeros(5)
    result = softmax_uncertainty(logits, CLASS_NAMES, confidence_threshold=0.7)
    assert abs(result.entropy - 1.0) < 1e-6
    assert result.probability == 0.2
    assert result.needs_review is True


def test_margin_is_difference_between_top_two_probabilities():
    logits = np.array([2.0, 1.0, 0.0, 0.0, 0.0])
    result = softmax_uncertainty(logits, CLASS_NAMES)
    probs = np.exp(logits) / np.exp(logits).sum()
    expected_margin = np.sort(probs)[::-1][0] - np.sort(probs)[::-1][1]
    assert abs(result.margin - expected_margin) < 1e-6


def test_confidence_threshold_controls_review_flag():
    logits = np.array([1.0, 0.9, 0.0, 0.0, 0.0])  # close call, moderate confidence
    lenient = softmax_uncertainty(logits, CLASS_NAMES, confidence_threshold=0.1)
    strict = softmax_uncertainty(logits, CLASS_NAMES, confidence_threshold=0.99)
    assert lenient.needs_review is False
    assert strict.needs_review is True
