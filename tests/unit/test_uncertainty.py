import numpy as np
import pytest
import torch
from torch import nn

from automotive_vision.models.uncertainty import (
    apply_temperature,
    mc_dropout_predict,
    softmax_uncertainty,
)

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


def test_mc_dropout_predict_returns_expected_shapes_and_restores_eval_mode():
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(p=0.5), nn.Linear(8, len(CLASS_NAMES)))
    model.eval()
    x = torch.randn(3, 4)

    result = mc_dropout_predict(model, x, n_passes=10)

    assert result["mean_probs"].shape == (3, len(CLASS_NAMES))
    assert result["predictive_variance"].shape == (3,)
    assert result["mean_entropy"].shape == (3,)
    assert np.allclose(result["mean_probs"].sum(axis=1), 1.0, atol=1e-5)
    assert model.training is False  # restored to the mode it was called in


def test_apply_temperature_softens_without_changing_the_prediction() -> None:
    logits = np.array([2.0, 0.5, 0.1])
    names = ["a", "b", "c"]
    raw = softmax_uncertainty(logits, names)
    soft = softmax_uncertainty(apply_temperature(logits, 1.4568), names)
    assert soft.label == raw.label
    assert soft.probability < raw.probability
    assert np.array_equal(apply_temperature(logits, 1.0), logits)
    with pytest.raises(ValueError):
        apply_temperature(logits, 0.0)
