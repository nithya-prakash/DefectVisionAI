"""Reusable uncertainty estimation for classifier predictions.

Distinguishes "model is confident" from "model is unsure and a human should review
this" using standard, well-understood quantities computed directly from the softmax
output (confidence, entropy, margin), plus optional Monte Carlo dropout for a second,
independent uncertainty signal (predictive variance across stochastic forward passes).

Review thresholds are configuration, not something this module claims to have
"optimally" tuned — see configs/inference.yaml and reports/evaluation_report.md for how
they were actually chosen (via the calibration/threshold sweep in
src/automotive_vision/training/evaluate.py), not an arbitrary constant.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class UncertaintyResult:
    label_idx: int
    label: str
    probability: float
    entropy: float
    margin: float
    needs_review: bool


def softmax_uncertainty(
    logits: np.ndarray,
    class_names: list[str],
    confidence_threshold: float = 0.7,
    entropy_threshold: float | None = None,
) -> UncertaintyResult:
    """Confidence/entropy/margin from a single softmax distribution over one example.

    entropy is normalized to [0, 1] by dividing by log(num_classes), so the threshold
    is comparable across different numbers of classes.
    """
    probs = _softmax_np(logits)
    sorted_probs = np.sort(probs)[::-1]
    top1, top2 = sorted_probs[0], sorted_probs[1] if len(sorted_probs) > 1 else 0.0

    entropy = -np.sum(probs * np.log(probs + 1e-12))
    normalized_entropy = float(entropy / np.log(len(class_names)))
    margin = float(top1 - top2)
    label_idx = int(np.argmax(probs))

    needs_review = top1 < confidence_threshold
    if entropy_threshold is not None:
        needs_review = needs_review or normalized_entropy > entropy_threshold

    return UncertaintyResult(
        label_idx=label_idx,
        label=class_names[label_idx],
        probability=float(top1),
        entropy=normalized_entropy,
        margin=margin,
        needs_review=bool(needs_review),
    )


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


@torch.no_grad()
def mc_dropout_predict(model, x: torch.Tensor, n_passes: int = 20) -> dict:
    """Monte Carlo dropout: run n_passes stochastic forward passes (dropout active) and
    return the mean prediction plus predictive uncertainty (mean entropy of individual
    passes, and variance of the predicted class probability across passes).

    Requires the model to contain at least one nn.Dropout layer; otherwise every pass
    is identical and the variance will be ~0 (not an error, just an uninformative signal).
    """
    was_training = model.training
    model.train()  # enables dropout; batchnorm is not used in eval-mode-sensitive ways here since we don't update running stats without backward()

    all_probs = []
    for _ in range(n_passes):
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())

    if not was_training:
        model.eval()

    stacked = np.stack(all_probs, axis=0)  # (n_passes, batch, num_classes)
    mean_probs = stacked.mean(axis=0)
    pred_variance = stacked.var(axis=0).sum(axis=-1)  # total variance per example
    per_pass_entropy = -np.sum(stacked * np.log(stacked + 1e-12), axis=-1)
    mean_entropy = per_pass_entropy.mean(axis=0)

    return {
        "mean_probs": mean_probs,
        "predictive_variance": pred_variance,
        "mean_entropy": mean_entropy,
    }
