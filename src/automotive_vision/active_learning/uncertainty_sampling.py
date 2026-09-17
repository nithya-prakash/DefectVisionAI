"""Uncertainty sampling: select the unlabeled examples the current model is least sure about.

Uncertainty here is normalized predictive entropy over the current model's class
probabilities, the same quantity used in models/uncertainty.py — just applied to the
probability array directly since the active-learning head (sklearn LogisticRegression)
exposes predict_proba rather than logits.
"""

import numpy as np


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Normalized entropy in [0, 1] per row of a (N, num_classes) probability matrix."""
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    return entropy / np.log(probs.shape[1])


class UncertaintyAcquisition:
    name = "uncertainty"

    def select(
        self,
        unlabeled_indices: np.ndarray,
        embeddings: np.ndarray,
        probs: np.ndarray | None,
        labeled_indices: np.ndarray,
        n_select: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if probs is None:
            raise ValueError("UncertaintyAcquisition requires model probabilities.")
        n_select = min(n_select, len(unlabeled_indices))
        entropy = predictive_entropy(probs[unlabeled_indices])
        top = np.argsort(entropy)[::-1][:n_select]
        return unlabeled_indices[top]
