"""Hybrid acquisition: alpha * normalized_uncertainty + (1 - alpha) * normalized_diversity.

This linear combination is a common, simple way to trade off the two signals — not a
claim that it is scientifically optimal (alpha is a config knob; see
configs/active_learning.yaml). Diversity is recomputed greedily within the batch (each
newly selected point updates the "already labeled" set used for the next point's
diversity score), while uncertainty is computed once per round from the current model.
"""

import numpy as np

from automotive_vision.active_learning.acquisition import min_max_normalize
from automotive_vision.active_learning.diversity_sampling import min_distance_to_labeled
from automotive_vision.active_learning.uncertainty_sampling import predictive_entropy


class HybridAcquisition:
    name = "hybrid"

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

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
            raise ValueError("HybridAcquisition requires model probabilities.")
        n_select = min(n_select, len(unlabeled_indices))
        entropy = predictive_entropy(probs[unlabeled_indices])
        uncertainty_norm = min_max_normalize(entropy)

        remaining = list(unlabeled_indices)
        remaining_uncertainty = dict(zip(unlabeled_indices.tolist(), uncertainty_norm.tolist()))
        current_labeled = list(labeled_indices)
        selected = []

        for _ in range(n_select):
            remaining_arr = np.array(remaining)
            dists = min_distance_to_labeled(embeddings, remaining_arr, np.array(current_labeled))
            diversity_norm = min_max_normalize(dists) if np.isfinite(dists).all() else np.zeros_like(dists)
            uncertainty_scores = np.array([remaining_uncertainty[i] for i in remaining_arr])
            combined = self.alpha * uncertainty_scores + (1 - self.alpha) * diversity_norm

            best = remaining_arr[np.argmax(combined)]
            selected.append(best)
            remaining.remove(best)
            current_labeled.append(best)

        return np.array(selected)
