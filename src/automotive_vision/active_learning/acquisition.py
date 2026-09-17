"""Shared acquisition-function interface and the random-sampling baseline.

All acquisition strategies share this signature:

    select(unlabeled_indices, embeddings, probs, labeled_indices, n_select, rng) -> indices to label next

so simulation.py can swap strategies without special-casing any of them.
"""

import numpy as np


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


class RandomAcquisition:
    """Baseline: select uniformly at random from the unlabeled pool."""

    name = "random"

    def select(
        self,
        unlabeled_indices: np.ndarray,
        embeddings: np.ndarray,
        probs: np.ndarray | None,
        labeled_indices: np.ndarray,
        n_select: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_select = min(n_select, len(unlabeled_indices))
        return rng.choice(unlabeled_indices, size=n_select, replace=False)
