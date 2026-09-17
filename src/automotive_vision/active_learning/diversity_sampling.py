"""Diversity sampling via greedy farthest-point selection in embedding space.

At each step, picks the unlabeled example whose minimum distance to everything already
labeled is largest — a standard, simple coreset-style heuristic for spreading selected
points across the embedding space rather than clustering in one region.
"""

import numpy as np
from scipy.spatial.distance import cdist


def min_distance_to_labeled(embeddings: np.ndarray, unlabeled_indices: np.ndarray, labeled_indices: np.ndarray) -> np.ndarray:
    if len(labeled_indices) == 0:
        return np.full(len(unlabeled_indices), np.inf)
    dists = cdist(embeddings[unlabeled_indices], embeddings[labeled_indices])
    return dists.min(axis=1)


class DiversityAcquisition:
    name = "diversity"

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
        remaining = list(unlabeled_indices)
        current_labeled = list(labeled_indices)
        selected = []

        for _ in range(n_select):
            remaining_arr = np.array(remaining)
            dists = min_distance_to_labeled(embeddings, remaining_arr, np.array(current_labeled))
            best = remaining_arr[np.argmax(dists)]
            selected.append(best)
            remaining.remove(best)
            current_labeled.append(best)

        return np.array(selected)
