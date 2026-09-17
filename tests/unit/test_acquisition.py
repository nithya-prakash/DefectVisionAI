import numpy as np

from automotive_vision.active_learning.acquisition import RandomAcquisition, min_max_normalize
from automotive_vision.active_learning.diversity_sampling import DiversityAcquisition
from automotive_vision.active_learning.hybrid_sampling import HybridAcquisition
from automotive_vision.active_learning.uncertainty_sampling import (
    UncertaintyAcquisition,
    predictive_entropy,
)


def test_min_max_normalize_maps_to_unit_range():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    normalized = min_max_normalize(values)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_min_max_normalize_handles_constant_input():
    values = np.array([5.0, 5.0, 5.0])
    normalized = min_max_normalize(values)
    assert np.allclose(normalized, 0.0)


def test_random_acquisition_selects_requested_count_from_unlabeled_only():
    rng = np.random.default_rng(0)
    unlabeled = np.array([2, 3, 5, 7, 9])
    selected = RandomAcquisition().select(unlabeled, None, None, np.array([]), n_select=3, rng=rng)
    assert len(selected) == 3
    assert set(selected).issubset(set(unlabeled))
    assert len(set(selected)) == 3  # no duplicates


def test_predictive_entropy_is_maximal_for_uniform_distribution():
    uniform_probs = np.full((1, 5), 0.2)
    peaked_probs = np.array([[0.96, 0.01, 0.01, 0.01, 0.01]])
    assert predictive_entropy(uniform_probs)[0] > predictive_entropy(peaked_probs)[0]
    assert abs(predictive_entropy(uniform_probs)[0] - 1.0) < 1e-6


def test_uncertainty_acquisition_picks_highest_entropy_samples():
    rng = np.random.default_rng(0)
    unlabeled = np.array([0, 1, 2])
    probs = np.array(
        [
            [0.9, 0.1],  # low entropy
            [0.5, 0.5],  # high entropy
            [0.6, 0.4],  # medium entropy
        ]
    )
    selected = UncertaintyAcquisition().select(unlabeled, None, probs, np.array([]), n_select=1, rng=rng)
    assert selected[0] == 1


def test_diversity_acquisition_prefers_farthest_point():
    unlabeled = np.array([0, 1, 2])
    embeddings = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])
    labeled = np.array([0])
    rng = np.random.default_rng(0)
    selected = DiversityAcquisition().select(unlabeled[1:], embeddings, None, labeled, n_select=1, rng=rng)
    assert selected[0] == 2  # index 2 is far from the only labeled point (index 0)


def test_hybrid_acquisition_returns_requested_count_without_duplicates():
    rng = np.random.default_rng(0)
    unlabeled = np.array([0, 1, 2, 3])
    embeddings = np.random.default_rng(1).normal(size=(4, 3))
    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.6, 0.4], [0.3, 0.7]])
    selected = HybridAcquisition(alpha=0.5).select(unlabeled, embeddings, probs, np.array([]), n_select=2, rng=rng)
    assert len(selected) == 2
    assert len(set(selected.tolist())) == 2
