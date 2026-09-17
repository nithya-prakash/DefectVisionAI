"""Active-learning simulation loop.

Human-in-the-loop simulation (see README "Active learning" section for the full
disclosure): ground-truth labels from MVTec AD stand in for a human reviewer. Labels
are hidden from the acquisition function until a sample is selected, at which point its
true label is "revealed" and it moves from the unlabeled pool into the labeled set.

Compute design: fitting a full CNN from scratch every round, for every strategy, for
every repeat seed, would be far too slow on a laptop (4 strategies x n_repeats x rounds
retrains). Instead this uses the SAME frozen ResNet18 embeddings the anomaly detector
uses (models/embeddings.py), with a lightweight multinomial logistic-regression head
retrained each round — a standard, fast substitute for the full classifier
specifically for this many-iteration experiment. The headline supervised-classification
numbers reported elsewhere in this project come from the full fine-tuned
DefectClassifier (scripts/train.py), not from this lightweight AL head.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class ActiveLearningResult:
    strategy: str
    seed: int
    n_labeled: list[int] = field(default_factory=list)
    labeled_fraction: list[float] = field(default_factory=list)
    macro_f1: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)


def stratified_initial_sample(labels: np.ndarray, n_initial: int, rng: np.random.Generator) -> np.ndarray:
    """Picks n_initial indices, stratified across classes, guaranteeing every class
    present in `labels` gets at least one labeled example."""
    classes = np.unique(labels)
    per_class = max(1, n_initial // len(classes))
    selected = []
    for c in classes:
        class_indices = np.where(labels == c)[0]
        n = min(per_class, len(class_indices))
        selected.extend(rng.choice(class_indices, size=n, replace=False))
    remaining_budget = n_initial - len(selected)
    if remaining_budget > 0:
        leftover = np.setdiff1d(np.arange(len(labels)), selected)
        extra = rng.choice(leftover, size=min(remaining_budget, len(leftover)), replace=False)
        selected.extend(extra)
    return np.array(selected)


def _full_class_probs(head: LogisticRegression, embeddings: np.ndarray, num_classes: int) -> np.ndarray:
    """Pads predict_proba's output to a fixed (N, num_classes) array indexed 0..num_classes-1,
    in case the current labeled set doesn't yet include every class."""
    probs = head.predict_proba(embeddings)
    full = np.zeros((embeddings.shape[0], num_classes))
    for col, class_idx in enumerate(head.classes_):
        full[:, class_idx] = probs[:, col]
    return full


def run_simulation(
    pool_embeddings: np.ndarray,
    pool_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int,
    strategy,
    initial_labeled_fraction: float,
    acquisition_size: int,
    rounds: int,
    seed: int,
) -> ActiveLearningResult:
    rng = np.random.default_rng(seed)
    n_pool = len(pool_labels)
    all_indices = np.arange(n_pool)

    n_initial = max(num_classes, round(initial_labeled_fraction * n_pool))
    labeled_indices = stratified_initial_sample(pool_labels, n_initial, rng)
    unlabeled_indices = np.setdiff1d(all_indices, labeled_indices)

    result = ActiveLearningResult(strategy=strategy.name, seed=seed)

    for round_idx in range(rounds + 1):
        head = LogisticRegression(max_iter=2000)
        head.fit(pool_embeddings[labeled_indices], pool_labels[labeled_indices])

        test_preds = head.predict(test_embeddings)
        macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        accuracy = accuracy_score(test_labels, test_preds)

        result.n_labeled.append(int(len(labeled_indices)))
        result.labeled_fraction.append(float(len(labeled_indices) / n_pool))
        result.macro_f1.append(float(macro_f1))
        result.accuracy.append(float(accuracy))

        if round_idx == rounds or len(unlabeled_indices) == 0:
            break

        probs = _full_class_probs(head, pool_embeddings, num_classes)
        selected = strategy.select(
            unlabeled_indices, pool_embeddings, probs, labeled_indices, acquisition_size, rng
        )
        labeled_indices = np.concatenate([labeled_indices, selected])
        unlabeled_indices = np.setdiff1d(unlabeled_indices, selected)

    return result
