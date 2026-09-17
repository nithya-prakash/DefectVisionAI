"""Calibration: is the classifier's confidence actually meaningful?

Implements Expected Calibration Error (ECE), reliability-diagram binning, and
temperature scaling (Guo et al., 2017) — a single scalar T that rescales logits
(logits / T) to better match empirical accuracy, fit on a held-out validation set by
minimizing NLL, never on the test set.
"""

import numpy as np
import torch
import torch.nn.functional as F


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Standard binned ECE: weighted average |accuracy - confidence| per confidence bin."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_confidence = confidences[in_bin].mean()
        bin_accuracy = correct[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def reliability_diagram_data(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> dict:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accuracies, bin_counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        bin_centers.append((lo + hi) / 2)
        bin_accuracies.append(float(correct[in_bin].mean()) if in_bin.sum() > 0 else np.nan)
        bin_counts.append(int(in_bin.sum()))
    return {"bin_centers": bin_centers, "bin_accuracies": bin_accuracies, "bin_counts": bin_counts}


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
    """Fits a single scalar temperature T > 0 by minimizing NLL on (logits, labels),
    intended to be a held-out validation set, not the test set."""
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.clamp(min=1e-3).item())
