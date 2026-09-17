"""Classical (non-deep-learning) anomaly-detection baseline.

Purpose: a reference point to contextualize the deep-learning anomaly detector's
results, not a competitive method in its own right. Uses hand-crafted texture/edge
statistics per whole image (no learned features), and scores anomalies by nearest-
neighbor distance to a bank of normal-image feature vectors — the same nearest-
neighbor principle as the PatchCore-style detector, but on classical features and at
whole-image (not patch) granularity, so it produces no localization/pixel map.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern


@dataclass
class ClassicalScores:
    image_scores: np.ndarray
    labels: np.ndarray
    paths: list[str]
    defect_types: list[str]


def extract_classical_features(image_path: str, size: int = 128) -> np.ndarray:
    """Grayscale intensity histogram + LBP texture histogram + Sobel edge-magnitude
    histogram, concatenated into one fixed-length feature vector per image."""
    image = Image.open(image_path).convert("L").resize((size, size))
    arr = np.array(image, dtype=np.float32) / 255.0

    intensity_hist, _ = np.histogram(arr, bins=32, range=(0, 1))
    intensity_hist = intensity_hist / (intensity_hist.sum() + 1e-8)

    lbp = local_binary_pattern((arr * 255).astype(np.uint8), P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-8)

    gy, gx = np.gradient(arr)
    edge_magnitude = np.sqrt(gx**2 + gy**2)
    edge_hist, _ = np.histogram(edge_magnitude, bins=16, range=(0, edge_magnitude.max() + 1e-8))
    edge_hist = edge_hist / (edge_hist.sum() + 1e-8)

    return np.concatenate([intensity_hist, lbp_hist, edge_hist]).astype(np.float32)


class ClassicalAnomalyBaseline:
    def __init__(self, feature_size: int = 128):
        self.feature_size = feature_size
        self.normal_features: np.ndarray | None = None

    def fit(self, normal_image_paths: list[str]) -> None:
        self.normal_features = np.stack(
            [extract_classical_features(p, self.feature_size) for p in normal_image_paths]
        )

    def score_dataset(self, records) -> ClassicalScores:
        if self.normal_features is None:
            raise RuntimeError("Call fit() before score_dataset().")

        image_scores, labels, paths, defect_types = [], [], [], []
        for record in records:
            feat = extract_classical_features(str(record.path), self.feature_size)
            dist = np.linalg.norm(self.normal_features - feat[None, :], axis=1).min()
            image_scores.append(dist)
            labels.append(float(record.is_anomalous))
            paths.append(str(record.path))
            defect_types.append(record.defect_type)

        return ClassicalScores(
            image_scores=np.array(image_scores),
            labels=np.array(labels),
            paths=paths,
            defect_types=defect_types,
        )
