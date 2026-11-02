"""Near-duplicate detection between dataset splits.

The welding dataset is a Roboflow-style export: augmented copies of one source photo can land in
different splits, so a test image may have a near-copy in training. A 16x16 average hash is
coarse, but identical hashes (distance 0) are strong evidence of a copy, and the distance to the
nearest training image lets us define a leakage-reduced test subset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HASH_SIZE = 16


def ahash(path: Path | str) -> np.ndarray:
    im = Image.open(path).convert("L").resize((HASH_SIZE, HASH_SIZE), Image.BILINEAR)
    a = np.asarray(im, dtype=float)
    return (a > a.mean()).flatten()


def nearest_train_distance(images: list[Path], train_images: list[Path]) -> np.ndarray:
    """Smallest Hamming distance (0 to 256) from each image to any training image."""
    tr = np.stack([ahash(p) for p in train_images])
    return np.array([int((tr != ahash(p)).sum(1).min()) for p in images])


def clean_subset(images: list[Path], distances: np.ndarray, min_distance: int) -> list[Path]:
    """Images whose nearest training image is strictly farther than `min_distance`."""
    return [p for p, d in zip(images, distances) if d > min_distance]
