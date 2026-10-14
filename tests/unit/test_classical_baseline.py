"""Classical (hand-crafted feature) anomaly baseline on tiny synthetic images."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from automotive_vision.models.classical_baseline import (
    ClassicalAnomalyBaseline,
    extract_classical_features,
)

SIZE = 32


@dataclass
class Record:
    path: Path
    is_anomalous: bool
    defect_type: str


def make_image(path: Path, kind: str, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    base = np.full((64, 64), 120.0)
    if kind == "normal":
        arr = base + rng.normal(0, 2, base.shape)
    else:  # strong stripes and a bright block: very different texture and edge statistics
        arr = base + 90 * (np.arange(64)[None, :] % 8 < 4) - 45
        arr[10:30, 10:30] = 255
    Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).save(path)
    return path


def test_features_have_a_fixed_shape_are_finite_and_deterministic(tmp_path) -> None:
    path = make_image(tmp_path / "n.png", "normal", 0)
    a = extract_classical_features(str(path), SIZE)
    b = extract_classical_features(str(path), SIZE)
    assert a.shape == (32 + 10 + 16,) and a.dtype == np.float32
    assert np.isfinite(a).all()
    assert np.array_equal(a, b)
    assert a[:32].sum() == pytest.approx(1.0, abs=1e-4)  # intensity histogram is normalised


def test_a_defective_looking_image_scores_higher_than_a_held_out_normal_one(tmp_path) -> None:
    normals = [make_image(tmp_path / f"n{i}.png", "normal", i) for i in range(6)]
    held_out = make_image(tmp_path / "held.png", "normal", 99)
    defect = make_image(tmp_path / "d.png", "defect", 1)
    baseline = ClassicalAnomalyBaseline(feature_size=SIZE)
    baseline.fit([str(p) for p in normals])
    scores = baseline.score_dataset(
        [Record(held_out, False, "good"), Record(defect, True, "bent")]
    )
    assert scores.image_scores[1] > scores.image_scores[0]
    assert scores.labels.tolist() == [0.0, 1.0]
    assert scores.defect_types == ["good", "bent"]


def test_scoring_before_fitting_is_an_error(tmp_path) -> None:
    record = Record(make_image(tmp_path / "x.png", "normal", 0), False, "good")
    with pytest.raises(RuntimeError):
        ClassicalAnomalyBaseline(SIZE).score_dataset([record])


def test_malformed_inputs_raise_clear_errors(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_classical_features(str(tmp_path / "missing.png"), SIZE)
    not_image = tmp_path / "text.png"
    not_image.write_text("not an image")
    with pytest.raises(UnidentifiedImageError):
        extract_classical_features(str(not_image), SIZE)
    with pytest.raises(ValueError):
        ClassicalAnomalyBaseline(SIZE).fit([])
