"""PatchCore-lite scoring on tiny synthetic images (random-weight backbone, no downloads)."""

import numpy as np
import pytest
import torch

from automotive_vision.models.anomaly_detector import PatchCoreLite

SIZE = 64


def normal_batch(seed: int, n: int = 4) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"image": 0.1 * torch.randn(n, 3, SIZE, SIZE, generator=g)}


@pytest.fixture
def detector(offline_resnet):
    det = PatchCoreLite(image_size=SIZE, memory_bank_size=400, seed=3)
    det.fit([normal_batch(0), normal_batch(1)])
    return det


def test_fit_builds_a_capped_memory_bank(detector) -> None:
    assert detector.memory_bank is not None
    assert detector.memory_bank.shape[0] == 400  # 8 images x 64 patches = 512, capped to 400
    assert detector.memory_bank.shape[1] == 128 + 256  # layer2 + layer3 channels
    assert detector.patch_grid_size == SIZE // 8


def test_a_defective_looking_image_scores_higher_than_a_held_out_normal_one(detector) -> None:
    normal = normal_batch(7, n=1)["image"][0]
    defect = normal.clone()
    defect[:, 16:48, 16:48] = 4.0  # large saturated block
    normal_score, normal_map = detector.score_image(normal)
    defect_score, defect_map = detector.score_image(defect)
    assert defect_score > normal_score
    assert normal_map.shape == (SIZE, SIZE) == defect_map.shape
    assert (defect_map >= 0).all() and np.isfinite(defect_map).all()


def test_scores_are_deterministic(detector) -> None:
    image = normal_batch(5, n=1)["image"][0]
    assert detector.score_image(image)[0] == detector.score_image(image)[0]


def test_save_and_load_round_trip_gives_identical_scores(detector, tmp_path) -> None:
    path = tmp_path / "bank.pt"
    detector.save(path)
    loaded = PatchCoreLite.load(path)
    image = normal_batch(6, n=1)["image"][0]
    assert loaded.score_image(image)[0] == pytest.approx(detector.score_image(image)[0])
    assert torch.equal(loaded.memory_bank, detector.memory_bank)


def test_using_the_detector_before_fit_or_load_raises(offline_resnet, tmp_path) -> None:
    fresh = PatchCoreLite(image_size=SIZE)
    image = normal_batch(0, n=1)["image"][0]
    with pytest.raises(RuntimeError):
        fresh.score_image(image)
    with pytest.raises(RuntimeError):
        fresh.save(tmp_path / "x.pt")


def test_malformed_input_shapes_are_rejected(detector) -> None:
    with pytest.raises(RuntimeError):
        detector.score_image(torch.zeros(SIZE, SIZE))  # missing the channel dimension
    with pytest.raises(RuntimeError):
        detector.score_image(torch.zeros(1, SIZE, SIZE))  # one channel instead of three
