import numpy as np
from PIL import Image

from welddet.leakage import ahash, clean_subset, nearest_train_distance


def _img(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(path)
    return path


def test_identical_images_have_distance_zero(tmp_path):
    a = _img(tmp_path / "a.png", 1)
    b = tmp_path / "b.png"
    b.write_bytes(a.read_bytes())
    assert nearest_train_distance([b], [a])[0] == 0


def test_different_images_are_far(tmp_path):
    a = _img(tmp_path / "a.png", 1)
    c = _img(tmp_path / "c.png", 2)
    assert nearest_train_distance([c], [a])[0] > 40


def test_hash_is_robust_to_small_brightness_change(tmp_path):
    a = _img(tmp_path / "a.png", 3)
    arr = np.asarray(Image.open(a), dtype=int)
    Image.fromarray(np.clip(arr + 10, 0, 255).astype(np.uint8)).save(tmp_path / "bright.png")
    assert (ahash(a) != ahash(tmp_path / "bright.png")).sum() < 25


def test_clean_subset_is_strictly_greater(tmp_path):
    ps = [tmp_path / n for n in "xyz"]
    out = clean_subset(ps, np.array([0, 4, 5]), 4)
    assert out == [ps[2]]
