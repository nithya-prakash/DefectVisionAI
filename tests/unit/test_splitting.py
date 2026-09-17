from pathlib import Path

import pytest

from automotive_vision.data.dataset import ImageRecord
from automotive_vision.data.splitting import anomaly_detection_split, labeled_pool_split, stratified_split


def make_records(n_per_class: int, classes: list[str], official_split: str = "test") -> list[ImageRecord]:
    records = []
    for cls in classes:
        for i in range(n_per_class):
            is_anomalous = cls != "good"
            records.append(ImageRecord(Path(f"{cls}_{i}.png"), official_split, cls, is_anomalous, None))
    return records


def test_anomaly_detection_split_separates_by_official_split():
    train = [ImageRecord(Path(f"t{i}.png"), "train", "good", False, None) for i in range(5)]
    test = make_records(3, ["good", "bent"])
    train_out, test_out = anomaly_detection_split(train + test)
    assert len(train_out) == 5
    assert len(test_out) == 6
    assert all(r.official_split == "train" for r in train_out)


def test_labeled_pool_split_is_stratified_and_disjoint():
    records = make_records(10, ["good", "bent", "scratch"])
    pool, held_out = labeled_pool_split(records, test_size=0.3, seed=42)
    assert len(pool) + len(held_out) == len(records)
    assert set(id(r) for r in pool).isdisjoint(set(id(r) for r in held_out))
    # Roughly proportional per class in the held-out set (10 * 0.3 = 3 per class).
    held_out_bent = [r for r in held_out if r.defect_type == "bent"]
    assert len(held_out_bent) == 3


def test_labeled_pool_split_rejects_train_split_records():
    train_records = [ImageRecord(Path(f"t{i}.png"), "train", "good", False, None) for i in range(5)]
    with pytest.raises(ValueError):
        labeled_pool_split(train_records, test_size=0.3, seed=42)


def test_stratified_split_is_reproducible_with_same_seed():
    records = make_records(10, ["good", "bent", "scratch"])
    a1, a2 = stratified_split(records, test_size=0.2, seed=1)
    b1, b2 = stratified_split(records, test_size=0.2, seed=1)
    assert [r.path for r in a1] == [r.path for r in b1]
    assert [r.path for r in a2] == [r.path for r in b2]
