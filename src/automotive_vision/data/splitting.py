"""Reproducible, leakage-free splits for the two tracks.

Anomaly detection uses MVTec's own official protocol (train=normal only, test=official
test folder) — no re-splitting needed there.

Supervised classification / active learning need labeled defect examples, which only
exist in the official *test* folder (220 train images are all "good", unlabeled by
defect type). So we carve our own stratified split out of that labeled pool. This is a
deliberate deviation from the anomaly-detection protocol, made explicit here and in the
README rather than silently reusing "test" to mean two different things.
"""

from sklearn.model_selection import train_test_split

from automotive_vision.data.dataset import ImageRecord


def anomaly_detection_split(
    records: list[ImageRecord],
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Official protocol: normal-only train, full test (good + all defect types)."""
    train = [r for r in records if r.official_split == "train"]
    test = [r for r in records if r.official_split == "test"]
    return train, test


def labeled_pool_split(
    records: list[ImageRecord],
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Stratified split of the officially-labeled (test-folder) images into an
    active-learning pool and a fixed held-out evaluation set.

    Only records from the official test split carry defect-type labels, so this
    function operates on those alone; it raises if given anything else.
    """
    labeled = [r for r in records if r.official_split == "test"]
    if len(labeled) != len(records):
        raise ValueError("labeled_pool_split expects only officially-labeled (test-split) records")

    labels = [r.defect_type for r in labeled]
    pool, held_out = train_test_split(
        labeled, test_size=test_size, random_state=seed, stratify=labels
    )
    return pool, held_out


def stratified_split(
    records: list[ImageRecord],
    test_size: float,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Generic stratified-by-defect_type split, reused for train/val carving."""
    labels = [r.defect_type for r in records]
    return train_test_split(records, test_size=test_size, random_state=seed, stratify=labels)
