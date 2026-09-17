"""MVTec-AD-style dataset scanning and PyTorch Dataset classes.

Expects the standard MVTec AD per-category folder layout:

    <category_dir>/
        train/good/*.png
        test/good/*.png
        test/<defect_type>/*.png
        ground_truth/<defect_type>/*_mask.png

This module only reads that layout and builds an in-memory manifest; it does not
download or mutate the data.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class ImageRecord:
    """One image's provenance and label, independent of any particular task/track."""

    path: Path
    official_split: str  # "train" or "test", as laid out on disk
    defect_type: str  # "good" or a defect subtype name (e.g. "bent")
    is_anomalous: bool
    mask_path: Path | None  # ground-truth mask, only present for anomalous test images


def scan_mvtec_category(category_dir: str | Path) -> list[ImageRecord]:
    """Walk a single MVTec-AD category folder and return every image as an ImageRecord.

    Raises FileNotFoundError if the expected train/test structure is missing, so a
    misconfigured path fails loudly instead of silently returning an empty dataset.
    """
    category_dir = Path(category_dir)
    train_good_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    ground_truth_dir = category_dir / "ground_truth"

    if not train_good_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{category_dir}' to contain train/good/ and test/*/ subfolders; "
            "run scripts/download_data.py first."
        )

    records: list[ImageRecord] = []

    for path in sorted(train_good_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            records.append(ImageRecord(path, "train", "good", False, None))

    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        is_anomalous = defect_type != "good"
        mask_dir = ground_truth_dir / defect_type if is_anomalous else None
        for path in sorted(defect_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = None
            if mask_dir is not None:
                candidate = mask_dir / f"{path.stem}_mask{path.suffix}"
                mask_path = candidate if candidate.exists() else None
            records.append(ImageRecord(path, "test", defect_type, is_anomalous, mask_path))

    return records


class AnomalyDetectionDataset(Dataset):
    """Official MVTec-AD protocol: images + binary anomaly label + optional pixel mask.

    Used by the anomaly-detection track. Training subset should be normal-only.
    """

    def __init__(self, records: list[ImageRecord], transform, mask_size: int | None = None):
        self.records = records
        self.transform = transform
        self.mask_size = mask_size or 224

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record.path).convert("RGB")
        image_tensor = self.transform(image)

        if record.mask_path is not None:
            mask = Image.open(record.mask_path).convert("L").resize(
                (self.mask_size, self.mask_size), Image.NEAREST
            )
            mask_tensor = torch.from_numpy((np.array(mask) > 127).astype(np.float32))
        else:
            mask_tensor = torch.zeros((self.mask_size, self.mask_size), dtype=torch.float32)

        return {
            "image": image_tensor,
            "label": torch.tensor(float(record.is_anomalous)),
            "mask": mask_tensor,
            "defect_type": record.defect_type,
            "path": str(record.path),
        }


class ClassificationDataset(Dataset):
    """Supervised multi-class classification dataset built from a custom manifest split.

    label_to_idx maps defect_type strings (including "good") to class indices.
    """

    def __init__(self, records: list[ImageRecord], label_to_idx: dict[str, int], transform):
        self.records = records
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(record.path).convert("RGB")
        image_tensor = self.transform(image)
        label_idx = self.label_to_idx[record.defect_type]
        return {
            "image": image_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "defect_type": record.defect_type,
            "path": str(record.path),
        }
