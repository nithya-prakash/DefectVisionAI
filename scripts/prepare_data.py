#!/usr/bin/env python3
"""Validate the downloaded dataset and print the exact splits every other script uses.

Usage:
    python scripts/prepare_data.py

Verifies every image is readable (catches a corrupt/partial download early) and
prints the real anomaly-detection and labeled-pool splits, so their sizes are always
taken from this scan rather than restated from memory anywhere else in the project.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from automotive_vision.data.dataset import scan_mvtec_category
from automotive_vision.data.splitting import (
    anomaly_detection_split,
    labeled_pool_split,
)
from automotive_vision.utils.config import load_config
from automotive_vision.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    args = parser.parse_args()

    config = load_config(args.base_config, args.training_config)
    category_dir = config["dataset"]["category_dir"]
    records = scan_mvtec_category(category_dir)

    logger.info(f"Validating {len(records)} images are readable...")
    corrupt = []
    for record in records:
        try:
            Image.open(record.path).verify()
        except Exception:  # noqa: BLE001 -- PIL raises varied exception types for corrupt images
            corrupt.append(record.path)
    if corrupt:
        raise RuntimeError(f"{len(corrupt)} corrupt/unreadable images found: {corrupt[:5]}...")
    logger.info("All images readable.")

    ad_train, ad_test = anomaly_detection_split(records)
    logger.info(f"Anomaly-detection split: {len(ad_train)} train (normal), {len(ad_test)} test")

    labeled_records = [r for r in records if r.official_split == "test"]
    pool, held_out = labeled_pool_split(
        labeled_records, test_size=config["training"]["held_out_fraction"], seed=config["seed"]
    )
    logger.info(f"Labeled pool (for supervised + active learning): {len(pool)} pool, {len(held_out)} held-out test")

    by_type = {}
    for record in records:
        by_type[record.defect_type] = by_type.get(record.defect_type, 0) + 1
    logger.info(f"Per-defect-type counts: {by_type}")


if __name__ == "__main__":
    main()
