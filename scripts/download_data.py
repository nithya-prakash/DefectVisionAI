#!/usr/bin/env python3
"""Download the MVTec AD category used by this project.

Usage:
    python scripts/download_data.py [--config configs/base.yaml]

Downloads via the Hugging Face Hub mirror configured in configs/base.yaml
(dataset.hf_repo_id), which preserves MVTec AD's original folder layout
(train/good, test/<type>, ground_truth/<type>). See data/README.md for
licensing and provenance notes on this mirror vs. the official MVTec source.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from automotive_vision.data.dataset import scan_mvtec_category  # noqa: E402
from automotive_vision.utils.config import load_config  # noqa: E402
from automotive_vision.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]

    raw_dir = Path(dataset_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {dataset_cfg['hf_repo_id']} to {raw_dir} ...")
    snapshot_download(
        repo_id=dataset_cfg["hf_repo_id"],
        repo_type="dataset",
        local_dir=str(raw_dir),
    )

    category_dir = Path(dataset_cfg["category_dir"])
    records = scan_mvtec_category(category_dir)
    n_train = sum(1 for r in records if r.official_split == "train")
    n_test = sum(1 for r in records if r.official_split == "test")
    n_anomalous = sum(1 for r in records if r.is_anomalous)
    logger.info(
        f"Verified {len(records)} images at {category_dir}: "
        f"{n_train} train (normal), {n_test} test ({n_test - n_anomalous} good, "
        f"{n_anomalous} anomalous)."
    )


if __name__ == "__main__":
    main()
