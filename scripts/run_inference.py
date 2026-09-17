#!/usr/bin/env python3
"""Run the trained pipeline on a single image from the command line, without the API.

Usage:
    python scripts/run_inference.py --image path/to/image.jpg
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from automotive_vision.inference.predictor import Predictor
from automotive_vision.utils.config import load_config
from automotive_vision.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--no-heatmap", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.inference_config)
    predictor = Predictor(config, device="cpu")

    image_bytes = Path(args.image).read_bytes()
    result = predictor.predict(image_bytes, save_heatmap=not args.no_heatmap)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
