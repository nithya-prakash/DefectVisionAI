"""Count how many valid and test images have a near-copy in the training split.

Distance is the Hamming distance between 16x16 average hashes (see welddet/leakage.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from welddet.leakage import nearest_train_distance

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0, 2, 4, 6, 10)


def images(split: str) -> list[Path]:
    return sorted((ROOT / "data" / "welding" / split / "images").glob("*.jpg"))


def main() -> None:
    train = images("train")
    out = {}
    for split in ("valid", "test"):
        imgs = images(split)
        d = nearest_train_distance(imgs, train)
        out[split] = {
            "images": len(imgs),
            "within_bits": {f"<={t}": int((d <= t).sum()) for t in THRESHOLDS},
        }
    path = ROOT / "results" / "leakage_check.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
