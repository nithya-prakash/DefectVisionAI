"""Validate the welding dataset and measure train/valid/test overlap.

Roboflow-style exports name augmented copies `<source>.rf.<hash>.jpg`, so images sharing the part
before `.rf.` come from the same source picture. If such a source appears in more than one split,
test scores are optimistic. This prints the counts instead of assuming there is no overlap.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

CLASSES = ["Bad Weld", "Good Weld", "Defect"]


def source_id(name: str) -> str:
    return name.split(".rf.")[0]


def scan(root: Path) -> dict:
    out: dict = {"splits": {}}
    sources: dict[str, set[str]] = collections.defaultdict(set)
    for split in ("train", "valid", "test"):
        imgs = sorted((root / split / "images").glob("*.jpg"))
        class_counts = collections.Counter()
        problems = []
        n_boxes = 0
        for img in imgs:
            sources[source_id(img.name)].add(split)
            label = root / split / "labels" / (img.stem + ".txt")
            if not label.exists():
                problems.append(f"missing label: {img.name}")
                continue
            for line in label.read_text().splitlines():
                p = line.split()
                if len(p) != 5:
                    problems.append(f"bad row in {label.name}: {line!r}")
                    continue
                c, *xywh = int(p[0]), *map(float, p[1:])
                if c not in (0, 1, 2) or not all(0 <= v <= 1 for v in xywh):
                    problems.append(f"out of range in {label.name}: {line!r}")
                class_counts[CLASSES[c] if c in (0, 1, 2) else str(c)] += 1
                n_boxes += 1
        out["splits"][split] = {
            "images": len(imgs),
            "boxes": n_boxes,
            "classes": dict(class_counts),
            "problems": problems[:10],
            "n_problems": len(problems),
        }
    multi = {k: sorted(v) for k, v in sources.items() if len(v) > 1}
    out["sources_total"] = len(sources)
    out["sources_in_multiple_splits"] = len(multi)
    out["test_images_with_source_in_train"] = sum(
        1
        for img in (root / "test" / "images").glob("*.jpg")
        if "train" in sources[source_id(img.name)]
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/welding")
    ap.add_argument("--out", default="results/data_check.json")
    args = ap.parse_args()
    res = scan(Path(args.root))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
