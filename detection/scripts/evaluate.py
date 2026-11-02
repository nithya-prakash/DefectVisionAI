"""Evaluate on the held-out test split, on the full set and on a leakage-reduced subset.

Per set it reports Ultralytics mAP, an independent AP@0.5 cross-check, and a trivial prior-box
baseline. The baseline predicts, for every image, the mean box of each class (normalised) with the
class frequency as its score; any real detector must beat it.

The leakage-reduced subset keeps test images whose nearest training image (16x16 average hash) is
more than `--min-distance` bits away. See welddet/leakage.py for why this exists.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from welddet.boxes import Box, average_precision, read_labels
from welddet.leakage import clean_subset, nearest_train_distance

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLASSES = ["Bad Weld", "Good Weld", "Defect"]


def list_images(split: str) -> list[Path]:
    return sorted((DATA / "welding" / split / "images").glob("*.jpg"))


def load_gts(imgs: list[Path]) -> dict[str, list[Box]]:
    gts = {}
    for img in imgs:
        with Image.open(img) as im:
            w, h = im.size
        label = img.parent.parent / "labels" / (img.stem + ".txt")
        gts[img.name] = read_labels(str(label), w, h)
    return gts


def prior_baseline(train_gts: dict[str, list[Box]], sizes: dict[str, tuple[int, int]]):
    rows: dict[int, list] = {c: [] for c in range(3)}
    for name, boxes in train_gts.items():
        w, h = sizes[name]
        for b in boxes:
            rows[b.cls].append((b.x1 / w, b.y1 / h, b.x2 / w, b.y2 / h))
    total = sum(len(v) for v in rows.values())
    mean = {c: np.mean(v, axis=0) for c, v in rows.items() if v}
    freq = {c: len(v) / total for c, v in rows.items()}

    def predict(name: str) -> list[Box]:
        w, h = sizes[name]
        return [Box(m[0] * w, m[1] * h, m[2] * w, m[3] * h, c, freq[c]) for c, m in mean.items()]

    return predict


def evaluate_set(model, imgs: list[Path], list_name: str, device: str, base) -> dict:
    listing = DATA / list_name
    listing.write_text("\n".join(str(p) for p in imgs) + "\n")
    cfg = DATA / f"eval_{list_name}.yaml"
    cfg.write_text(
        f"path: {DATA}\ntrain: {listing}\nval: {listing}\nnc: 3\n"
        "names: ['Bad Weld', 'Good Weld', 'Defect']\n"
    )
    gts = load_gts(imgs)
    official = model.val(data=str(cfg), imgsz=640, device=device, conf=0.001, plots=False)
    idx = [int(c) for c in official.ap_class_index]
    per_ap50 = {CLASSES[c]: float(v) for c, v in zip(idx, official.box.ap50)}
    per_ap = {CLASSES[c]: float(v) for c, v in zip(idx, official.box.ap)}

    preds: dict[str, list[Box]] = {}
    times = []
    for img in imgs:
        t0 = time.perf_counter()
        r = model.predict(str(img), imgsz=640, conf=0.001, device=device, verbose=False)[0]
        times.append(time.perf_counter() - t0)
        preds[img.name] = [
            Box(*map(float, xy), int(c), float(s))
            for xy, c, s in zip(
                r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()
            )
        ]
    own = {CLASSES[c]: average_precision(preds, gts, c, 0.5) for c in range(3)}
    bpred = {i.name: base(i.name) for i in imgs}
    bap = {CLASSES[c]: average_precision(bpred, gts, c, 0.5) for c in range(3)}
    return {
        "images": len(imgs),
        "boxes": sum(len(v) for v in gts.values()),
        "ultralytics": {
            "mAP50": float(official.box.map50),
            "mAP50_95": float(official.box.map),
            "precision": float(official.box.mp),
            "recall": float(official.box.mr),
            "per_class_AP50": per_ap50,
            "per_class_AP50_95": per_ap,
        },
        "independent_AP50": {**own, "mean": float(np.nanmean(list(own.values())))},
        "prior_box_baseline_AP50": {**bap, "mean": float(np.nanmean(list(bap.values())))},
        "mean_latency_ms": 1000 * float(np.mean(times)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="results/test_metrics.json")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--min-distance", type=int, default=10)
    args = ap.parse_args()

    model = YOLO(args.weights)
    train_imgs, test_imgs = list_images("train"), list_images("test")
    sizes = {}
    for p in train_imgs + test_imgs:
        with Image.open(p) as im:
            sizes[p.name] = im.size
    base = prior_baseline(load_gts(train_imgs), sizes)

    dist = nearest_train_distance(test_imgs, train_imgs)
    clean = clean_subset(test_imgs, dist, args.min_distance)
    res = {
        "device": args.device,
        "leakage_min_distance_bits": args.min_distance,
        "full_test": evaluate_set(model, test_imgs, "test_full.txt", args.device, base),
        "leakage_reduced_test": evaluate_set(model, clean, "test_clean.txt", args.device, base),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
