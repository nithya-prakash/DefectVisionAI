"""Box geometry and an independent average-precision implementation.

Ultralytics reports mAP itself. This module recomputes AP@0.5 from raw predictions so the
reported number can be cross-checked instead of taken on trust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    score: float = 1.0


def yolo_to_box(cls: int, cx: float, cy: float, w: float, h: float, width: int, height: int) -> Box:
    """Convert a normalised YOLO label row to a pixel-space corner box."""
    return Box(
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
        cls,
    )


def read_labels(path: str, width: int, height: int) -> list[Box]:
    boxes = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 5:
                continue
            boxes.append(yolo_to_box(int(parts[0]), *map(float, parts[1:]), width, height))
    return boxes


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def average_precision(
    preds: dict[str, list[Box]], gts: dict[str, list[Box]], cls: int, iou_thr: float = 0.5
) -> float:
    """All-point interpolated AP for one class at one IoU threshold.

    Detections are ranked by score across all images; each ground-truth box can be matched once.
    Returns nan when the class has no ground-truth boxes.
    """
    gt_by_img = {k: [b for b in v if b.cls == cls] for k, v in gts.items()}
    n_gt = sum(len(v) for v in gt_by_img.values())
    if n_gt == 0:
        return float("nan")
    dets = [(k, b) for k, v in preds.items() for b in v if b.cls == cls]
    dets.sort(key=lambda kb: -kb[1].score)
    used = {k: [False] * len(v) for k, v in gt_by_img.items()}
    tp = np.zeros(len(dets))
    for i, (img, det) in enumerate(dets):
        best, best_j = 0.0, -1
        for j, g in enumerate(gt_by_img.get(img, [])):
            o = iou(det, g)
            if o > best:
                best, best_j = o, j
        if best >= iou_thr and not used[img][best_j]:
            used[img][best_j] = True
            tp[i] = 1
    fp = 1 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
