"""Fine-tune YOLOv8 on the welding defect dataset (Apple silicon or CPU)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--name", default="yolov8n_welding")
    args = ap.parse_args()

    # data.yaml shipped with the dataset uses ../train/images; write an absolute-path copy.
    data = ROOT / "data" / "welding"
    cfg = ROOT / "data" / "welding.yaml"
    cfg.write_text(
        f"path: {data}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 3\nnames: ['Bad Weld', 'Good Weld', 'Defect']\n"
    )
    model = YOLO(args.weights)
    model.train(
        data=str(cfg),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=0,
        deterministic=False,  # MPS has no fully deterministic kernels; runs are not bit-identical
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
