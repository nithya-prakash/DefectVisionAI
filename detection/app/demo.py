"""Streamlit demo: run the fine-tuned YOLOv8 on a test image or an uploaded photo.

    cd detection && .venv/bin/streamlit run app/demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "yolov8n_welding.pt"
TEST_DIR = ROOT / "data" / "welding" / "test" / "images"

st.set_page_config(page_title="Weld defect detection", layout="wide")
st.title("Weld defect detection with YOLOv8")
st.caption(
    "Fine-tuned YOLOv8n on a public CC0 welding dataset. See the README for the measured results, "
    "including how much of the test set overlaps with training."
)


@st.cache_resource
def load_model() -> YOLO:
    return YOLO(str(WEIGHTS))


if not WEIGHTS.exists():
    st.error(f"Weights not found at {WEIGHTS}. Run scripts/train.py first.")
    st.stop()

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
source = st.sidebar.radio("Image source", ["Test set", "Upload"])
img_path = None
if source == "Test set":
    names = sorted(p.name for p in TEST_DIR.glob("*.jpg"))
    if not names:
        st.warning("Dataset not downloaded; use Upload or see the README.")
    else:
        pick = st.sidebar.selectbox("Test image", names)
        img_path = TEST_DIR / pick
else:
    up = st.sidebar.file_uploader("Upload a weld photo", type=["jpg", "jpeg", "png"])
    if up:
        img_path = up

if img_path is not None:
    image = Image.open(img_path).convert("RGB")
    result = load_model().predict(image, conf=conf, imgsz=640, verbose=False)[0]
    left, right = st.columns(2)
    left.image(image, caption="Input", width="stretch")
    right.image(result.plot()[:, :, ::-1], caption="Detections", width="stretch")
    rows = [
        {"class": result.names[int(c)], "confidence": round(float(s), 3)}
        for c, s in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
    ]
    st.dataframe(rows, width="stretch")

metrics = ROOT / "results" / "test_metrics.json"
if metrics.exists():
    with st.expander("Measured test metrics"):
        st.json(json.loads(metrics.read_text()))
