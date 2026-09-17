"""Lightweight Streamlit demo. Calls the FastAPI backend over HTTP -- the API remains
the primary interface; this is a thin UI on top of it, not a parallel prediction path.

Run with the API already up (uvicorn or docker compose), then:
    streamlit run demo/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Automotive Vision Defect Detection", layout="centered")
st.title("Automotive Vision Defect Detection")
st.caption(
    "Portfolio demo on MVTec AD `metal_nut` (not automotive-specific data -- see README). "
    "Calls the FastAPI backend for every prediction."
)

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.success(f"API reachable at {API_URL} — classifier_loaded={health['classifier_loaded']}")
except requests.RequestException:
    st.error(f"Could not reach the API at {API_URL}. Start it first (see README).")
    st.stop()

uploaded_file = st.file_uploader("Upload a part image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Input image", width=300)

    if st.button("Run inspection"):
        with st.spinner("Running inference..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=30)

        if response.status_code != 200:
            st.error(f"API error {response.status_code}: {response.text}")
        else:
            result = response.json()
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", result["prediction"])
            col2.metric("Confidence", f"{result['confidence']:.2f}")
            col3.metric("Uncertainty", f"{result['uncertainty']:.2f}")

            col4, col5 = st.columns(2)
            if result["anomaly_score"] is not None:
                col4.metric("Anomaly score", f"{result['anomaly_score']:.2f}")
            col5.metric("Human review needed", "Yes" if result["needs_human_review"] else "No")

            if result["needs_human_review"]:
                st.warning("This prediction is flagged for human review.")

            if result["heatmap_path"]:
                st.image(result["heatmap_path"], caption="Anomaly heatmap")
