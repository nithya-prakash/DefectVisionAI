# Demo Recording Guide

## Why this is a manual step

The demo's file upload widget requires a native OS "Choose File" dialog. The
browser-automation tools available in this environment cannot interact with native
file pickers (this is a real, documented limitation of remote browser automation, not
a bug in the demo). Instead, the complete app logic was verified headlessly with
Streamlit's own test harness (`streamlit.testing.v1.AppTest`), which simulates a real
file upload and a real call to the running FastAPI backend without needing a browser
at all -- confirming zero exceptions and correct rendering of every UI branch
(metrics, heatmap image, human-review warning) on both a normal and a defective real
MVTec image. What's below are the exact manual steps to capture the visual assets
that verification can't produce.

## Screenshots to capture manually

Save these to `docs/images/`.

**1. `docs/images/01_good_prediction.png` — a normal part**

1. Start the API: `uvicorn automotive_vision.api.main:app --reload` (or `docker compose up`)
2. Start the demo: `API_URL=http://localhost:8000 streamlit run demo/app.py` (use `:8020` if running via Docker)
3. Upload `data/raw/mvtec_ad_metal_nut/metal_nut/test/good/000.png`
4. Click "Run inspection"
5. Screenshot the full result. Expected values (verified during this audit, your run
   should match exactly since inference is deterministic): Prediction=`good`,
   Confidence=`0.85`, Uncertainty=`0.30`, Anomaly score=`4.46`, Human review
   needed=`No`.

**2. `docs/images/02_defect_prediction.png` — a defective part with heatmap**

1. Upload `data/raw/mvtec_ad_metal_nut/metal_nut/test/bent/000.png`
2. Click "Run inspection"
3. Screenshot the full result, including the anomaly heatmap image at the bottom.
   Expected: Prediction=`bent`, Confidence=`0.87`, Anomaly score=`5.75`, Human review
   needed=`No` (high-confidence correct classification -- for a case that instead
   triggers the review warning, use `test/scratch/000.png`, which this audit
   confirmed gets misclassified as `bent` at confidence `0.43` and correctly flags
   `Human review needed=Yes` with the warning banner shown).

**3. `docs/images/03_api_review.png` — the human-review API workflow**

1. With the API running, open `http://localhost:8000/docs` (FastAPI's auto-generated
   Swagger UI) -- or use `curl`/Postman.
2. Show a `POST /review` call against a prediction ID returned by step 2, and the
   resulting `GET /predictions/{id}` showing the review attached.
3. Screenshot the request/response.

## 20-40 second demo recording script

1. (0-5s) Show the empty demo UI: title, API-reachable banner.
2. (5-12s) Upload `test/bent/000.png`, click "Run inspection."
3. (12-20s) Show the prediction/confidence/uncertainty/anomaly-score metrics appearing.
4. (20-28s) Scroll to the anomaly heatmap image.
5. (28-35s) Upload `test/scratch/000.png` instead, re-run, and show the "flagged for
   human review" warning appearing (the misclassification case).
6. (35-40s) Cut.

### Recording locally

- **macOS**: `Cmd+Shift+5` opens the built-in screen recorder; select the browser
  window, click Record.
- **Convert to GIF** (optional, no new dependency needed if `ffmpeg` is already
  installed): `ffmpeg -i recording.mov -vf "fps=10,scale=800:-1" demo.gif`
