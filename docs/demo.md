# Demo Assets

## Status: re-recorded 2026-10-14 under the current policy

The GIF and screenshots in `docs/images/` are real captures of the running demo, re-recorded after the review policy
changed (temperature 1.4568, confidence threshold 0.74, anomaly gate off; see `configs/inference.yaml`). The earlier
set (2026-09-18) was recorded with temperature 1.0 and threshold 0.7, so its confidences no longer matched the code.

- `01_good_prediction.png`: `test/good/002.png` through the live demo: Prediction `good`, Confidence `0.87`,
  Uncertainty `0.32`, Anomaly score `4.27`, Human review needed `No`.
- `02_defect_prediction.png`: `test/scratch/000.png`, which the classifier misclassifies as `bent`
  (Confidence `0.36`, Uncertainty `0.88`, Anomaly score `3.90`): `Human review needed = Yes`, the review-warning
  banner and the anomaly heatmap.
- `03_api_review.png`: the API's `/docs` Swagger UI executing `POST /review` for prediction id 2 (the scratch image,
  label `scratch`, status `corrected`) with the real 200 response. Prediction id 2 was verified in the database.
- `demo.gif`: six frames of the same session (demo loaded, good image, its result, scratch image, its result, the review
  call), assembled with ffmpeg.

These are two illustrative test images, not a sample of the dataset. The same policy leaves some wrong predictions
unreviewed: in the probe run, `test/color/002.png` (defective) was predicted `good` at confidence 0.81 and not flagged;
the out-of-fold auto-decided accuracy is 0.907, so roughly 1 in 11 auto-decided parts is wrong.

**How they were captured:** the API (the real `docker/Dockerfile.api` image, real checkpoints, a fresh SQLite database)
and the Streamlit demo ran in containers; headless Chrome, driven over the DevTools protocol, attached the real image
files through the demo's file input, clicked "Run inspection", and then clicked through the Swagger UI. Every value shown
is a real response from that API. The recording script is not part of this repository.

## Regenerating the screenshots

```bash
# Terminal 1
uvicorn automotive_vision.api.main:app --port 8020 &
API_URL=http://localhost:8020 streamlit run demo/app.py --server.port 8511

# Then open http://localhost:8511, upload a real MVTec test image
# (data/raw/mvtec_ad_metal_nut/metal_nut/test/<type>/*.png), click "Run inspection",
# and screenshot the result manually (Cmd+Shift+5 on macOS).
```

## Regenerating demo.gif from updated screenshots

```python
from PIL import Image

CANVAS_SIZE = (900, 700)
BG = (14, 17, 23)

def fit(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail(CANVAS_SIZE, Image.LANCZOS)
    canvas = Image.new("RGB", CANVAS_SIZE, BG)
    canvas.paste(im, ((CANVAS_SIZE[0] - im.width) // 2, (CANVAS_SIZE[1] - im.height) // 2))
    return canvas

frames = [fit(f"docs/images/{n}") for n in
          ["01_good_prediction.png", "02_defect_prediction.png", "03_api_review.png"]]
frames[0].save("docs/images/demo.gif", save_all=True, append_images=frames[1:],
               duration=[2500, 2500, 2500], loop=0)
```

## Recording a real screen-capture video instead (optional)

If you'd rather have a real screen recording instead of/alongside the GIF:

1. (0-5s) Show the empty demo UI: title, API-reachable banner.
2. (5-12s) Upload `test/good/000.png`, click "Run inspection," show the result.
3. (12-20s) Upload `test/scratch/000.png` instead, re-run, and show the "flagged for
   human review" warning appearing (the misclassification case) plus its heatmap.
4. (20-30s) Switch to `/docs`, execute `POST /review` against that prediction, show
   the real response.
5. Cut.

- **macOS**: `Cmd+Shift+5` opens the built-in screen recorder.
- **Convert to GIF**: `ffmpeg -i recording.mov -vf "fps=10,scale=900:-1" demo.gif`
  (only if `ffmpeg` is already installed — not worth adding as a new dependency
  just for this).
