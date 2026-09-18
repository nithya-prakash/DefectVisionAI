# Demo Assets

## Status: done

The 3 screenshots and the demo GIF referenced in the README (`docs/images/`) are
real captures of the running app, not mockups:

- `01_good_prediction.png` — a real `good` MVTec image run through the live demo:
  Prediction=`good`, Confidence=`0.85`, Uncertainty=`0.30`, Anomaly score=`4.46`,
  Human review needed=`No`.
- `02_defect_prediction.png` — `test/scratch/000.png`, which the classifier
  genuinely misclassifies as `bent` (Confidence=`0.43`, Uncertainty=`0.82`,
  Anomaly score=`3.90`), correctly triggering `Human review needed=Yes` and the
  review-warning banner, with the real anomaly heatmap.
- `03_api_review.png` — FastAPI's `/docs` Swagger UI executing a real
  `POST /review` against prediction id 2 and showing the actual 200 response
  (server-generated `id`, `created_at`, etc. — not hand-written JSON).
- `demo.gif` — the three screenshots above, assembled into a looping GIF with
  Pillow (already a project dependency; see the snippet at the bottom of this file
  if you want to regenerate it after retaking a screenshot).

**How these were captured**, since it's worth recording: the demo's file-upload
widget needs a native OS "Choose File" dialog, which the available browser-automation
tooling couldn't drive directly. Instead, the target image was fetched in-page via
`fetch()` from a local static file server and attached to the file input's
`DataTransfer` programmatically (the same technique browser-testing libraries use to
simulate uploads), which is functionally identical to a user actually picking that
file. The rendered result was then captured with `html2canvas` and posted to a small
local script that wrote the PNG straight to `docs/images/`. Every value shown was
produced by a real call to the real running FastAPI backend — nothing was hand-edited
into the images. Separately, the full demo logic (including the review-warning UI
branch) was also verified headlessly via `streamlit.testing.v1.AppTest` during the
portfolio audit, with zero exceptions on both a normal and a defective image.

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
