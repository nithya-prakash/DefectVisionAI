# Dataset

## What this project uses

**MVTec AD — `metal_nut` category.**

- **Original source:** MVTec Software GmbH, *"MVTec AD — A Comprehensive Real-World Dataset for
  Unsupervised Anomaly Detection"* (Bergmann et al., CVPR 2019).
  https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Original license:** CC BY-NC-SA 4.0 (non-commercial, share-alike). This project uses the
  dataset strictly for non-commercial, educational/portfolio purposes, consistent with that
  license.
- **How it was obtained here:** via a third-party Hugging Face Hub mirror,
  [`MSherbinii/mvtec-ad-metal-nut`](https://huggingface.co/datasets/MSherbinii/mvtec-ad-metal-nut),
  which preserves MVTec's original folder layout unmodified. The mirror's own Hub metadata tags
  it "MIT," but the underlying image data is MVTec's; this project treats it under MVTec's
  original CC BY-NC-SA 4.0 terms, not the mirror's tag.
- Run `python scripts/download_data.py` to fetch it. Nothing is committed to this repository —
  see `.gitignore`.

## Why `metal_nut`, and why not an automotive-specific dataset

No MVTec AD category is literally automotive. `metal_nut` (a metal fastener/hardware component)
was chosen as the closest available analog to an automotive-manufacturing inspection target —
metal fasteners are ubiquitous in vehicle assembly — while `screw` is a plausible secondary
extension for future work. **This is a deliberate limitation, not automotive data.** The
project treats automotive manufacturing as the target *application/domain* the system is
designed for, while the *benchmark* is MVTec AD's `metal_nut`. This distinction is repeated in
the top-level README.

## Verified counts (from the downloaded files, not from memory)

```
train/good:      220 images (normal only)
test/good:        22 images
test/bent:        25 images
test/color:       22 images
test/flip:        23 images
test/scratch:     23 images
test total:      115 images (22 good + 93 anomalous)
overall total:   335 images
```

Each anomalous test image has a corresponding pixel-level ground-truth mask in
`ground_truth/<defect_type>/`, enabling both image-level and pixel-level evaluation.

## Two data tracks used in this project (read this before touching the code)

1. **Anomaly detection** follows MVTec's official protocol exactly: train on the 220 normal-only
   images, evaluate on the full 115-image test set.
2. **Supervised classification + active learning** need labeled *defective* examples for
   training, and those only exist in the 115-image test folder (the 220 training images carry no
   defect-type label). So this project carves its own stratified train/test split out of that
   115-image labeled pool (see `src/automotive_vision/data/splitting.py`). This means the
   supervised/AL track trains and evaluates on a much smaller pool than the anomaly-detection
   track — a real constraint of repurposing this benchmark, stated here rather than hidden.

## Directory layout after download

```
data/
  raw/mvtec_ad_metal_nut/metal_nut/
    train/good/*.png
    test/{good,bent,color,flip,scratch}/*.png
    ground_truth/{bent,color,flip,scratch}/*_mask.png
  processed/   # cached embeddings / resized tensors, gitignored
```
