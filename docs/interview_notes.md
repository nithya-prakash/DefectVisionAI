# Interview Notes

Concise, technically accurate answers grounded in what's actually implemented in this
repository — no invented features. Cross-references point to the real files/numbers.

## Problem & Dataset

**1. What problem does this project solve?**
Industrial visual inspection oversimplified as "one classifier" fails in practice:
labeled defect examples are scarce, a classifier is silently wrong on
out-of-distribution parts, and nothing tells you when to trust it. This project
builds inspection as a system — comparing detection methods, quantifying
uncertainty, and testing (not assuming) whether active learning reduces the labeling
burden.

**2. Why did you choose MVTec AD?**
It's the standard, legitimate benchmark for industrial anomaly detection, with real
pixel-level ground-truth masks (enabling both image- and pixel-level evaluation) and
a well-documented, citable protocol (Bergmann et al., CVPR 2019) — as opposed to
scraping unlicensed images or fabricating synthetic defects.

**3. Why `metal_nut`?**
Of MVTec's 15 categories, it's the closest domain analog to automotive hardware
(a metal fastener), has multiple realistic defect subtypes (`bent`, `color`, `flip`,
`scratch`), and has a small enough image count (335) to iterate quickly on a laptop.

**4. Why is this not actually an automotive dataset?**
Because no MVTec AD category is literally automotive — `metal_nut` is a generic
industrial fastener, not a car part photographed on a production line. This project
states that explicitly in `README.md` and `data/README.md` rather than implying
otherwise; automotive manufacturing is the target *application domain*, MVTec is the
*benchmark*.

## Modeling Choices

**5. Why anomaly detection (in addition to classification)?**
Classification needs labeled defects for every defect type you care about, which is
expensive and never exhaustive. Anomaly detection only needs "normal" examples (which
are cheap to collect) and flags anything that deviates — catching defect types never
seen during training, at the cost of not naming *which* defect it is.

**6. Why PatchCore-style detection?**
It's a strong, simple, well-understood approach: no training beyond a frozen
pretrained backbone, patch-level (not whole-image) comparison for localization, and
a memory bank of normal-image features scored by nearest-neighbor distance. It's
also fast enough to fit and run repeatedly on a laptop.

**7. How does PatchCore-lite work?**
(`src/automotive_vision/models/anomaly_detector.py`) Extract mid-level (layer2 +
layer3) patch embeddings from a frozen ResNet18 for every normal training image,
store them in a memory bank (randomly subsampled to 10,000 patches for speed), then
score a new image by the maximum, over its own patches, of each patch's nearest-
neighbor distance to that bank. It's explicitly a **simplified** variant: random
subsampling instead of the original paper's greedy coreset selection, and k=1 nearest
neighbor instead of k-NN re-weighting — not a reproduction of the paper's benchmark
numbers.

**8. Why ResNet18?**
The labeled pool for the supervised/active-learning track is tiny (~80 images across
5 classes) — a larger backbone (EfficientNet, ConvNeXt) would only add overfitting
risk without more data to justify it. ResNet18 is also fast enough to fine-tune
repeatedly during the active-learning simulation's many retrain rounds.

**9. Why transfer learning (not training from scratch)?**
335 total images (and far fewer per class) is nowhere near enough to train a CNN
from random initialization. ImageNet-pretrained features already encode general
edge/texture/shape structure that's directly useful for surface-defect appearance.

**10. Why frozen embeddings for active learning specifically?**
Comparing 4 acquisition strategies × 5 seeds × 8 rounds means ~180 model fits. Fine-
tuning the full CNN that many times is impractical on a laptop. Frozen ResNet18
embeddings + a lightweight logistic-regression head make each fit near-instant,
trading fidelity (it's not the same fully fine-tuned model reported as the headline
classifier) for the ability to run the experiment at all.

## Uncertainty & Calibration

**11. What is uncertainty estimation, in this project specifically?**
Three signals derived from the classifier's softmax output
(`models/uncertainty.py`): **confidence** (top-1 probability), **normalized entropy**
(0-1, comparable across class counts), and **margin** (top-1 minus top-2
probability). A prediction is routed to human review when confidence falls below a
configurable threshold.

**12. Why entropy and margin, not just confidence?**
Confidence alone can be high even when two classes are both plausible if a third,
irrelevant class absorbs the remaining probability mass. Margin directly measures
how close the top two competitors are; entropy captures how spread out the *whole*
distribution is, which confidence and margin alone can miss.

**13. Why did temperature scaling make ECE worse (0.115 → 0.179)?**
Temperature scaling fits one scalar on the validation set — here, only 16 images.
That's too few examples to estimate a stable temperature; the fitted value (T=0.14)
over-sharpened confidence in a way that fit noise in those 16 examples rather than a
real miscalibration pattern, and that overfit didn't generalize to the 35-image test
set. This is a known risk of temperature scaling under small validation sets in
general — the specific causal story here is an interpretation of this one run, not a
universal claim, and is reported as a negative result rather than hidden.

Follow-up: I then fit the temperature leave-one-fold-out on 80 out-of-fold predictions
(5-fold cross-validation over the labeled pool). That improved out-of-fold ECE
(0.121 → 0.080, T=1.457, softening rather than sharpening), which supports the
small-validation-set explanation, and that temperature and the 0.74 review threshold
are what the API now uses. I did not measure test-set ECE with the new temperature,
and the cross-validation models are not the deployed checkpoint.

## Active Learning

**14. What is active learning, in this project?**
A simulation (`active_learning/simulation.py`) comparing 4 strategies for picking
which unlabeled image to label next, given a fixed labeling budget: random,
uncertainty (highest predictive entropy), diversity (farthest point from the current
labeled set in embedding space), and hybrid (weighted combination of both).

**15. Why did active learning not beat random sampling here?**
The labeled pool is only 80 images across 5 classes. With that little data, per-seed
noise in which 8 images get selected each round dominates any systematic advantage a
smarter strategy might have — the confidence bands (±1 std over 5 seeds) overlap for
most of the labeling-budget curve. Uncertainty sampling also shows the textbook
"cold-start" problem (worse than random early on, because a poorly-trained model's
uncertainty estimates aren't yet informative). This is reported as an inconclusive
result, not spun into a claimed win.

**16. How does simulated human review work?**
MVTec's real ground-truth labels stand in for a human annotator. The acquisition
loop only sees an image's label after that image is selected — before selection, the
loop can use the image's pixels/embedding but not its class. This is explicitly
documented as a simulation, not live annotation, everywhere it's discussed.

## Engineering

**17. Why FastAPI?**
Async-native, automatic OpenAPI/Swagger docs, first-class Pydantic validation for
request/response schemas, and it's the standard modern choice for a Python ML
serving layer.

**18. Why PostgreSQL?**
A relational schema (images → predictions → reviews, with model_versions) fits this
domain well, and Postgres is a realistic, production-representative choice.
SQLite is used automatically for local dev/tests (via the `DATABASE_URL` env var) so
the test suite doesn't require a running database server.

**19. Why Docker?**
Reproducible environment (exact Python version, exact dependency versions) and a
realistic way to run the API next to a real Postgres instance — verified end-to-end
multiple times in this project (see `README.md` → Running with Docker).

**20. What does the API architecture look like?**
`POST /predict` and `/predict/batch` run the model and persist an `Image` +
`Prediction` row; `POST /review` attaches a `Review` row to an existing prediction;
`GET /predictions/{id}` returns a prediction with its review if any; `GET /health`
reports whether the classifier/anomaly-detector are loaded; `GET /metrics` returns
simple prediction/review counts (application-level, not Prometheus format). See
`src/automotive_vision/api/routes.py`.

## Failure Modes & Deployment

**21. What are the major failure modes?**
(a) A defect type not represented in the tiny training data — the classifier can
only predict classes it's seen; (b) domain shift in the "normal" distribution for
anomaly detection (new part variant, different lighting) degrading PatchCore-lite
silently; (c) calibration being unreliable with this little validation data, as
already observed; (d) small-N evaluation noise — with 7 held-out examples per class,
a single misclassification swings F1 by ~0.14.

**22. What causes false positives (normal flagged as defective)?**
For the anomaly detector: any real but benign appearance variation not represented
in the 220 normal training images (dust, minor lighting change, a normal part
variant). For the classifier: `bent` and `color` are visibly confused in the
confusion matrix (see `reports/figures/classifier_confusion_matrix.png`).

**23. What causes false negatives (defect missed)?**
A defect subtle enough that its patch embeddings still fall close to normal patches
in feature space (PatchCore-lite), or a defect type visually similar to a class the
classifier already predicts confidently.

**24. What happens under camera/lighting/domain shift?**
Both the classifier and the anomaly detector were trained on MVTec's fixed camera
setup and lighting. Neither has been tested against a shifted domain in this project
— this is an explicit, stated limitation, not something claimed to be handled.

**25. How would you deploy this in a real automotive factory?**
Retrain/fine-tune on real factory images under the actual camera and lighting setup;
re-run the calibration and threshold-selection process on a properly-sized held-out
set (not 16-35 images); add monitoring for anomaly-score drift over time as a proxy
for "normal" distribution shift; keep the human-review queue as a safety net,
especially early in deployment.

**26. How would you scale it to many inspection cameras?**
The current API is a single FastAPI process; scaling to many camera feeds would mean
running multiple stateless API replicas behind a load balancer (the Predictor is
already a lazy-loaded singleton per process, `api/dependencies.py`), with Postgres as
the shared state store already in place. This is a straightforward extension of the
existing design, not something implemented here.

## Limitations & Future Work

**27. What would you change with a much larger dataset?**
Fine-tune the full CNN in the active-learning loop instead of using frozen
embeddings, use a proper validation set for temperature scaling (cross-validation
on the pool already replaced the 16-image fit), and get a more statistically meaningful read on whether any acquisition
strategy actually beats random.

**28. What are the biggest limitations of the current experiment?**
Tiny labeled pool (80 training / 35 held-out images across 5 classes) driving wide
confidence intervals everywhere; simulated (not live) human labeling; a simplified,
not full, PatchCore implementation; and calibration/active-learning results that are
negative or inconclusive rather than clean wins — all stated directly in
`README.md` → Limitations rather than smoothed over.

**29. What would you do next scientifically?**
Extend to a second MVTec category (`screw`) to see whether the active-learning and
calibration findings generalize or are artifacts of this one category's small
sample; compare MC-dropout (already
implemented in `models/uncertainty.py`, not yet wired into the live inference path)
against softmax entropy as the review-routing signal.

**30. What would you do next from an engineering perspective?**
Push to GitHub with CI actually running (currently written and locally verified but
never executed by GitHub Actions); nothing else is currently blocking — the test
suite, Docker deployment, and API are already verified working end-to-end.
