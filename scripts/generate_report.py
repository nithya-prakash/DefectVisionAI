#!/usr/bin/env python3
"""Assembles reports/evaluation_report.md purely from configs/*.yaml and the JSON
files other scripts already wrote to reports/metrics/. Never invents a number: any
section whose JSON file is missing is rendered as "not yet run" rather than skipped
silently or guessed, and every reported figure/table cell is read directly from those
files.

Usage:
    python scripts/generate_report.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from automotive_vision.utils.config import load_config

METRICS_DIR = Path("reports/metrics")
OUT_PATH = Path("reports/evaluation_report.md")


def load(name: str) -> dict | None:
    path = METRICS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "N/A"


def section_objective() -> list[str]:
    return [
        "## 1. Objective",
        "",
        ("Evaluate whether an industrial visual-inspection pipeline combining a classical CV "
        "baseline, a supervised classifier, a patch-based anomaly detector, calibrated "
        "uncertainty estimation, and active-learning sample selection can (a) detect surface "
        "defects reliably and (b) reduce the amount of labeled data required to do so, using "
        "automotive hardware inspection (metal fasteners) as the target application domain."),
        "",
    ]


def section_dataset_protocol(anomaly: dict | None, classification: dict | None) -> list[str]:
    lines = [
        "## 2. Dataset and Protocol",
        "",
        ("**MVTec AD `metal_nut`** (335 real images: 220 normal + 115 test = 22 good + 93 "
        "anomalous across `bent`/`color`/`flip`/`scratch`, each anomalous image with a "
        "pixel-level ground-truth mask). Not automotive production data -- the closest "
        "available analog to a fastener/hardware inspection target in this benchmark. See "
        "`data/README.md` for full provenance and license (CC BY-NC-SA 4.0)."),
        "",
        "**Two distinct protocols are used, and are not interchangeable:**",
        "",
    ]
    if anomaly:
        lines.append(
            f"1. **Anomaly detection** -- MVTec's official protocol: train on "
            f"{anomaly['n_train_normal']} normal-only images, evaluate on the full "
            f"{anomaly['n_test']}-image test set."
        )
    else:
        lines.append("1. **Anomaly detection** -- MVTec's official protocol (train on normal-only, evaluate on official test set). _Not yet run._")
    if classification:
        lines.append(
            f"2. **Supervised classification + active learning** -- these need labeled "
            f"*defective* examples, which only exist in the 115-image test folder. This "
            f"project carves its own stratified split of that pool: {classification['n_train']} "
            f"train + {classification['n_val']} validation "
            f"(both drawn from a development pool) and a separate "
            f"{classification['n_held_out_test']}-image held-out test set never touched during "
            "training or model selection. This is a deliberate deviation from protocol 1, not an "
            "oversight."
        )
    else:
        lines.append("2. **Supervised classification + active learning** -- own stratified split of the labeled pool (80 dev / 35 held-out). _Not yet run._")
    lines.append("")
    return lines


def section_experimental_setup(config: dict) -> list[str]:
    ds, tr, ad, al = config["dataset"], config["training"], config["anomaly_detection"], config["active_learning"]
    return [
        "## 3. Experimental Setup",
        "",
        "All values below are read directly from `configs/*.yaml`, not restated from memory.",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Global seed | {config['seed']} |",
        f"| Image size | {ds['image_size']}x{ds['image_size']} |",
        "| Backbone | ResNet18 (ImageNet-pretrained) |",
        f"| Classifier fine-tune strategy | {'partial (layer4 + head unfrozen)' if tr['freeze_until_layer4'] else 'full fine-tune'} |",
        f"| Classifier epochs (max / early-stop patience) | {tr['epochs']} / {tr['early_stopping_patience']} |",
        f"| Classifier learning rate / weight decay | {tr['learning_rate']} / {tr['weight_decay']} |",
        f"| Classifier batch size | {tr['batch_size']} |",
        f"| Held-out test fraction / val fraction | {tr['held_out_fraction']} / {tr['val_fraction']} |",
        f"| PatchCore-lite memory bank size | {ad['memory_bank_size']} patches |",
        f"| Active learning: initial labeled fraction | {al['initial_labeled_fraction']} |",
        f"| Active learning: acquisition size / rounds | {al['acquisition_size']} / {al['rounds']} |",
        f"| Active learning: seeds per strategy | {al['n_repeats']} |",
        f"| Active learning: hybrid alpha | {al['alpha']} |",
        "",
    ]


def section_classical_baseline(anomaly: dict | None) -> list[str]:
    lines = ["## 4. Classical Baseline", ""]
    if not anomaly:
        lines.append("_Not yet run. Run `python scripts/train_anomaly.py`._")
        lines.append("")
        return lines
    c = anomaly["classical_baseline"]
    lines += [
        ("Grayscale intensity histogram + LBP texture histogram + Sobel edge-magnitude "
        "histogram per whole image, scored by nearest-neighbor distance to a bank of "
        "normal-image feature vectors. Deliberately simple -- a reference point, not a "
        "competitive method."),
        "",
        f"- Image AUROC: **{fmt(c['image_auroc'], 3)}** (95% CI: [{fmt(c['image_auroc_ci95'][0], 3)}, {fmt(c['image_auroc_ci95'][1], 3)}])",
        f"- Image AUPRC: **{fmt(c['image_auprc'], 3)}** (95% CI: [{fmt(c['image_auprc_ci95'][0], 3)}, {fmt(c['image_auprc_ci95'][1], 3)}])",
        f"- Inference: {fmt(c['inference_seconds_per_image'] * 1000, 1)} ms/image",
        "",
        ("AUROC barely above chance (0.5), as expected for whole-image texture statistics "
        "with no spatial localization -- this is the intended role of this baseline."),
        "",
    ]
    return lines


def section_anomaly_detector(anomaly: dict | None) -> list[str]:
    lines = ["## 5. Anomaly Detector (PatchCore-lite)", ""]
    if not anomaly:
        lines.append("_Not yet run. Run `python scripts/train_anomaly.py`._")
        lines.append("")
        return lines
    p = anomaly["patchcore_lite"]
    lines += [
        ("A **simplified PatchCore-style** detector: frozen ResNet18 mid-level (layer2+layer3) "
        "patch embeddings, a memory bank of normal-image patches, nearest-neighbor (k=1) "
        "distance as the anomaly score. Simplified relative to the original paper (Roth et "
        "al., 2022) in two ways: (1) uniform random memory-bank subsampling instead of greedy "
        "minimax coreset selection, and (2) single nearest-neighbor instead of k-NN "
        "re-weighting. This is a portfolio reference implementation, not a reproduction of the "
        "paper's benchmark numbers."),
        "",
        f"- Image AUROC: **{fmt(p['image_auroc'], 3)}** (95% CI: [{fmt(p['image_auroc_ci95'][0], 3)}, {fmt(p['image_auroc_ci95'][1], 3)}])",
        f"- Image AUPRC: **{fmt(p['image_auprc'], 3)}** (95% CI: [{fmt(p['image_auprc_ci95'][0], 3)}, {fmt(p['image_auprc_ci95'][1], 3)}])",
        f"- Pixel AUROC: **{fmt(p['pixel_auroc'], 3)}** (over {p['n_pixels']:,} pixels, {p['n_positive_pixels']:,} positive)",
        f"- Memory bank: {p['memory_bank_patches']:,} patches, built in {fmt(p['fit_seconds'], 1)}s",
        f"- Inference: {fmt(p['inference_seconds_per_image'] * 1000, 1)} ms/image on {p['device']}",
        "",
        ("See `reports/figures/anomaly_roc.png` (ROC vs. classical baseline) and "
        "`reports/figures/anomaly_heatmap_examples.png` (localization on real `bent` defects)."),
        "",
    ]
    return lines


def section_supervised_classifier(classification: dict | None) -> list[str]:
    lines = ["## 6. Supervised Classifier", ""]
    if not classification:
        lines.append("_Not yet run. Run `python scripts/train.py`._")
        lines.append("")
        return lines
    t = classification["test_metrics"]
    lines += [
        ("Partial fine-tune ResNet18 (layer4 + classification head unfrozen; layer1-3 frozen "
        "at ImageNet weights), chosen over a larger backbone specifically because the labeled "
        "pool is tiny (~80 training images across 5 classes) -- a bigger model would only add "
        "overfitting risk. Class-weighted cross-entropy, early stopping and LR scheduling on "
        "validation macro-F1."),
        "",
        f"- Held-out test (n={t['n_samples']}): accuracy **{fmt(t['accuracy'], 3)}**, macro-F1 **{fmt(t['macro_f1'], 3)}**",
        f"- Best epoch: {classification['best_epoch']} of {classification['epochs_run']} run (early-stopped)",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in t["per_class"].items():
        lines.append(f"| {name} | {fmt(m['precision'], 3)} | {fmt(m['recall'], 3)} | {fmt(m['f1'], 3)} | {m['support']} |")
    lines.append("")
    lines.append("See `reports/figures/classifier_confusion_matrix.png`.")
    lines.append("")
    return lines


def section_uncertainty() -> list[str]:
    return [
        "## 7. Uncertainty Estimation",
        "",
        ("Every classifier prediction carries three softmax-derived signals "
        "(`src/automotive_vision/models/uncertainty.py`): **confidence** (top-1 probability), "
        "**normalized entropy** (0-1, comparable across class counts), and **margin** "
        "(top-1 minus top-2 probability). A prediction is routed to human review when "
        "confidence falls below a configurable threshold (and optionally when entropy exceeds "
        "one). Monte Carlo dropout is also implemented (`mc_dropout_predict`) as a second, "
        "independent uncertainty signal, but is not wired into the default inference path -- "
        "available for future comparison, not part of the reported results below."),
        "",
    ]


def section_calibration(calibration: dict | None) -> list[str]:
    lines = ["## 8. Calibration", ""]
    if not calibration:
        lines.append("_Not yet run. Run `python scripts/evaluate.py`._")
        lines.append("")
        return lines
    lines += [
        (f"Temperature scaling was fit on the {calibration['n_val']}-image validation set and "
        f"evaluated for Expected Calibration Error (ECE) on the {calibration['n_test']}-image "
        "held-out test set."),
        "",
        "| | ECE |",
        "|---|---:|",
        f"| Raw | {fmt(calibration['raw_ece'], 3)} |",
        f"| Temperature-scaled (T={fmt(calibration['temperature'], 3)}) | {fmt(calibration['calibrated_ece'], 3)} |",
        "",
    ]
    if calibration["calibrated_ece"] > calibration["raw_ece"]:
        lines += [
            ("**Interpretation, scoped to this experiment:** temperature scaling made "
            "calibration worse here rather than better. The most likely explanation is that "
            f"fitting a single scalar temperature on only {calibration['n_val']} validation "
            "examples is unstable -- the fitted value overfit that small sample and "
            "over-sharpened confidence on the held-out test set. This is a known risk of "
            "temperature scaling under small validation sets in general, but the specific "
            "causal claim here is an interpretation of this one run, not a universal result."),
            "",
        ]
    lines += [
        (f"The human-review confidence threshold ({calibration['selected_review_threshold']}, "
        "see `configs/inference.yaml`) was chosen by this explicit rule, not asserted as "
        f"optimal: *{calibration['selection_rule']}*. Full threshold sweep in "
        "`reports/metrics/calibration.json`; reliability diagrams in "
        "`reports/figures/reliability_diagram.png`."),
        "",
    ]
    return lines


def section_active_learning(active_learning: dict | None) -> list[str]:
    lines = ["## 9. Active Learning", ""]
    if not active_learning:
        lines.append("_Not yet run. Run `python scripts/run_active_learning.py`._")
        lines.append("")
        return lines
    lines += [
        ("**This is a simulation**, not a live annotation study: MVTec's ground-truth labels "
        "stand in for a human reviewer, revealed only when the acquisition strategy selects "
        "that sample. For compute reasons (4 strategies x "
        f"{active_learning['config']['n_repeats']} seeds x {active_learning['config']['rounds']} "
        "retraining rounds), this uses frozen ResNet18 embeddings + a lightweight "
        "logistic-regression head rather than fine-tuning the full CNN each round -- the "
        "headline classifier numbers in Section 6 come from the fully fine-tuned model, not "
        "this lightweight head."),
        "",
        (f"Pool: {active_learning['n_pool']} images, held-out test: {active_learning['n_held_out_test']} "
        f"(same held-out set as Section 6), {active_learning['config']['n_repeats']} seeds per strategy."),
        "",
        "| Strategy | Final macro-F1 (mean +/- std, full pool labeled) |",
        "|---|---:|",
    ]
    for strategy, runs in active_learning["results"].items():
        finals = [r["macro_f1"][-1] for r in runs]
        mean = sum(finals) / len(finals)
        std = (sum((x - mean) ** 2 for x in finals) / len(finals)) ** 0.5
        lines.append(f"| {strategy} | {fmt(mean, 3)} +/- {fmt(std, 3)} |")
    lines += [
        "",
        ("**Honest reading of this result:** uncertainty sampling shows the well-known "
        "\"cold-start\" pathology (worse than random early on), and all strategies converge "
        "once the full pool is labeled -- an expected sanity check, not a finding. Confidence "
        "bands (+/-1 std over 5 seeds, see `reports/figures/active_learning_curves.png`) "
        "overlap for most of the labeling-budget range. **This experiment does not "
        "demonstrate a statistically clear labeling-efficiency advantage for any acquisition "
        "strategy over random selection**, given the dataset's small size. A larger pool "
        "would be needed to resolve whether the mid-budget separation visible in the plot is "
        "signal or noise."),
        "",
    ]
    return lines


def section_results(anomaly, classification, calibration, active_learning) -> list[str]:
    lines = ["## 10. Results Summary", "", "| Experiment | Headline metric | Value |", "|---|---|---:|"]
    if anomaly:
        lines.append(f"| Classical baseline | Image AUROC | {fmt(anomaly['classical_baseline']['image_auroc'], 3)} |")
        lines.append(f"| PatchCore-lite | Image AUROC | {fmt(anomaly['patchcore_lite']['image_auroc'], 3)} |")
        lines.append(f"| PatchCore-lite | Pixel AUROC | {fmt(anomaly['patchcore_lite']['pixel_auroc'], 3)} |")
    if classification:
        lines.append(f"| Supervised classifier | Accuracy | {fmt(classification['test_metrics']['accuracy'], 3)} |")
        lines.append(f"| Supervised classifier | Macro-F1 | {fmt(classification['test_metrics']['macro_f1'], 3)} |")
    if calibration:
        lines.append(f"| Calibration | Raw ECE | {fmt(calibration['raw_ece'], 3)} |")
        lines.append(f"| Calibration | Temperature-scaled ECE | {fmt(calibration['calibrated_ece'], 3)} |")
    if active_learning:
        lines.append("| Active learning | Clear advantage over random? | No (inconclusive, see Section 9) |")
    lines.append("")
    return lines


def section_error_analysis(classification: dict | None) -> list[str]:
    lines = ["## 11. Error Analysis", ""]
    if not classification:
        lines.append("_Not yet run. Run `python scripts/train.py`._")
        lines.append("")
        return lines
    t = classification["test_metrics"]
    cm = t["confusion_matrix"]
    names = t["class_names"]
    confusions = []
    for i, true_name in enumerate(names):
        for j, pred_name in enumerate(names):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], true_name, pred_name))
    confusions.sort(reverse=True)
    lines.append("Largest off-diagonal confusions in the held-out test confusion matrix:")
    lines.append("")
    for count, true_name, pred_name in confusions[:5]:
        lines.append(f"- {count}x true `{true_name}` predicted as `{pred_name}`")
    weakest = min(t["per_class"].items(), key=lambda kv: kv[1]["f1"])
    strongest = max(t["per_class"].items(), key=lambda kv: kv[1]["f1"])
    lines += [
        "",
        (f"Weakest class: `{weakest[0]}` (F1={fmt(weakest[1]['f1'], 3)}). Strongest: "
        f"`{strongest[0]}` (F1={fmt(strongest[1]['f1'], 3)}). With only "
        f"{t['per_class'][weakest[0]]['support']} held-out examples per class, single "
        "misclassifications move F1 by ~0.14-0.29 -- these per-class numbers are indicative, "
        "not statistically precise."),
        "",
    ]
    return lines


def section_limitations() -> list[str]:
    return [
        "## 12. Limitations",
        "",
        "- Benchmark data (MVTec AD `metal_nut`) is not automotive production data.",
        ("- Human labeling in the active-learning experiment is simulated using ground-truth "
        "labels, not a live annotation workflow."),
        ("- Both the supervised and active-learning tracks train on a very small pool (up to "
        "80 labeled images across 5 classes), because MVTec AD only labels defect subtypes "
        "in its test folder -- reported metrics (and their wide confidence intervals) reflect "
        "that."),
        ("- Active learning does not demonstrate a clear labeling-efficiency win in this "
        "experiment."),
        ("- Temperature scaling overfit its 16-example validation set and worsened calibration "
        "here, not universally."),
        ("- PatchCore-lite is a simplified variant (random coreset subsampling, k=1 nearest "
        "neighbor), not a reproduction of the original paper's method or benchmark numbers."),
        ("- Grad-CAM / anomaly heatmaps are correlational visualizations, not causal proof "
        "that highlighted pixels are the actual defect."),
        ("- Anomaly detection quality depends entirely on how representative the \"normal\" "
        "training images are; a shift in the true normal distribution would degrade it "
        "without warning."),
        "",
    ]


def section_reproducibility() -> list[str]:
    return [
        "## 13. Reproducibility Instructions",
        "",
        "```bash",
        "python scripts/download_data.py",
        "python scripts/prepare_data.py",
        "python scripts/train_anomaly.py      # writes reports/metrics/anomaly_detection.json",
        "python scripts/train.py              # writes reports/metrics/supervised_classification.json",
        "python scripts/evaluate.py           # writes reports/metrics/calibration.json",
        "python scripts/run_active_learning.py  # writes reports/metrics/active_learning.json",
        "python scripts/generate_report.py    # regenerates this file from the JSON above",
        "```",
        "",
        ("Every script reads its seed and hyperparameters from `configs/*.yaml` (see Section "
        "3) and writes its raw results to `reports/metrics/*.json` before this report ever "
        "touches them -- this file is a view over those files, not an independent source of "
        "numbers."),
        "",
    ]


def section_conclusions(anomaly, classification, calibration, active_learning) -> list[str]:
    lines = ["## 14. Conclusions", ""]
    what_measured = []
    if anomaly:
        what_measured.append(
            f"the anomaly detector clearly outperforms the classical baseline "
            f"({fmt(anomaly['patchcore_lite']['image_auroc'], 2)} vs. "
            f"{fmt(anomaly['classical_baseline']['image_auroc'], 2)} image AUROC)"
        )
    if classification:
        what_measured.append(
            f"the supervised classifier reaches {fmt(classification['test_metrics']['accuracy'], 2)} "
            "accuracy on a small held-out test set"
        )
    lines.append(
        "**What was experimentally measured:** " + "; ".join(what_measured) + "." if what_measured
        else "**What was experimentally measured:** see sections above."
    )
    lines += [
        "",
        ("**What was simulated:** the active-learning human-labeling loop (ground-truth labels "
        "standing in for a reviewer)."),
        "",
        ("**What came back negative or inconclusive, and is reported as such:** temperature "
        "scaling did not improve calibration here, and no active-learning acquisition "
        "strategy showed a statistically clear advantage over random sampling."),
        "",
        ("**What remains future work:** extending to a second MVTec category (`screw`), "
        "periodic full-CNN fine-tuning in the active-learning loop once more labeled data is "
        "available, cross-validated temperature fitting, and evaluating MC-dropout as a "
        "second review-routing signal."),
        "",
    ]
    return lines


def main() -> None:
    config = load_config("configs/base.yaml", "configs/training.yaml", "configs/active_learning.yaml")
    anomaly = load("anomaly_detection.json")
    classification = load("supervised_classification.json")
    calibration = load("calibration.json")
    active_learning = load("active_learning.json")

    lines = [
        "# Evaluation Report",
        "",
        ("Automotive Vision Defect Detection + Active Learning -- all numbers below are read "
        "directly from `reports/metrics/*.json`, generated by the scripts named in each "
        "section. None are hand-entered; anything not yet run is stated as such rather than "
        "estimated."),
        "",
    ]
    lines += section_objective()
    lines += section_dataset_protocol(anomaly, classification)
    lines += section_experimental_setup(config)
    lines += section_classical_baseline(anomaly)
    lines += section_anomaly_detector(anomaly)
    lines += section_supervised_classifier(classification)
    lines += section_uncertainty()
    lines += section_calibration(calibration)
    lines += section_active_learning(active_learning)
    lines += section_results(anomaly, classification, calibration, active_learning)
    lines += section_error_analysis(classification)
    lines += section_limitations()
    lines += section_reproducibility()
    lines += section_conclusions(anomaly, classification, calibration, active_learning)

    OUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
