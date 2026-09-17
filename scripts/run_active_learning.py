#!/usr/bin/env python3
"""Active-learning acquisition-strategy comparison.

Usage:
    python scripts/run_active_learning.py

Compares random / uncertainty / diversity / hybrid acquisition on the same 80-image
labeled pool and 35-image held-out test set used elsewhere in this project (see
data/README.md), repeated over multiple seeds for a confidence band. Writes real
results to reports/metrics/active_learning.json and a comparison plot to
reports/figures/active_learning_curves.png.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from automotive_vision.active_learning.acquisition import RandomAcquisition  # noqa: E402
from automotive_vision.active_learning.diversity_sampling import DiversityAcquisition  # noqa: E402
from automotive_vision.active_learning.hybrid_sampling import HybridAcquisition  # noqa: E402
from automotive_vision.active_learning.simulation import run_simulation  # noqa: E402
from automotive_vision.active_learning.uncertainty_sampling import UncertaintyAcquisition  # noqa: E402
from automotive_vision.data.dataset import ClassificationDataset, scan_mvtec_category  # noqa: E402
from automotive_vision.data.preprocessing import build_eval_transform  # noqa: E402
from automotive_vision.data.splitting import labeled_pool_split  # noqa: E402
from automotive_vision.models.embeddings import PatchFeatureExtractor  # noqa: E402
from automotive_vision.utils.config import load_config  # noqa: E402
from automotive_vision.utils.logging import get_logger  # noqa: E402
from automotive_vision.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)


def resolve_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return requested


@torch.no_grad()
def compute_embeddings(records, label_to_idx, dataset_cfg, device, batch_size=16) -> tuple[np.ndarray, np.ndarray]:
    transform = build_eval_transform(
        dataset_cfg["image_size"], dataset_cfg["normalize_mean"], dataset_cfg["normalize_std"]
    )
    dataset = ClassificationDataset(records, label_to_idx, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    extractor = PatchFeatureExtractor().to(device)

    all_embeddings, all_labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        embeddings = extractor.global_embedding(images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(batch["label"].numpy())
    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--al-config", default="configs/active_learning.yaml")
    args = parser.parse_args()

    config = load_config(args.base_config, args.training_config, args.al_config)
    set_seed(config["seed"])
    dataset_cfg = config["dataset"]
    al_cfg = config["active_learning"]
    device = resolve_device(al_cfg["device"])

    records = scan_mvtec_category(dataset_cfg["category_dir"])
    labeled_records = [r for r in records if r.official_split == "test"]
    pool_records, held_out_records = labeled_pool_split(
        labeled_records, test_size=config["training"]["held_out_fraction"], seed=config["seed"]
    )
    class_names = sorted({r.defect_type for r in labeled_records})
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    logger.info(f"Pool: {len(pool_records)} images, held-out test: {len(held_out_records)} images")

    logger.info("Computing frozen ResNet18 embeddings (one-time cost for all AL rounds)...")
    start = time.time()
    pool_embeddings, pool_labels = compute_embeddings(pool_records, label_to_idx, dataset_cfg, device)
    test_embeddings, test_labels = compute_embeddings(held_out_records, label_to_idx, dataset_cfg, device)
    logger.info(f"Embeddings computed in {time.time() - start:.1f}s")

    strategy_factories = {
        "random": lambda: RandomAcquisition(),
        "uncertainty": lambda: UncertaintyAcquisition(),
        "diversity": lambda: DiversityAcquisition(),
        "hybrid": lambda: HybridAcquisition(alpha=al_cfg["alpha"]),
    }

    all_results = {}
    for strategy_name in al_cfg["strategies"]:
        logger.info(f"Running strategy: {strategy_name}")
        runs = []
        for repeat in range(al_cfg["n_repeats"]):
            seed = al_cfg["base_seed"] + repeat
            strategy = strategy_factories[strategy_name]()
            result = run_simulation(
                pool_embeddings,
                pool_labels,
                test_embeddings,
                test_labels,
                num_classes=len(class_names),
                strategy=strategy,
                initial_labeled_fraction=al_cfg["initial_labeled_fraction"],
                acquisition_size=al_cfg["acquisition_size"],
                rounds=al_cfg["rounds"],
                seed=seed,
            )
            runs.append(result)
        all_results[strategy_name] = runs
        final_f1s = [r.macro_f1[-1] for r in runs]
        logger.info(
            f"{strategy_name}: final macro-F1 mean={np.mean(final_f1s):.4f} std={np.std(final_f1s):.4f}"
        )

    reports_dir = Path("reports")
    (reports_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    serializable = {
        "n_pool": len(pool_records),
        "n_held_out_test": len(held_out_records),
        "class_names": class_names,
        "config": al_cfg,
        "results": {
            strategy: [
                {
                    "seed": r.seed,
                    "n_labeled": r.n_labeled,
                    "labeled_fraction": r.labeled_fraction,
                    "macro_f1": r.macro_f1,
                    "accuracy": r.accuracy,
                }
                for r in runs
            ]
            for strategy, runs in all_results.items()
        },
    }
    out_json = reports_dir / "metrics" / "active_learning.json"
    out_json.write_text(json.dumps(serializable, indent=2))
    logger.info(f"Wrote {out_json}")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"random": "gray", "uncertainty": "tab:orange", "diversity": "tab:green", "hybrid": "tab:blue"}
    for strategy_name, runs in all_results.items():
        n_labeled = np.array(runs[0].n_labeled)
        f1_matrix = np.array([r.macro_f1 for r in runs])  # (n_repeats, n_rounds+1)
        mean_f1 = f1_matrix.mean(axis=0)
        std_f1 = f1_matrix.std(axis=0)
        color = colors.get(strategy_name)
        ax.plot(n_labeled, mean_f1, label=strategy_name, color=color, marker="o")
        ax.fill_between(n_labeled, mean_f1 - std_f1, mean_f1 + std_f1, color=color, alpha=0.15)

    ax.set_xlabel(f"Number of labeled samples (of {len(pool_records)} pool images)")
    ax.set_ylabel(f"Macro-F1 on held-out test (n={len(held_out_records)})")
    ax.set_title(f"Active learning: acquisition strategy comparison ({al_cfg['n_repeats']} seeds, shaded = ±1 std)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "figures" / "active_learning_curves.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote reports/figures/active_learning_curves.png")


if __name__ == "__main__":
    main()
