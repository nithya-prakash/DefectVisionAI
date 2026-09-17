"""End-to-end prediction: image in, structured decision out.

Loads the trained classifier checkpoint and (if present) the PatchCore-lite memory
bank once at process startup, and exposes a single predict() call used by both the
FastAPI service and scripts/run_inference.py.
"""

import io
from pathlib import Path

import torch
from PIL import Image

from automotive_vision.data.preprocessing import build_eval_transform
from automotive_vision.inference.explainability import (
    anomaly_heatmap_overlay,
    save_overlay,
)
from automotive_vision.inference.postprocessing import decide_needs_review
from automotive_vision.models.anomaly_detector import PatchCoreLite
from automotive_vision.models.classifier import DefectClassifier
from automotive_vision.models.uncertainty import softmax_uncertainty
from automotive_vision.utils.logging import get_logger

logger = get_logger(__name__)


class Predictor:
    def __init__(self, config: dict, device: str = "cpu", heatmap_dir: str = "data/heatmaps"):
        self.config = config
        self.device = device
        self.dataset_cfg = config["dataset"]
        self.inference_cfg = config["inference"]
        self.heatmap_dir = Path(heatmap_dir)
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)

        self.transform = build_eval_transform(
            self.dataset_cfg["image_size"], self.dataset_cfg["normalize_mean"], self.dataset_cfg["normalize_std"]
        )

        checkpoint_path = Path(self.inference_cfg["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"{checkpoint_path} not found — run scripts/train.py first.")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.class_names = checkpoint["class_names"]
        train_cfg = checkpoint["config"]
        self.model = DefectClassifier(
            num_classes=len(self.class_names),
            freeze_until_layer4=train_cfg["freeze_until_layer4"],
            dropout_p=train_cfg["dropout_p"],
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model_version = "defect_classifier-v1"

        memory_bank_path = Path("models/checkpoints/patchcore_memory_bank.pt")
        self.anomaly_detector = None
        if memory_bank_path.exists():
            self.anomaly_detector = PatchCoreLite.load(memory_bank_path, device=device)
        else:
            logger.warning(f"{memory_bank_path} not found — predictions will omit anomaly_score.")

    @torch.no_grad()
    def predict(self, image_bytes: bytes, save_heatmap: bool = True) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = self.transform(image).to(self.device)

        logits = self.model(image_tensor.unsqueeze(0))[0].cpu().numpy()
        result = softmax_uncertainty(
            logits,
            self.class_names,
            confidence_threshold=self.inference_cfg["confidence_threshold"],
            entropy_threshold=self.inference_cfg.get("entropy_threshold"),
        )

        anomaly_score = None
        heatmap_path = None
        if self.anomaly_detector is not None:
            anomaly_score, pixel_map = self.anomaly_detector.score_image(image_tensor)
            if save_heatmap:
                overlay = anomaly_heatmap_overlay(
                    image_tensor, pixel_map, self.dataset_cfg["normalize_mean"], self.dataset_cfg["normalize_std"]
                )
                heatmap_path = save_overlay(
                    overlay, self.heatmap_dir / f"anomaly_{abs(hash(image_bytes))}.png"
                )

        needs_review = decide_needs_review(
            result.needs_review,
            anomaly_score,
            self.inference_cfg.get("anomaly_review_threshold"),
        )

        return {
            "prediction": result.label,
            "confidence": result.probability,
            "uncertainty": result.entropy,
            "margin": result.margin,
            "anomaly_score": anomaly_score,
            "needs_human_review": needs_review,
            "heatmap_path": heatmap_path,
            "model_version": self.model_version,
        }
