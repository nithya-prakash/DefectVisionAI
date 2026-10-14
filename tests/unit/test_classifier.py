"""Classifier forward pass, tuning flags, logits collection and end-to-end Predictor behaviour."""

import io

import pytest
import torch
from PIL import Image

from automotive_vision.inference.predictor import Predictor
from automotive_vision.models.anomaly_detector import PatchCoreLite
from automotive_vision.models.classifier import DefectClassifier
from automotive_vision.training.train_classifier import collect_logits, compute_class_weights

NAMES = ["bent", "color", "flip", "good", "scratch"]
SIZE = 64


@pytest.fixture
def model(offline_resnet) -> DefectClassifier:
    return DefectClassifier(num_classes=5).eval()


def test_forward_gives_finite_logits_of_the_right_shape(model) -> None:
    out = model(torch.randn(3, 3, SIZE, SIZE))
    assert out.shape == (3, 5) and torch.isfinite(out).all()


def test_eval_mode_is_deterministic(model) -> None:
    x = torch.randn(2, 3, SIZE, SIZE)
    assert torch.equal(model(x), model(x))


def test_only_layer4_and_the_head_are_trainable_when_frozen(model) -> None:
    trainable = {n.split(".")[1] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"layer4", "fc"}
    assert all(p.requires_grad for p in model.trainable_parameters())


def test_unfrozen_model_trains_every_parameter(offline_resnet) -> None:
    full = DefectClassifier(num_classes=5, freeze_until_layer4=False)
    assert all(p.requires_grad for p in full.parameters())


def test_class_weights_favour_rare_classes_and_survive_an_absent_class() -> None:
    w = compute_class_weights([0, 0, 0, 1], num_classes=3)
    assert w[1] > w[0]
    assert torch.isfinite(w).all()  # class 2 has no samples


def test_collect_logits_matches_direct_forward(model) -> None:
    images = torch.randn(4, 3, SIZE, SIZE)
    labels = torch.tensor([0, 1, 2, 3])
    batches = [{"image": images[:2], "label": labels[:2]}, {"image": images[2:], "label": labels[2:]}]
    logits, got_labels = collect_logits(model, batches, "cpu")
    assert logits.shape == (4, 5) and torch.equal(got_labels, labels)
    assert torch.allclose(logits, model(images), atol=1e-5)


def png_bytes(color=(110, 110, 110)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 80), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def predictor_factory(offline_resnet, tmp_path):
    clf = DefectClassifier(num_classes=5)
    checkpoint = tmp_path / "clf.pt"
    torch.save(
        {
            "model_state_dict": clf.state_dict(),
            "class_names": NAMES,
            "config": {"freeze_until_layer4": True, "dropout_p": 0.3},
        },
        checkpoint,
    )
    detector = PatchCoreLite(image_size=SIZE, memory_bank_size=300, seed=1)
    g = torch.Generator().manual_seed(0)
    detector.fit([{"image": 0.1 * torch.randn(3, 3, SIZE, SIZE, generator=g)}])
    bank = tmp_path / "bank.pt"
    detector.save(bank)

    def build(temperature: float = 1.0, threshold: float = 0.74, anomaly_threshold=None):
        config = {
            "dataset": {"image_size": SIZE, "normalize_mean": [0.485, 0.456, 0.406],
                        "normalize_std": [0.229, 0.224, 0.225]},
            "inference": {
                "checkpoint_path": str(checkpoint), "anomaly_checkpoint_path": str(bank),
                "confidence_threshold": threshold, "temperature": temperature,
                "entropy_threshold": None, "anomaly_review_threshold": anomaly_threshold,
            },
        }
        return Predictor(config, device="cpu", heatmap_dir=str(tmp_path / "heatmaps"))

    return build


def test_predictor_returns_a_complete_well_formed_result(predictor_factory) -> None:
    result = predictor_factory().predict(png_bytes(), save_heatmap=False)
    assert result["prediction"] in NAMES
    assert 0.2 <= result["confidence"] <= 1.0
    assert isinstance(result["needs_human_review"], bool)
    assert isinstance(result["anomaly_score"], float)
    assert result["model_version"]


def test_review_follows_confidence_only_while_the_anomaly_gate_is_off(predictor_factory) -> None:
    result = predictor_factory(threshold=0.74, anomaly_threshold=None).predict(png_bytes(), False)
    assert result["needs_human_review"] == (result["confidence"] < 0.74)
    always = predictor_factory(threshold=0.0, anomaly_threshold=None).predict(png_bytes(), False)
    assert always["needs_human_review"] is False
    never = predictor_factory(threshold=1.1, anomaly_threshold=None).predict(png_bytes(), False)
    assert never["needs_human_review"] is True


def test_a_higher_temperature_lowers_confidence_without_changing_the_label(predictor_factory) -> None:
    raw = predictor_factory(temperature=1.0).predict(png_bytes(), False)
    soft = predictor_factory(temperature=1.4568).predict(png_bytes(), False)
    assert soft["prediction"] == raw["prediction"]
    assert soft["confidence"] < raw["confidence"]


def test_the_anomaly_gate_routes_review_only_when_a_threshold_is_set(predictor_factory) -> None:
    on = predictor_factory(threshold=0.0, anomaly_threshold=0.0).predict(png_bytes(), False)
    assert on["needs_human_review"] is True  # any anomaly score >= 0 is flagged
    off = predictor_factory(threshold=0.0, anomaly_threshold=None).predict(png_bytes(), False)
    assert off["needs_human_review"] is False


def test_the_predictor_rejects_bytes_that_are_not_an_image(predictor_factory) -> None:
    with pytest.raises(Exception):  # noqa: B017 (PIL's UnidentifiedImageError)
        predictor_factory().predict(b"definitely not an image", save_heatmap=False)
