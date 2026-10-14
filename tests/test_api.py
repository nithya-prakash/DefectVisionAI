"""API smoke tests. The real model is mocked out (see fake_predictor.py) so these run
fast and don't require a trained checkpoint to exist."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from automotive_vision.api.dependencies import get_predictor
from automotive_vision.api.main import app
from tests.fake_predictor import FakePredictor


@pytest.fixture
def client():
    app.dependency_overrides[get_predictor] = lambda: FakePredictor()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def make_test_image_bytes() -> bytes:
    image = Image.new("RGB", (64, 64), color=(120, 120, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_models_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["classifier_loaded"] is True


def test_predict_returns_structured_prediction(client):
    response = client.post(
        "/predict", files={"file": ("test.png", make_test_image_bytes(), "image/png")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "bent"
    assert body["needs_human_review"] is True
    assert 0 <= body["confidence"] <= 1


def test_predict_rejects_non_image_payload_whatever_the_declared_type(client):
    """The payload is validated, not the client-supplied Content-Type (see api/upload_limits.py)."""
    response = client.post(
        "/predict", files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400


def test_predict_rejects_corrupt_image_bytes(client):
    response = client.post(
        "/predict", files={"file": ("test.png", b"\x89PNGnotreallyapng", "image/png")}
    )
    assert response.status_code == 400


def test_get_missing_prediction_returns_404(client):
    response = client.get("/predictions/9999")
    assert response.status_code == 404


def test_metrics_endpoint_returns_counts(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "total_predictions" in body
