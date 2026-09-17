"""End-to-end API + database integration: predict -> persisted row -> human review ->
review visible on the prediction detail -> reflected in /metrics."""

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
    image = Image.new("RGB", (64, 64), color=(80, 80, 80))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_full_predict_review_cycle_persists_and_links_correctly(client):
    predict_response = client.post(
        "/predict", files={"file": ("test.png", make_test_image_bytes(), "image/png")}
    )
    assert predict_response.status_code == 200
    prediction_id = predict_response.json()["id"]

    detail_before = client.get(f"/predictions/{prediction_id}")
    assert detail_before.status_code == 200
    assert detail_before.json()["review"] is None

    review_response = client.post(
        "/review",
        json={"prediction_id": prediction_id, "human_label": "bent", "reviewer_status": "confirmed_defect"},
    )
    assert review_response.status_code == 200
    assert review_response.json()["human_label"] == "bent"

    detail_after = client.get(f"/predictions/{prediction_id}")
    assert detail_after.status_code == 200
    assert detail_after.json()["review"]["reviewer_status"] == "confirmed_defect"

    metrics = client.get("/metrics").json()
    assert metrics["total_predictions"] >= 1
    assert metrics["total_reviews"] >= 1


def test_duplicate_review_is_rejected(client):
    prediction_id = client.post(
        "/predict", files={"file": ("test.png", make_test_image_bytes(), "image/png")}
    ).json()["id"]

    first = client.post(
        "/review",
        json={"prediction_id": prediction_id, "human_label": "bent", "reviewer_status": "confirmed_defect"},
    )
    assert first.status_code == 200

    second = client.post(
        "/review",
        json={"prediction_id": prediction_id, "human_label": "good", "reviewer_status": "corrected"},
    )
    assert second.status_code == 409


def test_review_for_nonexistent_prediction_returns_404(client):
    response = client.post(
        "/review",
        json={"prediction_id": 99999, "human_label": "bent", "reviewer_status": "confirmed_defect"},
    )
    assert response.status_code == 404


def test_batch_predict_persists_multiple_predictions(client):
    files = [
        ("files", ("a.png", make_test_image_bytes(), "image/png")),
        ("files", ("b.png", make_test_image_bytes(), "image/png")),
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["id"] != results[1]["id"]
