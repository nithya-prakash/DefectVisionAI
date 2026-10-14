"""API behaviour under bad input: size limits, untrusted headers, invalid payloads, error shape."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from automotive_vision.api import routes
from automotive_vision.api import upload_limits as limits
from automotive_vision.api.dependencies import get_predictor
from automotive_vision.api.main import app
from tests.fake_predictor import FakePredictor


@pytest.fixture
def client():
    app.dependency_overrides[get_predictor] = lambda: FakePredictor()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def image_bytes(fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 48), (100, 100, 100)).save(buffer, format=fmt)
    return buffer.getvalue()


def total_predictions(client) -> int:
    return client.get("/metrics").json()["total_predictions"]


def test_declared_content_type_is_ignored_when_the_payload_is_a_valid_image(client) -> None:
    for declared in ("text/plain", "application/octet-stream", "image/gif"):
        response = client.post("/predict", files={"file": ("x.bin", image_bytes("PNG"), declared)})
        assert response.status_code == 200, declared


def test_a_valid_jpeg_labelled_as_png_is_accepted_by_content(client) -> None:
    response = client.post("/predict", files={"file": ("a.png", image_bytes("JPEG"), "image/png")})
    assert response.status_code == 200


def test_non_image_bytes_with_an_image_content_type_are_rejected(client) -> None:
    response = client.post("/predict", files={"file": ("a.png", b"just some text", "image/png")})
    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file is not a valid image"}


def test_unsupported_but_valid_image_format_is_415_even_if_labelled_png(client) -> None:
    response = client.post("/predict", files={"file": ("a.png", image_bytes("GIF"), "image/png")})
    assert response.status_code == 415
    assert "Unsupported image format" in response.json()["detail"]


def test_empty_upload_is_400(client) -> None:
    response = client.post("/predict", files={"file": ("a.png", b"", "image/png")})
    assert response.status_code == 400


def test_missing_file_field_is_a_422_validation_error(client) -> None:
    response = client.post("/predict", data={"unrelated": "x"})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_oversized_upload_is_413_and_nothing_is_stored(client, monkeypatch) -> None:
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", 2048)
    monkeypatch.setattr(limits, "MULTIPART_OVERHEAD_BYTES", 1024)
    before = total_predictions(client)
    response = client.post("/predict", files={"file": ("big.png", b"x" * 100_000, "image/png")})
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}
    assert total_predictions(client) == before


def test_upload_just_under_the_limit_still_works(client, monkeypatch) -> None:
    payload = image_bytes("PNG")
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", len(payload) + 100)
    response = client.post("/predict", files={"file": ("ok.png", payload, "image/png")})
    assert response.status_code == 200


def test_batch_with_too_many_files_is_413(client, monkeypatch) -> None:
    monkeypatch.setattr(routes, "MAX_BATCH_FILES", 2)
    files = [("files", (f"{i}.png", image_bytes(), "image/png")) for i in range(3)]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 413
    assert "Too many files" in response.json()["detail"]


def test_one_invalid_file_rejects_the_whole_batch_and_stores_nothing(client) -> None:
    before = total_predictions(client)
    files = [
        ("files", ("ok.png", image_bytes(), "image/png")),
        ("files", ("bad.png", b"not an image", "image/png")),
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 400
    assert total_predictions(client) == before


def test_valid_batch_still_works(client) -> None:
    files = [("files", (f"{i}.png", image_bytes(), "image/png")) for i in range(2)]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def noisy_png(size: int, seed: int = 0) -> bytes:
    """A PNG that does not compress much, so its byte size is predictable."""
    import numpy as np

    pixels = np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype="uint8")
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def test_a_batch_larger_in_total_than_one_image_is_accepted_when_each_image_fits(client, monkeypatch) -> None:
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", 3000)
    monkeypatch.setattr(limits, "MULTIPART_OVERHEAD_BYTES", 1024)
    images = [noisy_png(24, seed=i) for i in range(3)]
    assert all(len(img) < 3000 for img in images)
    assert sum(len(img) for img in images) > limits.body_limit_for("/predict")  # over the single-image body limit
    files = [("files", (f"{i}.png", img, "image/png")) for i, img in enumerate(images)]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    assert len(response.json()["results"]) == 3


def test_one_oversized_image_in_a_batch_is_413_and_nothing_is_stored(client, monkeypatch) -> None:
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", 3000)
    monkeypatch.setattr(limits, "MULTIPART_OVERHEAD_BYTES", 1024)
    big = noisy_png(40)
    assert len(big) > 3000
    before = total_predictions(client)
    files = [
        ("files", ("ok.png", noisy_png(24), "image/png")),
        ("files", ("big.png", big, "image/png")),
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 413
    assert total_predictions(client) == before


def test_a_chunked_body_without_content_length_is_cut_off_through_the_real_app(client, monkeypatch) -> None:
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", 2000)
    monkeypatch.setattr(limits, "MULTIPART_OVERHEAD_BYTES", 1000)

    def chunks():
        for _ in range(4):
            yield b"x" * 5000

    response = client.post(
        "/predict", content=chunks(), headers={"content-type": "multipart/form-data; boundary=xyz"}
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}
