"""Upload limits and content validation, independent of the client-supplied headers."""

import asyncio
import io

import pytest
from fastapi import HTTPException
from PIL import Image

from automotive_vision.api import upload_limits as limits


def image_bytes(fmt: str, size=(32, 32), mode="RGB", color=(90, 90, 90)) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color if mode == "RGB" else 90).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.parametrize(("fmt", "extension"), [("PNG", "png"), ("JPEG", "jpg"), ("BMP", "bmp")])
def test_allowed_formats_are_accepted_and_named_from_their_content(fmt, extension) -> None:
    assert limits.validate_image_bytes(image_bytes(fmt)) == extension


def test_a_real_image_in_an_unsupported_format_is_415() -> None:
    with pytest.raises(HTTPException) as error:
        limits.validate_image_bytes(image_bytes("GIF", mode="RGB"))
    assert error.value.status_code == 415


@pytest.mark.parametrize(
    "payload",
    [b"", b"plain text, not an image", b"\x89PNG\r\n\x1a\nnot really a png", image_bytes("PNG")[:40]],
)
def test_empty_corrupt_or_non_image_payloads_are_400(payload) -> None:
    with pytest.raises(HTTPException) as error:
        limits.validate_image_bytes(payload)
    assert error.value.status_code == 400


def test_a_truncated_image_body_is_400() -> None:
    data = image_bytes("PNG", size=(256, 256), color=(10, 200, 30))
    noisy = io.BytesIO()
    Image.effect_noise((256, 256), 80).convert("RGB").save(noisy, format="PNG")
    truncated = noisy.getvalue()[: len(noisy.getvalue()) // 2]
    assert data  # the valid reference decodes; the truncated noisy one must not
    with pytest.raises(HTTPException) as error:
        limits.validate_image_bytes(truncated)
    assert error.value.status_code == 400


def test_images_beyond_the_pixel_limit_are_422_even_when_the_file_is_tiny(monkeypatch) -> None:
    monkeypatch.setattr(limits, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(HTTPException) as error:
        limits.validate_image_bytes(image_bytes("PNG", size=(64, 64)))
    assert error.value.status_code == 422


class FakeUpload:
    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)
        self.bytes_read = 0

    async def read(self, n: int = -1) -> bytes:
        chunk = self._buffer.read(n)
        self.bytes_read += len(chunk)
        return chunk


def test_bounded_read_returns_everything_within_the_limit() -> None:
    data = b"x" * 5000
    assert asyncio.run(limits.read_upload_bounded(FakeUpload(data), limit=6000)) == data


def test_bounded_read_stops_early_with_413_when_the_limit_is_exceeded(monkeypatch) -> None:
    monkeypatch.setattr(limits, "READ_CHUNK_BYTES", 1024)
    upload = FakeUpload(b"x" * 50_000)
    with pytest.raises(HTTPException) as error:
        asyncio.run(limits.read_upload_bounded(upload, limit=2048))
    assert error.value.status_code == 413
    assert upload.bytes_read <= 2048 + 1024  # never buffered the whole 50 kB


def test_body_limits_per_path() -> None:
    assert limits.body_limit_for("/predict") == limits.MAX_UPLOAD_BYTES + limits.MULTIPART_OVERHEAD_BYTES
    assert limits.body_limit_for("/predict/batch") > limits.body_limit_for("/predict")
    assert limits.body_limit_for("/health") is None


async def _drive(path, headers, chunks, method="POST"):
    """Run the middleware around an app that reads the whole body; return (status, inner_ran)."""
    ran = {"inner": False}

    async def inner(scope, receive, send):
        ran["inner"] = True
        while True:
            message = await receive()
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    messages = [
        {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
        for i, c in enumerate(chunks)
    ] or [{"type": "http.request", "body": b"", "more_body": False}]
    queue = list(messages)
    sent = []

    async def receive():
        return queue.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": method, "path": path, "headers": headers}
    await limits.RequestSizeLimitMiddleware(inner)(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, ran["inner"]


def test_middleware_rejects_a_declared_oversize_body_without_running_the_app(monkeypatch) -> None:
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(limits, "MULTIPART_OVERHEAD_BYTES", 0)
    headers = [(b"content-length", b"5000")]
    assert asyncio.run(_drive("/predict", headers, [b"x" * 100])) == (413, False)


def test_middleware_stops_a_streamed_body_that_exceeds_the_limit(monkeypatch) -> None:
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(limits, "MULTIPART_OVERHEAD_BYTES", 0)
    # no Content-Length (chunked), or one that understates the real size
    chunks = [b"x" * 400] * 4
    assert asyncio.run(_drive("/predict", [], chunks)) == (413, True)
    assert asyncio.run(_drive("/predict", [(b"content-length", b"10")], chunks)) == (413, True)


def test_middleware_passes_normal_and_unlimited_requests(monkeypatch) -> None:
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(limits, "MULTIPART_OVERHEAD_BYTES", 0)
    assert asyncio.run(_drive("/predict", [], [b"x" * 500])) == (200, True)
    assert asyncio.run(_drive("/health", [], [b"x" * 5000])) == (200, True)
    assert asyncio.run(_drive("/predict", [], [b"x" * 5000], method="GET")) == (200, True)
