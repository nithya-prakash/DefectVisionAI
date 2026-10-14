"""Upload limits and image validation for the prediction endpoints.

Three layers, none of which trusts the client:
- `RequestSizeLimitMiddleware` rejects an oversized request before the multipart body is parsed
  (by Content-Length) and, for chunked or lying clients, while the body streams in. Starlette
  otherwise buffers the whole body before a route runs, so a size check inside the route is too late.
- `read_upload_bounded` reads an upload in bounded chunks as a second limit inside the route.
- `validate_image_bytes` decides from the bytes themselves: the declared Content-Type and file
  name are ignored. Only PNG, JPEG and BMP are accepted, with a pixel-count limit so a tiny file
  cannot decode into a huge image.

This reduces the abuse surface of a local demo API. It is not authentication, and it does not make
the service production-secure.
"""

from __future__ import annotations

import io
import json

from fastapi import HTTPException
from PIL import Image as PILImage
from PIL import UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_FILES = 16
MAX_IMAGE_PIXELS = 25_000_000
MULTIPART_OVERHEAD_BYTES = 64 * 1024
READ_CHUNK_BYTES = 64 * 1024
ALLOWED_FORMATS = {"PNG": "png", "JPEG": "jpg", "BMP": "bmp"}
LIMITED_PATHS = ("/predict", "/predict/batch")


def body_limit_for(path: str) -> int | None:
    """Largest request body (bytes) accepted for a path, or None if the path is not limited."""
    if path == "/predict":
        return MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
    if path == "/predict/batch":
        return MAX_UPLOAD_BYTES * MAX_BATCH_FILES + MULTIPART_OVERHEAD_BYTES
    return None


class _BodyTooLarge(HTTPException):
    """Raised while the body streams in. It is an HTTPException because FastAPI turns any other
    exception raised during body parsing into a 400, hiding that the real problem is size."""

    def __init__(self) -> None:
        super().__init__(status_code=413, detail="Request body too large")


class RequestSizeLimitMiddleware:
    """Pure ASGI middleware: 413 for oversized bodies, enforced before and during the read."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        limit = body_limit_for(scope.get("path", "")) if scope["type"] == "http" else None
        if limit is None or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await self._reject(send)
            return

        received = 0
        started = False

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await self._reject(send)

    @staticmethod
    async def _reject(send) -> None:
        body = json.dumps({"detail": "Request body too large"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"connection", b"close")],
            }
        )
        await send({"type": "http.response.body", "body": body})


async def read_upload_bounded(upload, limit: int | None = None) -> bytes:
    """Read an UploadFile in chunks, failing with 413 as soon as `limit` bytes are exceeded."""
    limit = MAX_UPLOAD_BYTES if limit is None else limit
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"Image exceeds the {limit} byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def validate_image_bytes(contents: bytes) -> str:
    """Validate the actual payload and return its file extension ('png', 'jpg' or 'bmp').

    400: empty, not an image, truncated or corrupt. 415: a real image in an unsupported format.
    422: dimensions beyond the pixel limit.
    """
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        with PILImage.open(io.BytesIO(contents)) as probe:
            image_format = probe.format
            width, height = probe.size
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, PILImage.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    if image_format not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=415, detail=f"Unsupported image format: {image_format} (allowed: PNG, JPEG, BMP)"
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=422, detail="Image dimensions are too large")
    try:
        with PILImage.open(io.BytesIO(contents)) as full:
            full.load()  # decodes every pixel, so truncated or corrupt data fails here
    except (OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc
    return ALLOWED_FORMATS[image_format]
