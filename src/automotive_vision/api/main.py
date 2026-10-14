"""FastAPI application entrypoint.

Run with:
    uvicorn automotive_vision.api.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from automotive_vision.api.routes import router
from automotive_vision.api.upload_limits import RequestSizeLimitMiddleware
from automotive_vision.database.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Automotive Vision Defect Detection API",
    description="Defect classification, anomaly scoring, uncertainty estimation, and human-review workflow.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(RequestSizeLimitMiddleware)
app.include_router(router)
