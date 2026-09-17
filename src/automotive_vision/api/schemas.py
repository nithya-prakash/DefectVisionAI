"""Pydantic request/response schemas for the FastAPI service."""

import datetime as dt

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    id: int
    prediction: str
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    margin: float
    anomaly_score: float | None = None
    needs_human_review: bool
    heatmap_path: str | None = None
    model_version: str


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


class ReviewRequest(BaseModel):
    prediction_id: int
    human_label: str
    reviewer_status: str = Field(description="confirmed_defect | confirmed_normal | corrected")


class ReviewResponse(BaseModel):
    id: int
    prediction_id: int
    human_label: str
    reviewer_status: str
    created_at: dt.datetime


class PredictionDetailResponse(BaseModel):
    id: int
    image_path: str
    predicted_label: str
    confidence: float
    uncertainty: float
    anomaly_score: float | None
    needs_review: bool
    model_version: str
    created_at: dt.datetime
    review: ReviewResponse | None = None


class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    anomaly_detector_loaded: bool
