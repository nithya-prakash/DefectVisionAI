"""API route handlers.

Uploaded images are saved under data/uploads/ by their database id, never by any
client-supplied filename or path, so responses never leak server filesystem layout.
"""

import io
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from PIL import Image as PILImage
from sqlalchemy import func
from sqlalchemy.orm import Session

from automotive_vision.api.dependencies import get_db, get_predictor
from automotive_vision.api.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    PredictionDetailResponse,
    PredictionResponse,
    ReviewRequest,
    ReviewResponse,
)
from automotive_vision.database import repository
from automotive_vision.database.models import Prediction, Review
from automotive_vision.inference.predictor import Predictor

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/bmp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _validate_and_read(upload: UploadFile, contents: bytes) -> None:
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {upload.content_type}")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10MB limit")
    try:
        PILImage.open(io.BytesIO(contents)).verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc


def _persist_and_predict(upload: UploadFile, contents: bytes, db: Session, predictor: Predictor) -> PredictionResponse:
    result = predictor.predict(contents)

    image_record = repository.create_image(db, path="")  # path filled in after we know the id
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    image_path = UPLOAD_DIR / f"{image_record.id}.png"
    image_path.write_bytes(contents)
    image_record.path = str(image_path)
    db.commit()

    model_version = repository.get_or_create_model_version(
        db, version=result["model_version"], model_type="resnet18_classifier"
    )
    prediction = repository.create_prediction(
        db,
        image_id=image_record.id,
        model_version_id=model_version.id,
        predicted_label=result["prediction"],
        confidence=result["confidence"],
        uncertainty=result["uncertainty"],
        anomaly_score=result["anomaly_score"],
        needs_review=result["needs_human_review"],
        heatmap_path=result["heatmap_path"],
    )
    return PredictionResponse(
        id=prediction.id,
        prediction=prediction.predicted_label,
        confidence=prediction.confidence,
        uncertainty=prediction.uncertainty,
        margin=result["margin"],
        anomaly_score=prediction.anomaly_score,
        needs_human_review=prediction.needs_review,
        heatmap_path=prediction.heatmap_path,
        model_version=model_version.version,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile,
    db: Session = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
) -> PredictionResponse:
    contents = await file.read()
    _validate_and_read(file, contents)
    return _persist_and_predict(file, contents, db, predictor)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: list[UploadFile],
    db: Session = Depends(get_db),
    predictor: Predictor = Depends(get_predictor),
) -> BatchPredictionResponse:
    responses = []
    for file in files:
        contents = await file.read()
        _validate_and_read(file, contents)
        responses.append(_persist_and_predict(file, contents, db, predictor))
    return BatchPredictionResponse(results=responses)


@router.post("/review", response_model=ReviewResponse)
def submit_review(request: ReviewRequest, db: Session = Depends(get_db)) -> ReviewResponse:
    prediction = repository.get_prediction(db, request.prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    existing = db.query(Review).filter_by(prediction_id=request.prediction_id).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="This prediction already has a review")

    review = repository.create_review(
        db,
        prediction_id=request.prediction_id,
        human_label=request.human_label,
        reviewer_status=request.reviewer_status,
    )
    return ReviewResponse(
        id=review.id,
        prediction_id=review.prediction_id,
        human_label=review.human_label,
        reviewer_status=review.reviewer_status,
        created_at=review.created_at,
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionDetailResponse)
def get_prediction_detail(prediction_id: int, db: Session = Depends(get_db)) -> PredictionDetailResponse:
    prediction = repository.get_prediction(db, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    review_response = None
    if prediction.review is not None:
        review_response = ReviewResponse(
            id=prediction.review.id,
            prediction_id=prediction.review.prediction_id,
            human_label=prediction.review.human_label,
            reviewer_status=prediction.review.reviewer_status,
            created_at=prediction.review.created_at,
        )

    return PredictionDetailResponse(
        id=prediction.id,
        image_path=prediction.image.path,
        predicted_label=prediction.predicted_label,
        confidence=prediction.confidence,
        uncertainty=prediction.uncertainty,
        anomaly_score=prediction.anomaly_score,
        needs_review=prediction.needs_review,
        model_version=prediction.model_version.version,
        created_at=prediction.created_at,
        review=review_response,
    )


@router.get("/health", response_model=HealthResponse)
def health(predictor: Predictor = Depends(get_predictor)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        classifier_loaded=predictor.model is not None,
        anomaly_detector_loaded=predictor.anomaly_detector is not None,
    )


@router.get("/metrics")
def metrics(db: Session = Depends(get_db)) -> dict:
    """Application-level metrics (prediction/review counts), not Prometheus format."""
    total_predictions = db.query(func.count(Prediction.id)).scalar()
    pending_review = db.query(func.count(Prediction.id)).filter(
        Prediction.needs_review.is_(True), ~Prediction.review.has()
    ).scalar()
    total_reviews = db.query(func.count(Review.id)).scalar()
    return {
        "total_predictions": total_predictions,
        "pending_review": pending_review,
        "total_reviews": total_reviews,
    }
