"""Thin data-access layer between the API routes and the ORM models."""

from sqlalchemy.orm import Session

from automotive_vision.database.models import Image, ModelVersion, Prediction, Review


def get_or_create_model_version(db: Session, version: str, model_type: str, metrics: dict | None = None) -> ModelVersion:
    existing = db.query(ModelVersion).filter_by(version=version).first()
    if existing:
        return existing
    model_version = ModelVersion(version=version, model_type=model_type, metrics=metrics)
    db.add(model_version)
    db.commit()
    db.refresh(model_version)
    return model_version


def create_image(db: Session, path: str, dataset_split: str | None = None, ground_truth_label: str | None = None) -> Image:
    image = Image(path=path, dataset_split=dataset_split, ground_truth_label=ground_truth_label)
    db.add(image)
    db.commit()
    db.refresh(image)
    return image


def create_prediction(
    db: Session,
    image_id: int,
    model_version_id: int,
    predicted_label: str,
    confidence: float,
    uncertainty: float,
    anomaly_score: float | None,
    needs_review: bool,
    heatmap_path: str | None = None,
) -> Prediction:
    prediction = Prediction(
        image_id=image_id,
        model_version_id=model_version_id,
        predicted_label=predicted_label,
        confidence=confidence,
        uncertainty=uncertainty,
        anomaly_score=anomaly_score,
        needs_review=needs_review,
        heatmap_path=heatmap_path,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def get_prediction(db: Session, prediction_id: int) -> Prediction | None:
    return db.query(Prediction).filter_by(id=prediction_id).first()


def create_review(db: Session, prediction_id: int, human_label: str, reviewer_status: str) -> Review:
    review = Review(prediction_id=prediction_id, human_label=human_label, reviewer_status=reviewer_status)
    db.add(review)
    db.commit()
    db.refresh(review)
    return review
