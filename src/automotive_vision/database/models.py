"""SQLAlchemy ORM models: images, predictions, reviews, model_versions.

Images are stored on disk (see api/routes.py's upload handling); only the file
reference is kept in the database, per the "don't store large binaries in Postgres
unless there's a compelling reason" guidance.
"""

import datetime as dt

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String, nullable=False)
    dataset_split: Mapped[str | None] = mapped_column(String, nullable=True)
    ground_truth_label: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="image")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    model_type: Mapped[str] = mapped_column(String, nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), nullable=False)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), nullable=False)
    predicted_label: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    uncertainty: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    heatmap_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    image: Mapped["Image"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")
    review: Mapped["Review | None"] = relationship(back_populates="prediction", uselist=False)


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), nullable=False, unique=True)
    human_label: Mapped[str] = mapped_column(String, nullable=False)
    reviewer_status: Mapped[str] = mapped_column(String, nullable=False)  # confirmed_defect | confirmed_normal | corrected
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    prediction: Mapped["Prediction"] = relationship(back_populates="review")
