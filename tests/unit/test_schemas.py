import pytest
from pydantic import ValidationError

from automotive_vision.api.schemas import PredictionResponse, ReviewRequest


def test_prediction_response_accepts_valid_confidence():
    response = PredictionResponse(
        id=1,
        prediction="bent",
        confidence=0.9,
        uncertainty=0.1,
        margin=0.5,
        anomaly_score=None,
        needs_human_review=False,
        heatmap_path=None,
        model_version="v1",
    )
    assert response.confidence == 0.9


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1])
def test_prediction_response_rejects_out_of_range_confidence(bad_confidence):
    with pytest.raises(ValidationError):
        PredictionResponse(
            id=1,
            prediction="bent",
            confidence=bad_confidence,
            uncertainty=0.1,
            margin=0.5,
            anomaly_score=None,
            needs_human_review=False,
            heatmap_path=None,
            model_version="v1",
        )


def test_review_request_requires_all_fields():
    with pytest.raises(ValidationError):
        ReviewRequest(prediction_id=1)  # missing human_label / reviewer_status
