from automotive_vision.inference.postprocessing import decide_needs_review


def test_classifier_flag_alone_triggers_review():
    assert decide_needs_review(True, anomaly_score=0.0, anomaly_review_threshold=10.0) is True


def test_high_anomaly_score_triggers_review_even_if_classifier_confident():
    assert decide_needs_review(False, anomaly_score=5.0, anomaly_review_threshold=3.0) is True


def test_low_anomaly_score_and_confident_classifier_does_not_trigger_review():
    assert decide_needs_review(False, anomaly_score=1.0, anomaly_review_threshold=3.0) is False


def test_missing_anomaly_score_falls_back_to_classifier_flag_only():
    assert decide_needs_review(False, anomaly_score=None, anomaly_review_threshold=3.0) is False
    assert decide_needs_review(True, anomaly_score=None, anomaly_review_threshold=None) is True
