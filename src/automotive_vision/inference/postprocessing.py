"""Combines classifier uncertainty and anomaly score into one needs_review decision.

Kept as a pure function (not a class) since it's a small, easily-tested rule: review is
triggered if either signal is uncertain, since either being true is reason enough for a
human to look — a classifier that is confident but flags a high anomaly score (or vice
versa) is exactly the disagreement case a human should resolve.
"""


def decide_needs_review(
    classifier_needs_review: bool,
    anomaly_score: float | None,
    anomaly_review_threshold: float | None,
) -> bool:
    if classifier_needs_review:
        return True
    if anomaly_score is not None and anomaly_review_threshold is not None:
        return anomaly_score >= anomaly_review_threshold
    return False
