import math

from welddet.boxes import Box, average_precision, iou, yolo_to_box


def test_yolo_to_box_roundtrip():
    b = yolo_to_box(1, 0.5, 0.5, 0.2, 0.4, 100, 200)
    assert (b.x1, b.y1, b.x2, b.y2, b.cls) == (40.0, 60.0, 60.0, 140.0, 1)


def test_iou_identical_disjoint_and_half():
    a = Box(0, 0, 10, 10, 0)
    assert iou(a, a) == 1.0
    assert iou(a, Box(20, 20, 30, 30, 0)) == 0.0
    assert math.isclose(iou(a, Box(5, 0, 15, 10, 0)), 50 / 150)


def test_ap_perfect_detections_is_one():
    gt = {"a": [Box(0, 0, 10, 10, 0)], "b": [Box(5, 5, 20, 20, 0)]}
    pred = {k: [Box(b.x1, b.y1, b.x2, b.y2, 0, 0.9)] for k, v in gt.items() for b in v}
    assert average_precision(pred, gt, 0) == 1.0


def test_ap_no_detections_is_zero():
    gt = {"a": [Box(0, 0, 10, 10, 0)]}
    assert average_precision({"a": []}, gt, 0) == 0.0


def test_ap_absent_class_is_nan():
    assert math.isnan(average_precision({"a": []}, {"a": [Box(0, 0, 1, 1, 0)]}, 2))


def test_duplicate_detection_is_a_false_positive():
    gt = {"a": [Box(0, 0, 10, 10, 0)]}
    pred = {"a": [Box(0, 0, 10, 10, 0, 0.9), Box(0, 0, 10, 10, 0, 0.8)]}
    # one true positive then one false positive: precision at full recall is 1.0
    assert average_precision(pred, gt, 0) == 1.0
    # a wrong high-scoring box ranked first lowers precision at the recall point
    worse = {"a": [Box(50, 50, 60, 60, 0, 0.95), Box(0, 0, 10, 10, 0, 0.8)]}
    assert math.isclose(average_precision(worse, gt, 0), 0.5)


def test_wrong_class_does_not_count():
    gt = {"a": [Box(0, 0, 10, 10, 0)]}
    pred = {"a": [Box(0, 0, 10, 10, 1, 0.9)]}
    assert average_precision(pred, gt, 0) == 0.0
