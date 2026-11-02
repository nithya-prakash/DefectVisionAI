from pathlib import Path

from scripts.check_data import scan, source_id


def _make(root: Path, split: str, name: str, rows: list[str]) -> None:
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    (root / split / "images" / f"{name}.jpg").write_bytes(b"x")
    (root / split / "labels" / f"{name}.txt").write_text("\n".join(rows))


def test_source_id_strips_roboflow_suffix():
    assert source_id("weld_1_jpg.rf.abc123.jpg") == "weld_1_jpg"
    assert source_id("plain.jpg") == "plain.jpg"


def test_scan_counts_boxes_and_flags_cross_split_sources(tmp_path):
    _make(tmp_path, "train", "w1.rf.aaa", ["0 0.5 0.5 0.2 0.2", "2 0.3 0.3 0.1 0.1"])
    _make(tmp_path, "valid", "w2.rf.bbb", ["1 0.5 0.5 0.2 0.2"])
    _make(tmp_path, "test", "w1.rf.ccc", ["0 0.5 0.5 0.2 0.2"])
    res = scan(tmp_path)
    assert res["splits"]["train"]["boxes"] == 2
    assert res["splits"]["train"]["classes"] == {"Bad Weld": 1, "Defect": 1}
    assert res["sources_in_multiple_splits"] == 1
    assert res["test_images_with_source_in_train"] == 1


def test_scan_reports_bad_rows(tmp_path):
    _make(tmp_path, "train", "w1", ["0 0.5 0.5 0.2", "5 0.5 0.5 0.2 0.2", "0 1.5 0.5 0.2 0.2"])
    _make(tmp_path, "valid", "w2", [])
    _make(tmp_path, "test", "w3", [])
    assert scan(tmp_path)["splits"]["train"]["n_problems"] == 3
