import csv
import json

import pytest

from mtmct.nvidia import CameraTiming, NvidiaDatasetAdapter, inspect_scene
from mtmct import download as download_module


def write_gt(path, duplicate=False):
    frames = {
        "0": [
            {"object type": "Person", "object id": 1,
             "2d bounding box visible": {"Camera_0000": [0, 0, 10, 20], "Camera_0001": [1, 0, 11, 20]}},
            {"object type": "Person", "object id": 2,
             "2d bounding box visible": {"Camera_0002": [0, 0, 10, 20]}},
            {"object type": "Forklift", "object id": 50,
             "2d bounding box visible": {"Camera_0000": [0, 0, 10, 20]}},
        ],
        "1": [
            {"object type": "Person", "object id": 1,
             "2d bounding box visible": {"Camera_0000": [0, 0, 10, 20], "Camera_0001": [1, 0, 11, 20]}},
            {"object type": "Person", "object id": 3,
             "2d bounding box visible": {"Camera_0003": [0, 0, 10, 20]}},
        ],
    }
    if duplicate:
        frames["1"].append({"object type": "Person", "object id": 1,
                            "2d bounding box visible": {"Camera_0000": [2, 0, 12, 20]}})
    path.write_text(json.dumps(frames), encoding="utf-8")


def test_timing_uses_offsets_and_pts():
    timing = CameraTiming(fps=10, frame_index_offset=5, time_offset_s=2, pts_s={7: 9.5})
    assert timing.timestamp_s(6) == pytest.approx(2.1)
    assert timing.timestamp_s(7) == 9.5


def test_validator_reports_missing_bbox_and_duplicate_id(tmp_path):
    gt = tmp_path / "ground_truth.json"
    write_gt(gt, duplicate=True)
    data = json.loads(gt.read_text(encoding="utf-8"))
    data["2"] = [{"object type": "Person", "object id": 4}]
    gt.write_text(json.dumps(data), encoding="utf-8")

    errors = NvidiaDatasetAdapter(gt).validate()
    assert any("duplicate object_id 1" in error for error in errors)
    assert any("missing visible bbox" in error for error in errors)


def test_non_strict_observations_skip_degenerate_visible_bbox(tmp_path):
    gt = tmp_path / "ground_truth.json"
    gt.write_text(json.dumps({"0": [{
        "object type": "Person", "object id": 1,
        "2d bounding box visible": {"Camera_0000": [5, 5, 5, 10]},
    }]}), encoding="utf-8")
    adapter = NvidiaDatasetAdapter(gt)
    assert len(adapter.validate()) == 1
    assert list(adapter.observations(strict=False)) == []


def test_metadata_download_uses_allow_patterns_without_videos(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(download_module, "snapshot_download", lambda **kwargs: captured.update(kwargs))
    download_module.download("val", "Warehouse_022", [], tmp_path, gt=True, videos=False)
    assert captured["allow_patterns"] == [
        "MTMC_Tracking_2026/val/Warehouse_022/calibration.json",
        "MTMC_Tracking_2026/val/Warehouse_022/ground_truth.json",
    ]


def test_inspect_scene_ranks_overlapping_cameras_without_exporting_gt_ids(tmp_path):
    scene = tmp_path / "Warehouse_test"
    videos = scene / "videos"
    videos.mkdir(parents=True)
    write_gt(scene / "ground_truth.json")
    (videos / "Camera_0000.mp4").touch()
    (videos / "Camera_0001.mp4").touch()

    report, pairs = inspect_scene(scene, tmp_path / "report", CameraTiming(fps=10))

    assert report["frame_count"] == 2
    assert report["unique_people"] == 3
    assert report["video_validation"]["missing_videos"] == [2, 3]
    assert pairs[0] == {
        "camera_a": 0, "camera_b": 1, "common_ids": 1,
        "co_visible_frames": 2, "positive_pairs": 2, "negative_pairs": 0,
    }
    pair_23 = next(row for row in pairs if row["camera_a"] == 2 and row["camera_b"] == 3)
    assert pair_23["co_visible_frames"] == 0
    report_text = (tmp_path / "report" / "scene_report.json").read_text(encoding="utf-8")
    assert "object_id" not in report_text
    with (tmp_path / "report" / "camera_pair_report.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 6
