import csv
import json

from mtmct.wildtrack import WildtrackAdapter, evaluate, inspect_dataset


def _dataset(tmp_path):
    annotations = tmp_path / "annotation_positions"
    annotations.mkdir()
    for raw_frame, person_id in [(0, 10), (5, 11)]:
        views = [
            {"viewNum": camera, "xmin": 10 + camera, "ymin": 20,
             "xmax": 30 + camera, "ymax": 60}
            for camera in range(7)
        ]
        views.append({"viewNum": 7, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1})
        (annotations / f"{raw_frame:08d}.json").write_text(
            json.dumps([{"personID": person_id, "views": views}]), encoding="utf-8"
        )
    return tmp_path


def _predictions(root, cameras):
    path = root / f"pred_{len(cameras)}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "frame_id", "camera_id", "global_id", "x1", "y1", "x2", "y2"
        ])
        writer.writeheader()
        for frame, identity in [(0, 10), (5, 11)]:
            for camera in cameras:
                writer.writerow({"frame_id": frame, "camera_id": camera,
                                 "global_id": identity, "x1": 10 + camera,
                                 "y1": 20, "x2": 30 + camera, "y2": 60})
    return path


def test_adapter_uses_annotated_moments_without_interpolation(tmp_path):
    root = _dataset(tmp_path)
    observations = list(WildtrackAdapter(root).observations())
    assert {item.raw_frame for item in observations} == {0, 5}
    assert {item.timestamp_s for item in observations} == {0.0, 0.5}
    assert len(observations) == 14
    report = inspect_dataset(root)
    assert report["cameras"] == list(range(7))
    assert report["interpolation"] is False
    assert report["calibration_required"] is False
    assert report["map_required"] is False


def test_perfect_bbox_identity_evaluation_for_three_and_seven_cameras(tmp_path):
    root = _dataset(tmp_path)
    for cameras in [list(range(3)), list(range(7))]:
        report = evaluate(_predictions(root, cameras), root, cameras)
        assert report["global"]["IDF1"] == 1.0
        assert report["pairwise_association"]["F1"] == 1.0
        assert report["geometry_used"] is False
        assert report["interpolation"] is False
