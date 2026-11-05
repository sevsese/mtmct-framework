import csv

from mtmct.association import Config, pair_cost
from mtmct.citic import evaluate_world_positions
from mtmct.time_sync import CameraTransitionModel, track_times


def track(camera, start, end):
    return {"camera_id": camera, "local_id": camera + 1, "start_frame": start,
            "end_frame": end, "embedding": [1, 0], "positions": []}


def test_offsets_and_unknown_topology_do_not_forbid_match():
    left, right = track(0, 0, 30), track(1, 0, 30)
    assert track_times(right, 30, {1: 10}) == (10, 11)
    cfg = Config(use_geometry=False, use_time=True, spatial_weight=0,
                 camera_offsets_s={1: 100}, max_gap_seconds=1,
                 transition_model=CameraTransitionModel())
    assert pair_cost(left, right, cfg) is not None


def test_forbidden_and_measured_topology():
    left, right = track(0, 0, 30), track(1, 60, 90)
    forbidden = CameraTransitionModel({(0, 1): {"status": "forbidden"}})
    assert pair_cost(left, right, Config(use_geometry=False, transition_model=forbidden)) is None
    measured = CameraTransitionModel({(0, 1): {"status": "measured", "min_s": 0.5, "max_s": 2.0}})
    assert pair_cost(left, right, Config(use_geometry=False, transition_model=measured)) is not None


def test_citic_world_position_protocol_is_separate_and_perfect(tmp_path):
    pred, gt = tmp_path / "pred.csv", tmp_path / "gt.csv"
    with pred.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp_s", "global_id", "x", "y"])
        writer.writeheader(); writer.writerow({"timestamp_s": 0, "global_id": 7, "x": 1, "y": 2})
    with gt.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp_s", "person_id", "x", "y"])
        writer.writeheader(); writer.writerow({"timestamp_s": 0, "person_id": 42, "x": 1, "y": 2})
    result = evaluate_world_positions(pred, gt)
    assert result["protocol_version"] == "citic-world-position-idf1-v1"
    assert result["official_bbox_idf1"] is False
    assert result["global"]["IDF1"] == 1.0
