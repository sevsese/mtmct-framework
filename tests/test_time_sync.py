import csv

from mtmct.association import Config, pair_cost
from mtmct.citic import evaluate_world_positions
from mtmct.time_sync import CameraTransitionModel, track_times


def track(camera, start, end):
    return {"camera_id": camera, "local_id": camera + 1, "start_frame": start,
            "end_frame": end, "embedding": [1, 0], "positions": []}


def test_offsets_and_unknown_topology_allow_a_match_inside_global_gap():
    left, right = track(0, 0, 30), track(1, 60, 90)
    assert track_times(right, 30, {1: 10}) == (12, 13)
    cfg = Config(use_geometry=False, use_time=True, spatial_weight=0,
                 camera_offsets_s={1: 0}, max_gap_seconds=1,
                 transition_model=CameraTransitionModel())
    assert pair_cost(left, right, cfg) is not None


def test_per_camera_fps_and_global_gap_limit_are_enforced_with_topology():
    left, right = track(0, 0, 25), track(1, 220, 240)
    assert track_times(left, {0: 25, 1: 20}) == (0, 1)
    assert track_times(right, {0: 25, 1: 20}) == (11, 12)
    cfg = Config(use_geometry=False, use_time=True, spatial_weight=0,
                 camera_fps={0: 25, 1: 20}, max_gap_seconds=9,
                 transition_model=CameraTransitionModel())
    assert pair_cost(left, right, cfg) is None


def test_explicit_track_seconds_receive_camera_offset_once():
    row = track(2, 0, 30)
    row.update(start_s=4.0, end_s=5.0)
    assert track_times(row, {2: 25}, {2: 3.0}) == (7.0, 8.0)


def test_only_declared_camera_pairs_may_overlap():
    left, right = track(0, 0, 25), track(1, 0, 25)
    blocked = Config(use_geometry=False, transition_model=CameraTransitionModel())
    allowed = Config(use_geometry=False,
                     transition_model=CameraTransitionModel(overlap_pairs={frozenset((0, 1))}))
    assert pair_cost(left, right, blocked) is None
    assert pair_cost(left, right, allowed) is not None


def test_interior_loss_can_reconnect_beyond_global_absence_limit():
    left, right = track(0, 0, 10), track(0, 150, 160)
    left["end_bbox_norm"] = [.4, .3, .5, .8]
    right["start_bbox_norm"] = [.42, .3, .52, .8]
    cfg = Config(use_geometry=False, fps=10, max_gap_seconds=10,
                 max_interior_gap_seconds=20, spatial_weight=0)
    assert pair_cost(left, right, cfg) is not None
    right["start_bbox_norm"] = [.9, .3, 1.0, 1.0]
    assert pair_cost(left, right, cfg) is None


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
