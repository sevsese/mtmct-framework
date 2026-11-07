from mtmct.association import Config, appearance_similarity
from mtmct.track_report import build_global_track_report


def row(camera, local_id, embedding, start=0, end=10, features=None):
    return {"camera_id": camera, "local_id": local_id, "start_frame": start,
            "end_frame": end, "embedding": embedding, "positions": [],
            "feature_vectors": features or []}


def test_quality_gallery_rescues_a_matching_view_without_discarding_track_mean():
    left = row(0, 1, [1, 0], features=[[1, 0], [0, 1]])
    right = row(1, 2, [0, 1], features=[[1, 0]])
    assert appearance_similarity(left, right, Config(gallery_weight=0)) == 0
    assert appearance_similarity(left, right, Config(gallery_weight=.75)) == .75


def test_visible_body_parts_can_support_a_difficult_reid_pair():
    left = row(0, 1, [1, 0])
    right = row(1, 2, [0, 1])
    left["body_parts"] = [{"upper": [1, 0], "lower": [0, 1]}]
    right["body_parts"] = [{"upper": [1, 0], "lower": [0, 1]}]
    assert appearance_similarity(left, right, Config(part_weight=.4)) == .4


def test_global_report_orders_and_deduplicates_camera_sequence():
    tracks = [row(0, 1, [1, 0], 0, 25), row(0, 2, [1, 0], 30, 50),
              row(1, 3, [1, 0], 60, 80)]
    mapping = {(0, 1): 7, (0, 2): 7, (1, 3): 7}
    report = build_global_track_report(tracks, mapping, {0: 25, 1: 20}, {1: 1})
    assert report[0]["camera_sequence"] == [0, 1]
    assert report[0]["start_s"] == 0
    assert report[0]["end_s"] == 5
