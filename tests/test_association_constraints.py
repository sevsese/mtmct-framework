from mtmct.association import Config, associate_graph


def track(camera, local_id, embedding, start=0, end=10):
    return {"camera_id": camera, "local_id": local_id, "start_frame": start,
            "end_frame": end, "embedding": embedding, "positions": []}


def test_graph_allows_cross_camera_overlap_but_forbids_same_camera_conflict():
    rows = [track(0, 1, [1, 0]), track(1, 2, [1, 0]), track(0, 3, [1, 0])]
    mapping, journal = associate_graph(
        rows, Config(use_geometry=False, use_time=False, spatial_weight=0, temporal_weight=0,
                     min_similarity=0.8, max_cost=0.2))
    assert mapping[(0, 1)] == mapping[(1, 2)]
    assert mapping[(0, 1)] != mapping[(0, 3)]
    assert any(edge["reason"] == "component_conflict" for edge in journal)


def test_graph_keeps_missing_reid_as_singleton_and_logs_reason():
    rows = [track(0, 1, []), track(1, 2, [1, 0])]
    mapping, journal = associate_graph(
        rows, Config(use_geometry=False, use_time=False, spatial_weight=0, temporal_weight=0))
    assert len(set(mapping.values())) == 2
    assert journal[0]["reason"] == "missing_reid" and journal[0]["decision"] == "rejected"


def test_graph_builds_a_long_track_from_short_valid_gaps():
    rows = [track(0, 1, [1, 0], 0, 10),
            track(0, 2, [1, 0], 20, 30),
            track(0, 3, [1, 0], 40, 50)]
    mapping, _ = associate_graph(
        rows, Config(use_geometry=False, use_time=True, fps=10,
                     max_gap_seconds=1, spatial_weight=0, temporal_weight=0,
                     min_similarity=.8, max_cost=.2),
        allow_same_camera_fragments=True)
    assert len(set(mapping.values())) == 1


def test_overlapping_tracker_intervals_may_merge_when_detections_do_not_coexist():
    left, right = track(0, 1, [1, 0], 0, 20), track(0, 2, [1, 0], 10, 30)
    left["observation_frames"], right["observation_frames"] = [0, 10], [20, 30]
    mapping, _ = associate_graph(
        [left, right], Config(use_geometry=False, use_time=True, fps=10,
                              spatial_weight=0, temporal_weight=0),
        allow_same_camera_fragments=True)
    assert len(set(mapping.values())) == 1

    right["observation_frames"] = [10, 20]
    mapping, _ = associate_graph(
        [left, right], Config(use_geometry=False, use_time=True, fps=10,
                              spatial_weight=0, temporal_weight=0),
        allow_same_camera_fragments=True)
    assert len(set(mapping.values())) == 2


def test_duplicate_boxes_in_one_frame_do_not_create_a_false_conflict():
    left, right = track(0, 1, [1, 0], 0, 10), track(0, 2, [1, 0], 10, 20)
    left["observation_frames"], right["observation_frames"] = [10], [10]
    left["observation_boxes"] = {"10": [10, 10, 30, 50]}
    right["observation_boxes"] = {"10": [11, 10, 31, 50]}
    mapping, _ = associate_graph(
        [left, right], Config(use_geometry=False, use_time=True, fps=10,
                              spatial_weight=0, temporal_weight=0),
        allow_same_camera_fragments=True)
    assert len(set(mapping.values())) == 1
