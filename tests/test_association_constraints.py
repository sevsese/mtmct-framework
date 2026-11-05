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
