from mtmct.association import Config, associate_graph
from mtmct.memory import apply_memory


def track(camera, local_id, start, end, embedding):
    return {"camera_id": camera, "local_id": local_id, "start_frame": start,
            "end_frame": end, "embedding": embedding, "positions": []}


def test_memory_recovers_same_identity_after_long_gap():
    rows = [track(0, 1, 0, 10, [1, 0]), track(1, 2, 1000, 1010, [1, 0])]
    cfg = Config(use_geometry=False, use_time=True, spatial_weight=0,
                 max_gap_seconds=1, fps=30, min_similarity=0.8)
    baseline, _journal = associate_graph(rows, cfg)
    assert baseline[(0, 1)] != baseline[(1, 2)]
    recovered, records, decisions = apply_memory(
        rows, baseline, similarity_threshold=0.9, config=cfg)
    assert recovered[(0, 1)] != recovered[(1, 2)]
    assert len(records[0]["prototypes"]) >= 1
    assert decisions[0]["reason"] == "physical_conflict"


def test_memory_off_is_exact_graph_mapping_and_conflicts_stay_separate():
    rows = [track(0, 1, 0, 20, [1, 0]), track(0, 2, 10, 30, [1, 0])]
    mapping, _journal = associate_graph(rows, Config(use_geometry=False, use_time=False))
    assert mapping[(0, 1)] != mapping[(0, 2)]
    recovered, _records, decisions = apply_memory(rows, mapping, similarity_threshold=0.9)
    assert recovered[(0, 1)] != recovered[(0, 2)]
    assert decisions[0]["reason"] == "physical_conflict"


def test_memory_cannot_override_forbidden_cross_camera_overlap():
    from mtmct.time_sync import CameraTransitionModel

    rows = [track(0, 1, 0, 20, [1, 0]), track(4, 2, 0, 20, [1, 0])]
    mapping = {(0, 1): 1, (4, 2): 2}
    cfg = Config(use_geometry=False, use_time=True,
                 transition_model=CameraTransitionModel())
    recovered, _records, decisions = apply_memory(
        rows, mapping, similarity_threshold=.9, config=cfg)
    assert recovered[(0, 1)] != recovered[(4, 2)]
    assert decisions[0]["reason"] == "physical_conflict"
