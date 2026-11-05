import copy

from mtmct.association import Config
from mtmct.correction import rebuild


def track(camera, local_id, embedding):
    return {"camera_id": camera, "local_id": local_id, "start_frame": 0,
            "end_frame": 10, "embedding": embedding, "positions": []}


def test_false_transitive_merge_is_rebuilt_without_mutating_tracklets():
    rows = [track(0, 1, [1, 0]), track(1, 2, [0.9, 0.435]), track(2, 3, [0, 1])]
    original = copy.deepcopy(rows)
    journal = [
        {"i": 0, "j": 1, "cost": 0.05, "decision": "accepted", "status": "provisional"},
        {"i": 1, "j": 2, "cost": 0.10, "decision": "accepted", "status": "provisional"},
    ]
    cfg = Config(use_geometry=False, use_time=False, spatial_weight=0, temporal_weight=0,
                 min_similarity=0.8, max_cost=0.2)
    mapping, corrected, conflicts = rebuild(rows, journal, cfg)
    assert mapping[(0, 1)] == mapping[(1, 2)]
    assert mapping[(0, 1)] != mapping[(2, 3)]
    assert corrected[0]["status"] == "confirmed"
    assert corrected[1]["status"] == "rejected" and conflicts
    assert rows == original


def test_correction_assignment_is_deterministic():
    rows = [track(0, 1, [1, 0]), track(1, 2, [1, 0])]
    journal = [{"i": 0, "j": 1, "cost": 0, "decision": "accepted", "status": "provisional"}]
    cfg = Config(use_geometry=False, use_time=False, spatial_weight=0, temporal_weight=0)
    assert rebuild(rows, journal, cfg)[0] == rebuild(rows, journal, cfg)[0]
