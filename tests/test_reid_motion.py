"""CPU-only regression tests for the WILDTRACK ReID-aware tracker."""
import numpy as np
from mtmct.tracking_reid import ReIDLocalTracker, MatchConfig


def _det(x, emb, conf=0.8):
    return {"bbox": np.array([x, 0, x+30, 80],dtype=np.float32),
            "embedding": np.asarray(emb,dtype=np.float32),
            "confidence":conf,"quality":0.8}


def test_woman_preserves_id_across_three_images():
    t = ReIDLocalTracker(MatchConfig())
    ids=[]
    for i,x in enumerate([300,270,240]):
        d=t.step(65+5*i,13+i,[_det(x,[1,0,0])])
        ids.append(d[0]["local_id"])
    assert ids == [1,1,1]


def test_ancient_track_cannot_reuse_local_id():
    t=ReIDLocalTracker(MatchConfig(max_missed=2))
    first=t.step(0,0,[_det(100,[1,0,0])])[0]["local_id"]
    next_one=t.step(65,13,[_det(100,[1,0,0])])[0]["local_id"]
    assert first != next_one


def test_different_appearance_never_inherits_previous_id():
    t=ReIDLocalTracker(MatchConfig())
    id_a=t.step(0,0,[_det(100,[1,0,0])])[0]["local_id"]
    id_b=t.step(5,1,[_det(100,[0,1,0])])[0]["local_id"]
    assert id_a != id_b


def test_same_frame_requires_unique_ids():
    t=ReIDLocalTracker(MatchConfig())
    t.step(0,0,[_det(100,[1,0,0])])
    rows=t.step(5,1,[_det(110,[1,0,0]),_det(117,[1,0,0])])
    assert rows[0]["local_id"] != rows[1]["local_id"]


def test_missing_detection_one_frame_recovers_id():
    t=ReIDLocalTracker(MatchConfig())
    first=t.step(0,0,[_det(100,[1,0,0])])[0]["local_id"]
    t.step(5,1,[])
    later=t.step(10,2,[_det(120,[1,0,0])])[0]["local_id"]
    assert later==first


def test_low_conf_unmatched_is_in_raw_but_not_tracked():
    t=ReIDLocalTracker(MatchConfig(min_new_conf=0.12))
    d=t.step(0,0,[_det(100,[1,0,0],conf=0.10)])[0]
    assert d["local_id"] is None and d["decision"]=="low_new_conf"
