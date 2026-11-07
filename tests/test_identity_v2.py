from dataclasses import replace
import math
import sqlite3

import pytest

from mtmct.association import Config
from mtmct.identity_store import prior_revocations, register
from mtmct.identity_v2 import OFFICE_SETTINGS, associate_v2, observed_lifetime
from mtmct.refinement import split_sustained
from mtmct.scoring import Evidence, physical, score
from mtmct.time_sync import CameraTransitionModel


def row(camera, local, start=0, embedding=(1, 0), q=.8, fps=25):
    return {"camera_id": camera, "local_id": local, "start_frame": start,
            "end_frame": start+9, "source_fps": fps, "observation_frames": list(range(start,start+10)),
            "embedding": list(embedding), "feature_frames": [start,start+3,start+6],
            "feature_quality": [q]*3, "feature_vectors": [list(embedding)]*3}


def test_missing_geometry_and_time_do_not_become_confidence():
    cfg = Config(use_geometry=False, transition_model=CameraTransitionModel())
    a, b = Evidence(row(1, 1), cfg), Evidence(row(2, 1, 20, q=.25), cfg)
    result = score(a,b,cfg,physical(a,b,cfg))
    assert result["probability"] is None
    assert result["weights"] == {"appearance":1., "spatial":0., "temporal":0.}
    assert result["reliability"] == pytest.approx(.75 * math.sqrt(.8*.25))
    b = Evidence(row(2, 1, 20, embedding=()), cfg)
    assert len(associate_v2([a.row, b.row], cfg)[1]) == 2


def test_mixed_fps_offsets_use_timestamp_and_geometry_can_be_disabled():
    first, second = row(1,1,25), row(2,1,0,fps=20)
    first["positions"] = [[25,0,0],[27,0,0]]
    second["positions"] = [[0,20,0],[2,20,0]]
    second["frame_timestamps"] = {"0":0., "2":.08, "9":.45}
    cfg = Config(camera_offsets_s={2:.75}, camera_start_s={2:.25})
    conflict = physical(Evidence(first,cfg), Evidence(second,cfg), cfg)
    assert conflict["conflict"] == "simultaneous_floor_positions"
    assert conflict["sync_samples"] == 2
    assert len(associate_v2([first,second], replace(cfg,use_geometry=False))[1]) == 1


def test_unknown_long_gap_is_soft_but_forbidden_and_concurrent_same_camera_are_hard():
    a, b = row(1,1), row(2,1,2500)
    cfg = Config(use_geometry=False, max_gap_seconds=10)
    assert len(associate_v2([a,b], cfg)[1]) == 1
    forbidden = CameraTransitionModel({(1,2): {"status":"forbidden"}})
    assert len(associate_v2([a,b], replace(cfg, transition_model=forbidden))[1]) == 2
    assert len(associate_v2([a,row(1,2)], cfg)[1]) == 2


def test_revising_one_edge_splits_and_keeps_one_stable_id(tmp_path):
    rows = [row(1,1),row(2,1),row(3,1,embedding=(0,1))]
    cfg = Config(use_geometry=False)
    path = tmp_path/"identity.sqlite"
    tracks, groups, edges = associate_v2(rows, cfg)
    initial, state = register(path, "office", tracks, groups, edges)
    assert initial[1,1] == initial[2,1]
    assert register(path,"office",tracks,groups,edges)[1]["revision"] == state["revision"]
    revoked = [(tracks[0].uid,tracks[1].uid)]
    tracks, groups, edges = associate_v2(rows,cfg,revoked=revoked)
    updated, state = register(path,"office",tracks,groups,edges)
    assert updated[1,1] != updated[2,1]
    assert initial[1,1] in {updated[1,1],updated[2,1]}
    assert initial[3,1] == updated[3,1]
    assert state["events"] == [{"type":"split","from":initial[1,1],"to":sorted({updated[1,1],updated[2,1]})}]
    assert prior_revocations(path) == revoked


def test_provisional_attachment_does_not_enter_persistent_gallery(tmp_path):
    cfg = Config(use_geometry=False)
    tracks, groups, edges = associate_v2([row(1,1),row(2,1),row(3,1,q=.2)], cfg)
    assert len(groups) == 1
    assert any(e["state"] == "provisional" and e["decision"] == "accepted" for e in edges)
    path = tmp_path/"identity.sqlite"
    register(path, "office", tracks, groups, edges)
    with sqlite3.connect(path) as db:
        import json
        assert {json.loads(sample)["camera_id"] for sample, in db.execute("select sample from prototypes")} == {1,2}


def test_local_split_requires_sustained_appearance_and_motion_jump():
    track = row(1,1)
    track["temporal_frames"] = list(range(6))
    track["temporal_vectors"] = [[1,0]]*3 + [[0,1]]*3
    track["observation_boxes"] = {str(f):[0,0,10,20] for f in range(10)}
    cfg = Config(use_geometry=False)
    assert len(split_sustained([track],cfg)[0]) == 1
    track["observation_boxes"]["3"] = [100,0,110,20]
    parts, remap, _ = split_sustained([track],cfg)
    assert [(t["start_frame"],t["end_frame"]) for t in parts] == [(0,2),(3,9)]
    assert [r["new_local_id"] for r in remap] == [1,2]


def test_office_clothing_cannot_override_weak_reid_between_cameras():
    cfg = Config(use_geometry=False, min_similarity=.45, gallery_weight=.25, part_weight=.3)
    a = row(1, 1)
    b = row(2, 1, embedding=(.703, math.sqrt(1 - .703**2)))
    for t in (a, b):
        t["body_parts"] = [{"upper":[1,0],"lower":[1,0]}]
    _, groups, edges = associate_v2([a, b], cfg, settings=OFFICE_SETTINGS)
    assert len(groups) == 2
    assert edges[0]["reason"] == "weak_cross_camera_reid"


def test_office_handoff_repairs_partial_bbox_without_joining_two_people():
    cfg = Config(use_geometry=False)
    a, b = row(1, 1), row(1, 2, 8)
    a["observation_boxes"] = {str(f): [20,20,40,55] for f in a["observation_frames"]}
    b["observation_boxes"] = {str(f): [10,20,50,100] for f in b["observation_frames"]}
    tracks, groups, edges = associate_v2([a,b], cfg, settings=OFFICE_SETTINGS)
    assert groups == [[0,1]]
    assert edges[0]["reason"] == "overlapping_handoff"
    assert observed_lifetime(tracks) == pytest.approx(18 / 25)
    b["observation_boxes"]["9"] = [30,20,70,100]
    _, groups, edges = associate_v2([a,b], cfg, settings=OFFICE_SETTINGS)
    assert len(groups) == 2
    assert edges[0]["reason"] == "same_camera_same_observation_time"
