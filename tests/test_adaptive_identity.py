import copy
import json

import cv2
import numpy as np
import pytest

from mtmct.association import Config, hard_compatible, groups_compatible
from mtmct.background import ForegroundEvidence
from mtmct.features import reliable_gallery
from mtmct.frame_sources import FrameSource
from mtmct.identity_graph import IdentityGraph, associate_adaptive, save_identity_state, resume_identity_memory
from mtmct.memory import apply_memory
from mtmct.refinement import refine
from mtmct.time_sync import CameraTransitionModel, track_times


def row(cam, lid, start=0, end=10, vector=None):
    vector = vector or [1., 0.]
    return {'camera_id': cam, 'local_id': lid, 'start_frame': start, 'end_frame': end,
            'embedding': vector, 'positions': [], 'feature_vectors': [vector]*3,
            'feature_frames': [start, (start+end)//2, end], 'feature_quality': [.8]*3}


def config(**kwargs):
    return Config(use_geometry=False, spatial_weight=0, temporal_weight=0, **kwargs)


def test_unknown_absence_allowed_but_measured_impossibility_vetoed_everywhere():
    a, b, c = row(0, 1), row(1, 2, 1000, 1010), row(0, 3, 5, 20)
    cfg = config(max_gap_seconds=1)
    assert hard_compatible(a, b, cfg)
    assert not groups_compatible([a, b], [c], cfg)
    cfg.transition_model = CameraTransitionModel({(0, 1): {'status': 'forbidden'}})
    assert not hard_compatible(a, b, cfg)
    ids, _, journal = apply_memory([a, b], {(0, 1): 1, (1, 2): 2}, cfg=cfg, mode='constrained')
    assert ids[(0, 1)] != ids[(1, 2)] and journal[0]['reason'] == 'hard_constraint'


def test_offsets_align_geometry_and_explicit_seconds():
    a, b = row(0, 1, 0, 30), row(1, 2, 30, 60)
    a['positions'] = [[0, 0, 0], [30, 10, 0]]
    b['positions'] = [[30, 0, 0], [60, 10, 0]]
    cfg = Config(camera_offsets_s={1: -1}, overlap_distance_m=1)
    assert hard_compatible(a, b, cfg)
    b.update(start_s=1., end_s=2.)
    assert track_times(b, 30, {1: -1}) == (0, 1)


def test_low_quality_outlier_does_not_enter_gallery():
    t = row(0, 1)
    t['feature_vectors'].append([0., 1.])
    t['feature_quality'].append(.01)
    vectors, reliability = reliable_gallery(t)
    assert reliability == 1 and all(v[0] > .9 for v in vectors)


def test_ambiguous_competing_people_are_deferred():
    tracks = [row(0, 1), row(1, 2), row(1, 3)]
    ids, edges = associate_adaptive(tracks, config())
    assert len(set(ids.values())) == 3
    assert any(e['decision'] == 'deferred' and e['reason'] == 'ambiguous' for e in edges)


def test_revoke_rebuilds_components_and_does_not_leave_union_find_merge():
    tracks = [row(0, 1), row(1, 2), row(2, 3)]
    graph = IdentityGraph(tracks, config())
    graph.edges = [{'i': 0, 'j': 1, 'cost': .01, 'status': 'provisional'},
                   {'i': 1, 'j': 2, 'cost': .02, 'status': 'provisional'}]
    assert len(set(graph.resolve().values())) == 1
    graph.revoke(1, 2, 'new_conflicting_observation')
    ids = graph.resolve()
    assert ids[(0, 1)] == ids[(1, 2)] != ids[(2, 3)]


def test_sustained_split_updates_all_time_and_geometry_fields():
    t = row(0, 1, 0, 50)
    t.update(start_s=0., end_s=5., fps=10, samples=6, prototypes=[[1, 0]],
             feature_frames=[0, 10, 20, 30, 40, 50], feature_quality=[1]*6,
             feature_vectors=[[1, 0]]*3+[[0, 1]]*3,
             positions=[[f, f, 0] for f in range(51)])
    original = copy.deepcopy(t)
    parts, report, _ = refine([t], auto_split=True, sustained_samples=3)
    assert t == original and len(parts) == 2 and report['splits']
    for p in parts:
        assert p['start_s'] == p['start_frame']/10 and p['end_s'] == p['end_frame']/10
        assert all(p['start_frame'] <= f <= p['end_frame'] for f, *_ in p['positions'])
        assert p['samples'] == 3 and p['prototypes'] == []
    assert parts[0]['tracklet_uid'] != parts[1]['tracklet_uid']
    noisy = copy.deepcopy(t)
    noisy['feature_vectors'] = [[1, 0]]*2 + [[0, 1]] + [[1, 0]]*3
    assert not refine([noisy], auto_split=True, sustained_samples=3)[1]['splits']


def test_persistent_ids_survive_restart_merge_and_scope_guard(tmp_path):
    path = tmp_path/'memory.json'
    tracks = [row(0, 1), row(1, 2)]
    first = save_identity_state(path, tracks, {(0, 1): 1, (1, 2): 2}, scope='scene:model', now_s=0)
    second = save_identity_state(path, tracks, {(0, 1): 9, (1, 2): 9}, scope='scene:model', now_s=20)
    assert second['groups'][0]['global_id'] == first['groups'][0]['global_id']
    assert second['aliases'] == {'2': 1} and second['groups'][0]['status'] == 'missing'
    with pytest.raises(ValueError, match='scope'):
        save_identity_state(path, tracks, {}, scope='different_scene', now_s=30)
    assert json.loads(path.read_text()) == second


def test_stationary_foreground_is_valid_and_shadows_are_not_people():
    frame = np.full((360, 640, 3), 100, dtype=np.uint8)
    evidence = ForegroundEvidence('median_static', frame.copy())
    mask = evidence.update(frame)
    assert evidence.fraction(mask, [10, 10, 30, 50], frame.shape) == 0.
    assert evidence.fraction(None, [10, 10, 30, 50], frame.shape) is None


def test_persistent_memory_reactivates_new_local_id_and_rejects_conflict(tmp_path):
    path = tmp_path/'memory.json'
    a = row(0, 1)
    first, _ = resume_identity_memory(path, [a], {(0, 1): 100}, scope='test', now_s=1, cfg=config())
    b = row(1, 7, 1000, 1010)
    second, _ = resume_identity_memory(path, [b], {(1, 7): 1}, scope='test', now_s=34, cfg=config())
    assert first[(0, 1)] == second[(1, 7)]
    c = row(1, 8, 1005, 1015)
    third, _ = resume_identity_memory(path, [c], {(1, 8): 1}, scope='test', now_s=35, cfg=config())
    assert third[(1, 8)] != second[(1, 7)]


def test_frame_interval_keeps_original_numbers(tmp_path):
    for i in range(4):
        cv2.imwrite(str(tmp_path/f'{i}.png'), np.zeros((8, 8, 3), dtype=np.uint8))
    with FrameSource(tmp_path) as source:
        assert [f for f, _ in source.iter_frames(4, start_frame=2)] == [2, 3]


def test_bad_embedding_never_becomes_a_perfect_adaptive_match():
    bad = row(0, 1, vector=[float('nan'), 0])
    good = row(1, 2)
    ids, _ = associate_adaptive([bad, good], config())
    assert ids[(0, 1)] != ids[(1, 2)]
