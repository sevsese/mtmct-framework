import csv
import pytest
from mtmct.episodes import export_episodes
from mtmct.correlation import solve
from mtmct.probability import fit, predict


def test_episodes_apply_offsets_and_sixty_second_boundary(tmp_path):
    p=tmp_path/'observations.csv'
    p.write_text('camera_id,frame_id,global_id\n0,0,7\n1,0,7\n0,1525,7\n0,3026,7\n')
    episodes=export_episodes(p,{0:25,1:20},{1:1},60)
    assert len(episodes)==2
    assert episodes[0]['end_time']==61
    assert episodes[1]['start_time']==pytest.approx(121.04)


def test_correlation_respects_conflict_and_transitivity():
    groups,report=solve(3,{(0,1):.99,(1,2):.8,(0,2):.01},[(0,2)])
    assert sorted(map(sorted,groups))==[[0,1],[2]]
    assert report['optimal']


def test_probability_validation_leakage_and_monotonicity(tmp_path):
    train=[dict(cost=c,label=y,recording='train') for c,y in [(0,1),(.1,1),(.8,0),(1,0)]]
    validation=[dict(r,recording='heldout') for r in train]
    m=fit(train,validation,tmp_path/'calibration.json')
    assert predict(.1,m)>predict(.8,m)
    with pytest.raises(ValueError):
        fit(train,train,tmp_path/'bad.json')


def test_persistent_bank_reuses_identity_without_double_counting(tmp_path):
    import sqlite3
    from types import SimpleNamespace
    from mtmct.identity_bank import update_bank
    track=SimpleNamespace(key=(0,1),anchor=True,prototypes=[dict(e=[1.,0.],q=.8,source='crop')])
    edges=[dict(state='confirmed',i=0,j=0)]
    path=tmp_path/'bank.sqlite'
    first=update_bank(path,'model','recording1',[track],[[0]],edges,{(0,1):7})
    assert update_bank(path,'model','recording1',[track],[[0]],edges,{(0,1):7})==first
    second=update_bank(path,'model','recording2',[track],[[0]],edges,{(0,1):9})
    assert second[9]==first[7]
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT support FROM prototypes').fetchone()[0]==pytest.approx(1.6)
