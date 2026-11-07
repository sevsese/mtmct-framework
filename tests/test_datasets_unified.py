"""Synthetic fixtures exercise adapters without downloading copyrighted dataset media."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from mtmct.datasets import open_dataset, CITICDataset, SCOUTDataset
from mtmct.frame_sources import FrameSource
from mtmct.dataset_evaluate import evaluate_bbox
from mtmct.visualize import render


def _img(path: Path, val: int = 127):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, np.full((48, 64, 3), val, dtype=np.uint8))
    assert ok
    encoded.tofile(path)


def _write_pred(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=['camera_id','frame_id','global_id','x1','y1','x2','y2'])
        w.writeheader(); w.writerows(rows)


def test_frame_source_image_keeps_real_frame_ids(tmp_path):
    _img(tmp_path/'image_0.jpg')
    _img(tmp_path/'image_2.jpg')
    with FrameSource(tmp_path, fps=10.0) as src:
        assert [fid for fid, _ in src.iter_frames(3)] == [0,2]
        assert src.get(1) is None
        assert src.fps == 10.0


def test_frame_source_selects_sparse_real_frame_ids(tmp_path):
    _img(tmp_path/'00000000.png')
    _img(tmp_path/'00000005.png')
    with FrameSource(tmp_path) as src:
        assert [fid for fid, _ in src.iter_frames(0, frame_ids={5})] == [5]


def test_video_buffer_reuse_preserves_frame_contents(tmp_path):
    path = tmp_path/"camera.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25., (64,48))
    for value in (30, 90, 150):
        writer.write(np.full((48,64,3), value, dtype=np.uint8))
    writer.release()
    with FrameSource(path) as reference, FrameSource(path, reuse_buffer=True) as source:
        frames = source.iter_frames(3, stride=2)
        index, first = next(frames)
        assert index == 0 and np.array_equal(first, reference.get(0))
        index, last = next(frames)
        assert index == 2 and np.array_equal(last, reference.get(2))
        assert np.shares_memory(first, last)


def test_render_sparse_image_frame_ids(tmp_path):
    import csv
    for cam in (0, 1):
        for frame in (0, 5):
            _img(tmp_path/f'cam_{cam}'/f'{frame:08d}.png')
    predictions = tmp_path/'tracks.csv'
    with predictions.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['camera_id','frame_id','global_id','x1','y1','x2','y2'])
        writer.writeheader()
        for cam in (0, 1):
            for frame in (0, 5):
                writer.writerow({'camera_id':cam, 'frame_id':frame, 'global_id':1,
                                 'x1':10, 'y1':10, 'x2':30, 'y2':40})
    video = tmp_path/'mosaic.mp4'
    render(tmp_path, predictions, video, max_frames=0, stride=1, scale=.25,
           cameras=[0,1], camera_sources={0:tmp_path/'cam_0', 1:tmp_path/'cam_1'},
           fps_fallback=2, frame_ids={0,5})
    assert video.is_file() and video.stat().st_size > 0


def test_duration_render_keeps_predictions_after_default_frame_limit(tmp_path):
    import csv
    for cam in (0, 1):
        _img(tmp_path/f'late_{cam}'/'00000314.png')
    predictions = tmp_path/'late.csv'
    with predictions.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['camera_id','frame_id','global_id','x1','y1','x2','y2'])
        writer.writeheader()
        for cam in (0, 1):
            writer.writerow({'camera_id':cam, 'frame_id':314, 'global_id':7,
                             'x1':8, 'y1':8, 'x2':40, 'y2':40})
    video = tmp_path/'late.mp4'
    render(tmp_path, predictions, video, stride=2, scale=1., cameras=[0,1],
           camera_sources={0:tmp_path/'late_0', 1:tmp_path/'late_1'},
           fps_fallback=2, duration_seconds=160)
    capture = cv2.VideoCapture(str(video))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
    ok, frame = capture.read()
    capture.release()
    assert ok and int(frame.max()) - int(frame.min()) > 30


def test_citic_world_only_no_fake_bboxes(tmp_path):
    capture = tmp_path/'capture1'
    capture.mkdir()
    (capture/'HallWide.mp4').touch()
    (capture/'PF.mp4').touch()
    (tmp_path/'capture1_gt.mot.txt').write_text('59,1,-1,-1,-1,-1,-1,25.2,12.6,0\n')
    ds = CITICDataset(tmp_path)
    assert ds.list_scenes() == ['capture1']
    assert ds.list_cameras('capture1') == [0,2]
    obs = list(ds.observations('capture1'))
    assert obs[0].bbox_xyxy is None
    assert obs[0].camera_id is None
    assert obs[0].world_xy == (25.2,12.6)
    assert not ds.supports_bbox_evaluation('capture1',[0,2])
    with pytest.raises(ValueError,match='trusted camera bbox'):
        evaluate_bbox(tmp_path/'nothing.csv',ds,'capture1',[0,2],100)


def test_scout_observations_and_idf1(tmp_path):
    for cam in (0,1):
        _img(tmp_path/'images'/'sequence_1'/f'cam_{cam}'/'image_0.jpg')
        mot = tmp_path/'annotations'/'mot_format'/'sequence_1'/f'cam_{cam}.txt'
        mot.parent.mkdir(parents=True,exist_ok=True)
        mot.write_text('0,42,10,10,20,20,1,1,2,3\n')
    ds = SCOUTDataset(tmp_path)
    assert ds.list_scenes()==['sequence_1']
    assert ds.list_cameras('sequence_1')==[0,1]
    assert ds.get_video('sequence_1',0).is_dir()
    assert ds.supports_bbox_evaluation('sequence_1',[0,1])
    obs = list(ds.observations('sequence_1',10,[0,1]))
    assert {o.identity for o in obs} == {42}
    assert {o.bbox_xyxy for o in obs} == {(10,10,30,30)}
    pred=tmp_path/'prediction.csv'
    _write_pred(pred,[{'camera_id':cam,'frame_id':0,'global_id':7,'x1':10,'y1':10,'x2':30,'y2':30}
                      for cam in (0,1)])
    metrics=evaluate_bbox(pred,ds,'sequence_1',[0,1],10)
    assert metrics['global']['IDF1'] == 1
    assert metrics['pairwise_association']['F1'] == 1
    assert metrics['detection']['recall']==1
    out=tmp_path/'mosaic.mp4'
    render(tmp_path,pred,out,max_frames=1,stride=1,scale=.5,cameras=[0,1],
           camera_sources=ds.frame_sources('sequence_1',[0,1]),fps_fallback=ds.fps)
    assert out.exists() and out.stat().st_size>0


def test_scout_disallows_pseudo_labels(tmp_path):
    for cam in (0,8):
        _img(tmp_path/'images'/'sequence_1'/f'cam_{cam}'/'image_0.jpg')
        mot = tmp_path/'annotations'/'mot_format'/'sequence_1'/f'cam_{cam}.txt'
        mot.parent.mkdir(parents=True,exist_ok=True)
        mot.write_text('0,42,10,10,20,20,1,1,2,3\n')
    ds=SCOUTDataset(tmp_path)
    assert not ds.supports_bbox_evaluation('sequence_1',[0,8])
    assert SCOUTDataset(tmp_path,allow_pseudo_labels=True).supports_bbox_evaluation('sequence_1',[0,8])


def test_nvidia_optional_calibration(tmp_path):
    scene=tmp_path/'train'/'Warehouse_000'
    (scene/'videos').mkdir(parents=True)
    (scene/'videos'/'Camera_0003.mp4').touch()
    ds = open_dataset('nvidia',tmp_path)
    assert ds.list_cameras('Warehouse_000')==[3]
    assert ds.get_calibration('Warehouse_000') is None
    assert not ds.supports_bbox_evaluation('Warehouse_000',[3])


def test_tracking_on_scout_frames_without_video_or_calibration(tmp_path, monkeypatch):
    import sys
    import types
    from mtmct.tracking import run
    from mtmct.link import run as relink
    from mtmct.association import Config

    class Tensor:
        def __init__(self, data): self.arr = np.asarray(data)
        def int(self): return self
        def cpu(self): return self
        def numpy(self): return self.arr

    class FakeYOLO:
        def __init__(self, name): self.name=name
        def track(self, frame, **kwargs):
            boxes = types.SimpleNamespace(
                id=Tensor([1]), xyxy=Tensor([[10., 10., 30., 40.]]),
                conf=Tensor([.9]))
            return [types.SimpleNamespace(boxes=boxes)]

    monkeypatch.setitem(sys.modules, 'ultralytics', types.SimpleNamespace(YOLO=FakeYOLO))
    for cam in (0, 1):
        _img(tmp_path/'images'/'sequence_1'/f'cam_{cam}'/'image_0.jpg')
        _img(tmp_path/'images'/'sequence_1'/f'cam_{cam}'/'image_1.jpg')
    ds=SCOUTDataset(tmp_path)
    out=tmp_path/'run'
    run(tmp_path, out, 'mock-yolo', 'color', None, 'cpu', 2, 1, 1,
        640, .1, cameras=[0,1], camera_sources=ds.frame_sources('sequence_1',[0,1]),
        fps_fallback=10)
    tracklets=json.loads((out/'local_tracklets.json').read_text())
    assert len(tracklets)==2
    assert {t['camera_id'] for t in tracklets} == {0,1}
    assert not tracklets[0]['positions']
    assert tracklets[0]['source_fps'] == 10
    assert tracklets[0]['start_s'] == 0 and tracklets[0]['end_s'] == .1
    assert (out/'tracking_diagnostics.csv').is_file()
    manifest=json.loads((out/'manifest.json').read_text())
    assert any(x.get('type')=='directory-inventory' for x in manifest['inputs'])
    _,mapping,_,preds,_=relink(out, Config(use_geometry=False,appearance_weight=1.,
                                           spatial_weight=0., temporal_weight=0.,
                                           use_time=False,min_similarity=.8,max_cost=.125))
    assert Path(preds).is_file()
    assert len(mapping)==2
