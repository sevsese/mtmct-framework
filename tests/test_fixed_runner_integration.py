"""Mocked neural-network smoke test of the complete fixed runner."""
import csv
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np


def test_runner_saves_predictions_metrics_and_video(tmp_path, monkeypatch):
    from mtmct import tracking_reid
    import run_wildtrack_fixed

    root = tmp_path/'WILDTRACK'
    ann = root/'annotations_positions'
    ann.mkdir(parents=True)
    for cam in (0,1):
        (root/'Image_subsets'/f'C{cam+1}').mkdir(parents=True)
    for step, fid in enumerate([0,5,10]):
        positions=[]
        for cam in (0,1):
            x=20 + 3*step + 7*cam
            view={"viewNum":cam,"xmin":x,"ymin":10,"xmax":x+22,"ymax":65}
            positions.append(view)
            img = np.zeros((80,100,3),np.uint8)
            cv2.rectangle(img,(x,10),(x+22,65),(0,255,0),-1)
            cv2.imencode('.png', img)[1].tofile(
                root/'Image_subsets'/f'C{cam+1}'/f'{fid:08d}.png')
        (ann/f'{fid:08d}.json').write_text(json.dumps([{"personID":3,"views":positions}]))

    class FakeT:
        def __init__(self,a): self.a=np.asarray(a,dtype=np.float32)
        def cpu(self): return self
        def numpy(self): return self.a

    class FakeYOLO:
        def __init__(self,model): pass
        def predict(self, source, **kwargs):
            green=(source[:,:,1]>150).astype(np.uint8)
            ys,xs=np.where(green)
            bb=[float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)]
            return [types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=FakeT([bb]),conf=FakeT([0.9])))]

    class FakeReID:
        def __init__(self,*a,**kwargs): pass
        def encode(self,crops): return [np.asarray([1.,0.,0.],dtype=np.float32) for c in crops]

    monkeypatch.setitem(sys.modules,'ultralytics',types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(tracking_reid,'ReID',FakeReID)
    weights=tmp_path/'fake_weights.pt'
    weights.write_bytes(b'fakeweights')
    out=tmp_path/'result'
    result=run_wildtrack_fixed.main(["--root",str(root),"--out",str(out),
        "--cameras","0","1","--max-frames","3", "--device","cpu",
        "--reid-weights",str(weights), "--model","not_a_file.pt"])
    assert result['new']['detection_recall'] == 100.0
    assert result['new']['global_idf1'] == 100.0
    assert result['new']['detection_precision'] == 100.0
    assert result['local']['raw_yolo_boxes']==6
    for name in ['observations.csv','raw_detections.csv','local_tracklets.json',
                 'global_tracks_2d.csv','metrics.json','raw_detection_metrics.json',
                 'comparison.json','mosaic.mp4','observation_embeddings.npz']:
        assert (out/name).exists(), name
    assert (out/'mosaic.mp4').stat().st_size>1000
    with (out/'global_tracks_2d.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==6
    assert len({int(r['global_id']) for r in rows})==1
