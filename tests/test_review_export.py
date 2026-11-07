import json
import sqlite3
import cv2
import numpy as np
import pytest
from backend import export_review


def test_review_export_uses_time_for_different_fps(tmp_path,monkeypatch):
    pytest.importorskip('imageio_ffmpeg')
    cameras=[]
    for cid,fps in [(1,10),(2,20)]:
        path=tmp_path/f'{cid}.mp4'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(64,48))
        for _ in range(fps):writer.write(np.zeros((48,64,3),dtype=np.uint8))
        writer.release()
        cameras.append({'camera_id':cid,'path':str(path),'fps':fps,'width':64,'height':48,'duration':1,'offset':0})
    (tmp_path/'dataset.json').write_text(json.dumps({'cameras':cameras}))
    with sqlite3.connect(tmp_path/'index.sqlite') as db:
        db.execute('CREATE TABLE observations(camera_id,frame_id,local_id,global_id,timestamp,x1,y1,x2,y2)')
    calls={1:[],2:[]}
    class Reader(export_review.FrameSource):
        def get(self,fid):
            calls[int(self.path.stem)].append(fid)
            return super().get(fid)
    monkeypatch.setattr(export_review,'FrameSource',Reader)
    export_review.render(tmp_path,tmp_path/'review.mp4',seconds=1,fps=10)
    assert calls[1]==list(range(10))
    assert calls[2]==list(range(0,20,2))
    cap=cv2.VideoCapture(str(tmp_path/'review.mp4'))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==10
    cap.release()
