"""Opt-in real GPU integration check; uses short excerpts, not mock tracking."""
import json
from pathlib import Path
import time
import cv2
import httpx
from backend.adapter import ROOT


def main():
    out=ROOT/'runs/web_pipeline_smoke'
    out.mkdir(exist_ok=True)
    scene=ROOT/'data/MTMC_Tracking_2026/train/Warehouse_000/videos'
    for cam in (3,15):
        reader=cv2.VideoCapture(str(scene/f'Camera_{cam:04d}.mp4'))
        writer=cv2.VideoWriter(str(out/f'Camera_{cam:04d}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1920,1080))
        try:
            for _ in range(12):
                ok,frame=reader.read()
                if not ok: raise RuntimeError('Cannot decode real NVIDIA excerpt')
                writer.write(frame)
        finally:
            reader.release();writer.release()
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=60) as client:
        handles=[(out/f'Camera_{c:04d}.mp4').open('rb') for c in (3,15)]
        try:
            result=client.post('/api/datasets/upload',data={'name':'NVIDIA — проверка API, 12 кадров','mode':'raw',
                'cameras':json.dumps([{'camera_id':c,'filename':f'Camera_{c:04d}.mp4','name':f'Camera {c}'} for c in (3,15)])},
                files=[('files',(f'Camera_{c:04d}.mp4',stream,'video/mp4')) for c,stream in zip((3,15),handles)])
        finally:
            for stream in handles: stream.close()
        result.raise_for_status();identifier=result.json()['id']
        job=client.post(f'/api/datasets/{identifier}/process',json={'max_frames':12,'device':'cuda:0'})
        job.raise_for_status()
        print('JOB',identifier,flush=True)
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            job=client.get(f'/api/jobs/{identifier}').json()
            if job['status'] in {'completed','failed'}: break
            time.sleep(2)
        (out/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
        assert job['status']=='completed',job
        result=client.get(f'/api/datasets/{identifier}').json()
        performance=client.get(f'/api/datasets/{identifier}/performance').json()
        assert [p['processed_frames'] for p in performance['cameras']]==[12,12]
        assert result['summary']['observations']>0 and result['summary']['global_tracks']>0
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
