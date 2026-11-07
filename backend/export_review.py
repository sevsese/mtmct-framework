"""Export time-aligned camera overlays and the approximate plan as H.264.

Usage: python -m backend.export_review DATASET_DIRECTORY output.mp4 --seconds 30
Requires optional imageio-ffmpeg. Reads existing observations; runs no detector.
"""
import argparse
import colorsys
from collections import defaultdict
from bisect import bisect_left, bisect_right
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import cv2
import numpy as np
from backend.adapter import read_json,rows
from backend.floor_map import project
from mtmct.frame_sources import FrameSource


def shade(gid):
    if gid is None:return (197,180,170)
    r,g,b=colorsys.hls_to_rgb((gid*137.508%360)/360,.67,.78)
    return tuple(round(v*255) for v in (b,g,r))


def review_interval(data):
    """Limit playback to processed frames when a run records its limits."""
    config = data.get('config') or {}
    tracking = (data.get('manifest') or {}).get('tracking_config') or {}
    limits = tracking.get('per_camera_max_frames', {})
    bounds = {}
    for camera in data['cameras']:
        if not camera.get('path') or not camera.get('fps') or not camera.get('duration'):
            continue
        duration = camera['duration']
        frame_limit = limits.get(str(camera['camera_id']), tracking.get('max_frames'))
        if frame_limit:
            duration = min(duration, frame_limit / camera['fps'])
        seconds = tracking.get('max_seconds') or config.get('duration_seconds') or config.get('max_seconds')
        if seconds and str(camera['camera_id']) not in limits:
            duration = min(duration, seconds)
        bounds[camera['camera_id']] = duration
    if not bounds:
        raise ValueError('Для мозаики требуется хотя бы одно подключённое видео')
    cameras = [c for c in data['cameras'] if c['camera_id'] in bounds]
    start = min(0., *(c.get('offset', 0.) for c in cameras))
    end = max(c.get('offset', 0.) + bounds[c['camera_id']] for c in cameras)
    return start, end, bounds


def review_window(dataset_dir, data, global_id):
    start, end, bounds = review_interval(data)
    if global_id is not None:
        track = rows(dataset_dir/'index.sqlite',
                     'SELECT MIN(timestamp) start,MAX(timestamp) end FROM observations WHERE global_id=?',
                     [global_id])[0]
        if track['start'] is None:
            raise ValueError('Для этого Global ID нет наблюдений с известным временем')
        start = max(start, track['start'] - .5)
        end = min(end, track['end'] + .5)
        if start >= end:
            raise ValueError('Трек вне диапазона подключённых видео')
    return start, end, bounds


def frame_boxes(records, cameras):
    indexed = defaultdict(list)
    previous = {}
    for row in records:
        cid, fid = row['camera_id'], row['frame_id']
        if cid not in cameras:
            continue
        indexed[cid,fid].append(row)
        key = cid,row['local_id'],row['global_id']
        left = previous.get(key)
        previous[key] = row
        if left is None:
            continue
        gap = fid - left['frame_id']
        if 1 < gap <= max(2, math.floor(cameras[cid]['fps'] * .12)):
            for step in range(1, gap):
                box = {k: left[k] + step / gap * (row[k] - left[k]) for k in ('x1','y1','x2','y2')}
                indexed[cid,left['frame_id'] + step].append(row | box)
    return indexed


def render(dataset_dir,output,seconds=30,fps=10,*,global_id=None,include_plan=True):
    import imageio_ffmpeg
    d=read_json(dataset_dir,'dataset.json')
    start,end,bounds=review_window(dataset_dir,d,global_id)
    seconds=min(seconds,end-start) if seconds is not None else end-start
    cams={c['camera_id']:c for c in sorted(d['cameras'],key=lambda c:c['camera_id']) if c['camera_id'] in bounds}
    records=rows(dataset_dir/'index.sqlite',
        'SELECT * FROM observations WHERE timestamp>=? AND timestamp<? '
        'AND (? IS NULL OR global_id=?) ORDER BY timestamp',
        [start,start+seconds,global_id,global_id])
    boxes=frame_boxes(records,cams)
    plan_path=dataset_dir/'floorplan.png'
    plan=cv2.imread(str(plan_path)) if include_plan and plan_path.is_file() else None
    points=project(records,cams,read_json(dataset_dir,'floor_calibration.json',{})) if plan is not None else []
    points.sort(key=lambda p:p['timestamp'])
    point_times=[p['timestamp'] for p in points]
    width,height=648,486;tiles_count=len(cams)+int(include_plan)
    columns=min(3,tiles_count);lines=math.ceil(tiles_count/columns)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width*columns}x{height*lines}',
         '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    cv2.setNumThreads(1)
    with ExitStack() as stack:
        readers={k:stack.enter_context(FrameSource(c['path'])) for k,c in cams.items()}
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            previous={}
            for tick in range(math.ceil(seconds*fps)):
                t=start+tick/fps;tiles=[]
                for cid,c in cams.items():
                    local=t-c.get('offset',0);fid=math.floor(max(0,local)*c['fps']+1e-6)
                    tile=np.zeros((height,width,3),dtype=np.uint8)
                    active=0<=local<bounds[cid]
                    if active:
                        if cid not in previous or previous[cid][0]!=fid:
                            frame=readers[cid].get(fid)
                            if frame is None:
                                raise RuntimeError(f'Camera {cid}: не удалось прочитать кадр {fid}')
                            # Letterbox to retain the original geometry on mixed camera resolutions.
                            fh,fw=frame.shape[:2];scale=min(width/fw,height/fh)
                            resized=cv2.resize(frame,(round(fw*scale),round(fh*scale)),interpolation=cv2.INTER_AREA)
                            h,w=resized.shape[:2];x,y=(width-w)//2,(height-h)//2
                            tile[y:y+h,x:x+w]=resized
                            previous[cid]=(fid,tile,scale,x,y)
                        _,base,scale,x,y=previous[cid]
                        tile=base.copy()
                    for r in boxes.get((cid,fid),[]) if active else []:
                        x1,x2=[round(x+r[k]*scale) for k in ('x1','x2')];y1,y2=[round(y+r[k]*scale) for k in ('y1','y2')]
                        color=shade(r['global_id']);cv2.rectangle(tile,(x1,y1),(x2,y2),color,2)
                        label=f"G{r['global_id']} / L{r['local_id']}"
                        cv2.putText(tile,label,(x1,max(45,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
                    cv2.rectangle(tile,(0,0),(width,28),(25,20,15),-1)
                    cv2.putText(tile,f'Camera {cid} | {t:05.1f}s | {c["fps"]:g} FPS',(10,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
                    tiles.append(tile)
                board=np.full((height,width,3),(32,24,15),dtype=np.uint8)
                if plan is not None:
                    scale=min((width-20)/plan.shape[1],(height-66)/plan.shape[0]);pw,ph=round(plan.shape[1]*scale),round(plan.shape[0]*scale);left=(width-pw)//2;top=48
                    board[top:top+ph,left:left+pw]=cv2.resize(plan,(pw,ph))
                    recent=defaultdict(list)
                    for p in points[bisect_left(point_times,t-10):bisect_right(point_times,t)]:
                        recent[(p['camera_id'],p['local_id'],p['global_id'])].append(p)
                    for ps in recent.values():
                        for a,b in zip(ps,ps[1:]):
                            if b['timestamp']-a['timestamp']<=.5:
                                cv2.line(board,(round(left+a['x']*pw),round(top+a['y']*ph)),(round(left+b['x']*pw),round(top+b['y']*ph)),shade(b['global_id']),2,cv2.LINE_AA)
                        p=ps[-1]
                        if t-p['timestamp']<=.25:
                            pos=(round(left+p['x']*pw),round(top+p['y']*ph));cv2.circle(board,pos,5,shade(p['global_id']),-1)
                            cv2.putText(board,f"G{p['global_id']}",(pos[0]+6,pos[1]),cv2.FONT_HERSHEY_SIMPLEX,.42,(10,10,10),2,cv2.LINE_AA)
                cv2.putText(board,'APPROXIMATE FLOOR MAP',(12,21),cv2.FONT_HERSHEY_SIMPLEX,.58,(80,210,250),1,cv2.LINE_AA)
                cv2.putText(board,'Manual estimate / no interpolation across gaps',(12,40),cv2.FONT_HERSHEY_SIMPLEX,.39,(220,220,220),1,cv2.LINE_AA)
                if include_plan:tiles.append(board)
                while len(tiles)<columns*lines:tiles.append(np.zeros((height,width,3),dtype=np.uint8))
                mosaic=np.vstack([np.hstack(tiles[j:j+columns]) for j in range(0,len(tiles),columns)])
                process.stdin.write(mosaic.tobytes())
                if tick%fps==0:print(f'Export {t:.0f}/{seconds}s',flush=True)
        except BaseException:
            process.terminate();raise
        finally:
            process.stdin.close();code=process.wait()
        if code:raise RuntimeError(f'FFmpeg failed: {code}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);p.add_argument('output',type=Path);p.add_argument('--seconds',type=float);p.add_argument('--fps',type=int,default=25)
    p.add_argument('--global-id',type=int);p.add_argument('--cameras-only',action='store_true')
    a=p.parse_args()
    if (a.seconds is not None and (not math.isfinite(a.seconds) or a.seconds<=0)) or a.fps<1:
        p.error('seconds and fps must be positive')
    render(a.dataset,a.output,a.seconds,a.fps,global_id=a.global_id,include_plan=not a.cameras_only)
