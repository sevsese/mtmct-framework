"""Export time-aligned camera overlays and the approximate plan as H.264.

Usage: python -m backend.export_review DATASET_DIRECTORY output.mp4 --seconds 30
Requires optional imageio-ffmpeg. Reads existing observations; runs no detector.
"""
import argparse
import colorsys
from collections import defaultdict
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


def render(dataset_dir,output,seconds=30,fps=10):
    import imageio_ffmpeg
    d=read_json(dataset_dir,'dataset.json');cams={c['camera_id']:c for c in d['cameras']}
    records=rows(dataset_dir/'index.sqlite','SELECT * FROM observations WHERE timestamp>=0 AND timestamp<? ORDER BY timestamp',[seconds])
    by_frame=defaultdict(list)
    for r in records:by_frame[(r['camera_id'],r['frame_id'])].append(r)
    cal=read_json(dataset_dir,'floor_calibration.json',{})
    plan=cv2.imread(str(dataset_dir/'floorplan.png'));points=project(records,cams,cal)
    width,height=648,486;columns=3;lines=math.ceil((len(cams)+1)/columns)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width*columns}x{height*lines}',
         '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with ExitStack() as stack:
        readers={k:stack.enter_context(FrameSource(c['path'])) for k,c in cams.items()}
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for tick in range(math.ceil(seconds*fps)):
                t=tick/fps;tiles=[]
                for cid,c in cams.items():
                    local=t-c.get('offset',0);fid=math.floor(max(0,local)*c['fps']+1e-6)
                    frame=readers[cid].get(fid) if 0<=local<c['duration'] else None
                    tile=cv2.resize(frame,(width,height)) if frame is not None else np.zeros((height,width,3),dtype=np.uint8)
                    for r in by_frame.get((cid,fid),[]) if frame is not None else []:
                        x1,x2=[round(r[k]*width/c['width']) for k in ('x1','x2')];y1,y2=[round(r[k]*height/c['height']) for k in ('y1','y2')]
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
                    for p in points:
                        if t-10<=p['timestamp']<=t:recent[(p['camera_id'],p['local_id'],p['global_id'])].append(p)
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
                tiles.append(board)
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);p.add_argument('output',type=Path);p.add_argument('--seconds',type=float,default=30);p.add_argument('--fps',type=int,default=10)
    a=p.parse_args()
    if a.seconds<=0 or a.fps<1:p.error('seconds and fps must be positive')
    render(a.dataset,a.output,a.seconds,a.fps)
