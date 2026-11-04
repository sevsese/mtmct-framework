"""Render two or more synchronized views with global IDs into an MP4 mosaic."""
from __future__ import annotations
from collections import defaultdict
import argparse
import csv
from pathlib import Path
import cv2
from .geometry import camera_number


def id_color(gid: int) -> tuple[int,int,int]:
    # Stable BGR colors for high contrast.
    return (int((37 * gid + 105) % 180 + 60),
            int((67 * gid + 35) % 180 + 60),
            int((101 * gid + 55) % 180 + 60))


def render(scene_dir: Path, predictions: Path, output: Path,
           max_frames:int=300, stride:int=3, scale:float=0.5,
           cameras:list[int] | None=None):
    paths=sorted((scene_dir/"videos").glob("Camera_*.mp4"))
    if cameras is not None:
        wanted=set(cameras)
        paths=[p for p in paths if camera_number(p.stem) in wanted]
    if len(paths)<2:
        raise ValueError("Need at least 2 video files for an MTMC mosaic")
    tracks=defaultdict(list)
    with open(predictions,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fid=int(r["frame_id"])
            if max_frames and fid>=max_frames:continue
            tracks[(camera_number(r["camera_id"]),fid)].append(r)
    caps={camera_number(p.stem):cv2.VideoCapture(str(p)) for p in paths}
    if not all(c.isOpened() for c in caps.values()):
        raise RuntimeError("Cannot open all camera videos")
    try:
        first=next(iter(caps.values()))
        source_fps=first.get(cv2.CAP_PROP_FPS) or 30.
        fh=int(first.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale)
        fw=int(first.get(cv2.CAP_PROP_FRAME_WIDTH)*scale)
        if fw<=0 or fh<=0:raise ValueError("Invalid scale or video resolution")
        output.parent.mkdir(parents=True,exist_ok=True)
        writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),
                               max(1.,source_fps/stride),(fw*len(caps),fh))
        if not writer.isOpened():
            raise RuntimeError("Could not create output mp4")
        frame_id=0
        while max_frames<=0 or frame_id<max_frames:
            images=[]
            for cam,cap in caps.items():
                ok,frame=cap.read()
                if not ok:break
                if frame_id % stride != 0:
                    images.append(None)
                    continue
                h,w=frame.shape[:2]
                for r in tracks.get((cam,frame_id),[]):
                    x1,y1,x2,y2=[int(float(r[k])) for k in ("x1","y1","x2","y2")]
                    gid=int(r["global_id"])
                    col=id_color(gid)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),col,3)
                    cv2.putText(frame,f"GLOBAL {gid}", (x1,max(20,y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.75,col,2)
                cv2.putText(frame,f"Camera {cam}  frame {frame_id}",(25,40),
                            cv2.FONT_HERSHEY_SIMPLEX,1.,(255,255,255),3)
                images.append(cv2.resize(frame,(fw,fh),interpolation=cv2.INTER_AREA))
            if len(images)!=len(caps):break
            if frame_id%stride==0:
                import numpy as np
                writer.write(np.concatenate(images,axis=1))
            frame_id+=1
        writer.release()
        print("Created mosaic:",output)
    finally:
        for cap in caps.values():cap.release()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scene-dir",type=Path,required=True)
    p.add_argument("--pred",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("runs/mosaic.mp4"))
    p.add_argument("--max-frames",type=int,default=300)
    p.add_argument("--stride",type=int,default=3)
    p.add_argument("--scale",type=float,default=0.5)
    p.add_argument("--cameras",type=int,nargs="+",default=None)
    a=p.parse_args()
    render(a.scene_dir,a.pred,a.out,a.max_frames,a.stride,a.scale,a.cameras)

if __name__=="__main__":main()
