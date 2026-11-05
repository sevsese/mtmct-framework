"""YOLO26 + per-camera ByteTrack + ReID feature aggregation.

Each camera gets a NEW YOLO() instance so local tracker state cannot leak
between independent video sequences. Detected samples are streamed to CSV.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
from .geometry import calibration_maps, camera_number, project_pixel
from .reid import ReID, normalized
from .export import FIELDS
from .cache import write_manifest


def run(scene_dir: Path, output: Path, model_name: str,
        reid_method: str, reid_weights: str | None, device: str,
        max_frames: int, stride: int, sample_every: int,
        imgsz: int, conf: float, reid_model: str = "osnet_x1_0",
        cameras: list[int] | None = None):
    from ultralytics import YOLO
    if stride<1 or sample_every<1:
        raise ValueError("stride and sample_every must be positive")
    videos=sorted((scene_dir/"videos").glob("Camera_*.mp4"))
    if cameras is not None:
        wanted=set(cameras)
        videos=[p for p in videos if camera_number(p.stem) in wanted]
        found={camera_number(p.stem) for p in videos}
        if found != wanted:
            raise FileNotFoundError(f"Missing camera videos: {sorted(wanted-found)}")
    if not videos:
        raise FileNotFoundError(f"No Camera_*.mp4 files under {scene_dir}/videos")
    calibration=calibration_maps(scene_dir/"calibration.json")
    if reid_method == "osnet" and not reid_weights:
        from .weights import download_weights
        reid_weights = str(download_weights(reid_model))
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA-enabled PyTorch detected. Run python -m mtmct.gpu_check")
        print('CUDA GPU:', torch.cuda.get_device_name(0))
    reid=ReID(reid_method, device=device, weights=reid_weights, model_name=reid_model)
    output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    performance=[]
    with open(output/"observations.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        for video in videos:
            cam=camera_number(video.stem)
            H=calibration.get(cam)
            if H is None:
                print(f"WARNING camera {cam}: no valid cameraMatrix, world positions unavailable")
            cap=cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot read video: {video}")
            model=YOLO(model_name)  # fresh tracker for each camera
            print(f"Processing camera {cam}: {video.name}",flush=True)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  video FPS={video_fps:.2f}; size={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            # id -> data. Only a sum and count for embeddings, not every crop.
            by_id=defaultdict(lambda: {"start_frame":10**12,"end_frame":-1,
                                       "positions":[],"embedding_sum":None,"samples":0})
            fid=0
            processed=0
            if device.startswith("cuda"):
                import torch
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            try:
                while fid<max_frames or max_frames<=0:
                    ok,frame=cap.read()
                    if not ok:break
                    if fid % stride != 0:
                        fid+=1;continue
                    result=model.track(frame,persist=True,tracker="bytetrack.yaml",
                                       classes=[0],conf=conf,imgsz=imgsz,
                                       device=device,verbose=False)[0]
                    crops=[]; sampled_keys=[]
                    if result.boxes is not None and result.boxes.id is not None:
                        ids=result.boxes.id.int().cpu().numpy()
                        rects=result.boxes.xyxy.cpu().numpy()
                        scores=result.boxes.conf.cpu().numpy()
                        for local_id,box,score in zip(ids,rects,scores):
                            local_id=int(local_id)
                            x1,y1,x2,y2=[float(x) for x in box]
                            h,w=frame.shape[:2]
                            x1=max(0.,min(x1,w-1.)); x2=max(0.,min(x2,w*1.))
                            y1=max(0.,min(y1,h-1.)); y2=max(0.,min(y2,h*1.))
                            if x2-x1<2 or y2-y1<2:continue
                            key=by_id[local_id]
                            key["start_frame"]=min(key["start_frame"],fid)
                            key["end_frame"]=max(key["end_frame"],fid)
                            foot=project_pixel(H,(x1+x2)/2,y2)
                            if foot is not None:
                                key["positions"].append([fid,foot[0],foot[1]])
                            writer.writerow({
                                "camera_id":cam,"frame_id":fid,"local_id":local_id,
                                "x1":f"{x1:.2f}","y1":f"{y1:.2f}",
                                "x2":f"{x2:.2f}","y2":f"{y2:.2f}",
                                "confidence":f"{float(score):.4f}",
                                "world_x":f"{foot[0]:.4f}" if foot is not None else "",
                                "world_y":f"{foot[1]:.4f}" if foot is not None else ""})
                            # sample features every few frames, not at every frame
                            if key["samples"] == 0 or (fid // stride) % sample_every == 0:
                                crop=frame[int(y1):int(y2),int(x1):int(x2)]
                                if crop.size:
                                    crops.append(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
                                    sampled_keys.append(local_id)
                    embeddings=reid.encode(crops)
                    processed += 1
                    for lid,e in zip(sampled_keys,embeddings):
                        row=by_id[lid]
                        if row["embedding_sum"] is None:
                            row["embedding_sum"]=np.zeros_like(e,dtype=np.float64)
                        row["embedding_sum"]+=e
                        row["samples"]+=1
                    if fid%100==0:
                        print(f" camera={cam} frame={fid} tracks={len(by_id)}",flush=True)
                    fid+=1
            finally:
                cap.release()
            if device.startswith("cuda"):
                import torch
                torch.cuda.synchronize()
            seconds=time.perf_counter()-started
            peak_vram = round(torch.cuda.max_memory_allocated()/1024**3,2) if device.startswith("cuda") else None
            print(f"  processed={processed} frames in {seconds:.1f} s = "
                  f"{processed/max(seconds, 1e-6):.2f} processed FPS; peak CUDA tensor VRAM={peak_vram} GiB")
            summaries_stats={"camera_id":cam,"processed_frames":processed,"seconds":round(seconds,2),
                             "fps":round(processed/max(seconds,1e-6),2),"peak_cuda_tensor_gib":peak_vram}
            performance.append(summaries_stats)
            for lid,track in by_id.items():
                emb=track["embedding_sum"]
                summaries.append({"camera_id":cam,"local_id":lid,
                                  "start_frame":int(track["start_frame"]),
                                  "end_frame":int(track["end_frame"]),
                                  "positions":track["positions"],
                                  "embedding":normalized(emb).tolist() if emb is not None else [],
                                  "samples":track["samples"]})
            del model
            if device.startswith("cuda"):
                import torch
                torch.cuda.empty_cache()
    with open(output/"local_tracklets.json","w",encoding="utf-8") as f:
        json.dump(summaries,f)
    with open(output/"performance.json","w",encoding="utf-8") as f:
        json.dump(performance,f,indent=2)
    dimensions={len(row["embedding"]) for row in summaries if row["embedding"]}
    if len(dimensions)>1:
        raise ValueError(f"Inconsistent ReID feature dimensions: {sorted(dimensions)}")
    write_manifest(
        output, scene=scene_dir.name,
        cameras=[camera_number(path.stem) for path in videos],
        inputs=[scene_dir/"calibration.json", *videos],
        detector=model_name, reid=reid_weights or reid_method,
        feature_dimension=next(iter(dimensions), None),
        config={"max_frames":max_frames,"stride":stride,"sample_every":sample_every,
                "imgsz":imgsz,"conf":conf,"reid_method":reid_method,
                "reid_model":reid_model},
    )
    print(f"Wrote {len(summaries)} tracklets to {output/'local_tracklets.json'}")


def main():
    p=argparse.ArgumentParser(description="Run local YOLO+ByteTrack+ReID inference")
    p.add_argument("--scene-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("runs/demo"))
    p.add_argument("--model",default="yolo26l.pt")
    p.add_argument("--reid",choices=["osnet","color"],default="color")
    p.add_argument("--reid-weights",type=str,default=None)
    p.add_argument("--reid-model",choices=["osnet_x0_25","osnet_x0_75","osnet_x1_0"],default="osnet_x1_0")
    p.add_argument("--device",default="cuda:0",help="cpu or cuda:0")
    p.add_argument("--max-frames",type=int,default=300,help="0 = all frames")
    p.add_argument("--stride",type=int,default=1)
    p.add_argument("--sample-every",type=int,default=5)
    p.add_argument("--imgsz",type=int,default=960)
    p.add_argument("--conf",type=float,default=0.12)
    p.add_argument("--cameras",type=int,nargs="+",default=None)
    a=p.parse_args()
    run(a.scene_dir,a.out,a.model,a.reid,a.reid_weights,a.device,
        a.max_frames,a.stride,a.sample_every,a.imgsz,a.conf,
        reid_model=a.reid_model,cameras=a.cameras)

if __name__=="__main__":main()
