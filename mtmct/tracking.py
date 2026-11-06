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
from .features import MAX_SAMPLES, crop_quality
from .frame_sources import FrameSource


def run(scene_dir: Path, output: Path, model_name: str,
        reid_method: str, reid_weights: str | None, device: str,
        max_frames: int, stride: int, sample_every: int,
        imgsz: int, conf: float, reid_model: str = "osnet_x1_0",
        cameras: list[int] | None = None, tracker: str | Path = "bytetrack.yaml",
        camera_sources: dict[int, Path] | None = None, fps_fallback: float = 30.0,
        frame_ids: set[int] | None = None):
    from ultralytics import YOLO
    if stride<1 or sample_every<1:
        raise ValueError("stride and sample_every must be positive")
    if camera_sources is None:
        found_sources = {camera_number(p.stem): p for p in sorted((scene_dir / 'videos').glob('Camera_*.mp4'))}
    else:
        found_sources = {int(c): Path(path) for c, path in camera_sources.items()}
    selected_cameras = sorted(set(cameras)) if cameras is not None else sorted(found_sources)
    absent = sorted(set(selected_cameras) - set(found_sources))
    if absent:
        raise FileNotFoundError(f'Missing camera sources: {absent}')
    if not selected_cameras:
        raise FileNotFoundError('No camera video or image sources found')
    source_paths = {c: found_sources[c] for c in selected_cameras}
    calibration_path = scene_dir / 'calibration.json'
    calibration = calibration_maps(calibration_path) if calibration_path.is_file() else {}
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
        for cam, video in source_paths.items():
            H = calibration.get(cam)
            if H is None:
                print(f'Camera {cam}: no verified floor projection; using appearance-only positions')
            reader = FrameSource(video, fps=fps_fallback)
            model = YOLO(model_name)  # independent ByteTrack instance per camera
            print(f'Processing camera {cam}: {video}', flush=True)
            video_fps = reader.fps
            vw, vh = reader.dimensions()
            print(f'  FPS={video_fps:.2f}; size={vw}x{vh}')
            # id -> data. Only a sum and count for embeddings, not every crop.
            by_id=defaultdict(lambda: {"start_frame":10**12,"end_frame":-1,
                                       "positions":[],"embedding_sum":None,"samples":0,
                                       "feature_samples":[]})
            fid=0
            processed=0
            if device.startswith("cuda"):
                import torch
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            try:
                for fid, frame in reader.iter_frames(max_frames, stride, frame_ids):
                    result=model.track(frame,persist=True,tracker=str(tracker),
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
                                    clipped = x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h
                                    sampled_keys.append((local_id, crop_quality(crops[-1], float(score), clipped)))
                    embeddings=reid.encode(crops)
                    processed += 1
                    for (lid,quality),e in zip(sampled_keys,embeddings):
                        row=by_id[lid]
                        if row["embedding_sum"] is None:
                            row["embedding_sum"]=np.zeros_like(e,dtype=np.float64)
                        row["embedding_sum"]+=e
                        row["samples"]+=1
                        row["feature_samples"].append((quality,fid,e.tolist()))
                        row["feature_samples"] = sorted(
                            row["feature_samples"], key=lambda sample: -sample[0])[:MAX_SAMPLES]
                    if fid%100==0:
                        print(f" camera={cam} frame={fid} tracks={len(by_id)}",flush=True)
            finally:
                reader.close()
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
                                  "feature_quality":[sample[0] for sample in track["feature_samples"]],
                                  "feature_frames":[sample[1] for sample in track["feature_samples"]],
                                  "feature_vectors":[sample[2] for sample in track["feature_samples"]],
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
        cameras=selected_cameras,
        inputs=[*( [calibration_path] if calibration_path.is_file() else [] ), *source_paths.values()],
        detector=model_name, reid=reid_weights or reid_method,
        feature_dimension=next(iter(dimensions), None),
        config={"max_frames":max_frames,"stride":stride,"sample_every":sample_every,
                "imgsz":imgsz,"conf":conf,"tracker":str(tracker),"reid_method":reid_method,
                "reid_model":reid_model,"camera_sources":{str(c):str(p) for c,p in source_paths.items()},
                "fps_fallback":fps_fallback,
                "frame_ids": sorted(frame_ids) if frame_ids is not None else None},
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
    p.add_argument("--imgsz",type=int,default=1536)
    p.add_argument("--conf",type=float,default=0.12)
    p.add_argument("--tracker",default="bytetrack.yaml")
    p.add_argument("--cameras",type=int,nargs="+",default=None)
    a=p.parse_args()
    run(a.scene_dir,a.out,a.model,a.reid,a.reid_weights,a.device,
        a.max_frames,a.stride,a.sample_every,a.imgsz,a.conf,
        reid_model=a.reid_model,cameras=a.cameras,tracker=a.tracker)

if __name__=="__main__":main()
