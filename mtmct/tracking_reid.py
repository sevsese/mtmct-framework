"""WILDTRACK: YOLO detections + appearance-aware online association (not ByteTrack).

Every annotated image is processed in chronological order (2 FPS). Matching uses
both OSNet cosine similarity and displacement; tracker IDs expire after a small
number of processed images. Raw YOLO detections are saved before tracking.

The module is intentionally separate from tracking.py to preserve the baseline.
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .cache import write_manifest
from .export import FIELDS
from .features import MAX_SAMPLES, crop_quality
from .frame_sources import FrameSource
from .reid import ReID, normalized
from .weights import MODELS, download_weights


RAW_FIELDS = ["camera_id", "frame_id", "detection_index", "x1", "y1", "x2", "y2",
              "confidence", "assigned_local_id", "decision", "cosine", "motion_heights"]


def bbox_iou(a, b):
    x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0:
        return 0.0
    aa = max(0.0, float(a[2]-a[0])) * max(0.0, float(a[3]-a[1]))
    ab = max(0.0, float(b[2]-b[0])) * max(0.0, float(b[3]-b[1]))
    return area / max(aa + ab - area, 1e-9)


def motion_heights(a, b):
    cx1, cy1 = (float(a[0])+float(a[2]))/2, (float(a[1])+float(a[3]))/2
    cx2, cy2 = (float(b[0])+float(b[2]))/2, (float(b[1])+float(b[3]))/2
    mean_h = max(1.0, ((float(a[3])-float(a[1]))+(float(b[3])-float(b[1])))/2)
    return math.hypot(cx1-cx2, cy1-cy2) / mean_h


def _normalize(x):
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-9 else x


@dataclass
class MatchConfig:
    max_missed: int = 2  # annotated images, 0.5 sec each; not raw filename difference
    max_motion_heights: float = 1.8  # per annotated image
    match_cos: float = 0.72
    lost_cos: float = 0.82
    iou_override: float = 0.30
    min_new_conf: float = 0.12


@dataclass
class Track:
    local_id: int
    last_index: int
    last_frame: int
    last_box: np.ndarray
    prototype: np.ndarray
    first_frame: int
    embedding_sum: np.ndarray
    samples: int = 1
    positions: list = field(default_factory=list)
    feature_samples: list = field(default_factory=list)


class ReIDLocalTracker:
    """Greedy one-to-one matching with hard appearance, motion and age gates.

    Only recent tracks are eligible: a track from frame 0 cannot be blindly
    assigned to a stranger at frame 65 because its ID was lost long ago.
    """

    def __init__(self, config: MatchConfig):
        self.cfg = config
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def step(self, frame_id: int, frame_index: int, detections: list[dict]) -> list[dict]:
        cfg = self.cfg
        candidates = []
        for lid, track in self.tracks.items():
            dt = frame_index - track.last_index
            if dt < 1 or dt > cfg.max_missed + 1:
                continue
            for i, det in enumerate(detections):
                vec = det["embedding"]
                sim = float(track.prototype @ vec)
                iou = bbox_iou(track.last_box, det["bbox"])
                displacement = motion_heights(track.last_box, det["bbox"])
                # Reject any jump that is too large for the elapsed time.
                if displacement > cfg.max_motion_heights * dt:
                    continue
                threshold = cfg.match_cos if dt == 1 else cfg.lost_cos
                # A large box overlap may compensate for a moderate appearance
                # change, but not an entirely different person.
                if dt == 1 and iou >= cfg.iou_override:
                    threshold = min(threshold, 0.62)
                if sim < threshold:
                    continue
                score = sim + 0.09 * iou - 0.045 * (displacement / dt) - 0.015 * (dt - 1)
                candidates.append((score, lid, i, sim, displacement))
        # One-to-one assignment: an observation cannot inherit two IDs and an
        # existing ID cannot be assigned to two different people on one image.
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        matched_tracks, matched_dets = set(), set()
        for score, lid, i, sim, displacement in candidates:
            if lid in matched_tracks or i in matched_dets:
                continue
            matched_tracks.add(lid); matched_dets.add(i)
            det = detections[i]
            track = self.tracks[lid]
            vec = det["embedding"]
            track.prototype = _normalize(0.7 * track.prototype + 0.3 * vec)
            track.embedding_sum += vec
            track.samples += 1
            track.last_index = frame_index
            track.last_frame = frame_id
            track.last_box = det["bbox"].copy()
            track.feature_samples.append((det["quality"], frame_id, vec.copy()))
            det.update(local_id=lid, decision="matched", cosine=sim, motion_heights=displacement)
        for i, det in enumerate(detections):
            if i in matched_dets:
                continue
            if det["confidence"] < cfg.min_new_conf:
                det.update(local_id=None, decision="low_new_conf", cosine=None, motion_heights=None)
                continue
            lid = self.next_id
            self.next_id += 1
            vec = det["embedding"].copy()
            track = Track(local_id=lid, last_index=frame_index, last_frame=frame_id,
                          last_box=det["bbox"].copy(), prototype=vec.copy(),
                          first_frame=frame_id, embedding_sum=vec.copy(),
                          feature_samples=[(det["quality"], frame_id, vec.copy())])
            self.tracks[lid] = track
            det.update(local_id=lid, decision="new", cosine=None, motion_heights=None)
        return detections

    def summaries(self, camera_id: int) -> list[dict]:
        result = []
        for lid, tr in sorted(self.tracks.items()):
            best = sorted(tr.feature_samples, key=lambda row: -row[0])[:MAX_SAMPLES]
            best.sort(key=lambda row: row[1])
            result.append({
                "camera_id": int(camera_id), "local_id": lid,
                "start_frame": tr.first_frame, "end_frame": tr.last_frame,
                "positions": [],
                "embedding": _normalize(tr.embedding_sum).tolist(),
                "feature_frames": [int(fid) for q, fid, e in best],
                "feature_quality": [float(q) for q, fid, e in best],
                "feature_vectors": [e.tolist() for q, fid, e in best],
                "samples": tr.samples,
                "tracking_method": "reid_motion_v1",
            })
        return result


def _bbox_rows(boxes, shape):
    h, w = shape[:2]
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    if len(xyxy) == 0:
        return []
    conf = boxes.conf.cpu().numpy()
    result = []
    for idx, (box, score) in enumerate(zip(xyxy, conf)):
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = max(0., min(w - 1., x1)), max(0., min(float(w), x2))
        y1, y2 = max(0., min(h - 1., y1)), max(0., min(float(h), y2))
        result.append({"detection_index": idx, "bbox": np.array([x1,y1,x2,y2], dtype=np.float32),
                       "confidence": float(score), "decision": "unprocessed"})
    return result


def run_wildtrack_local(root: Path, output: Path, cameras: list[int], frame_ids: list[int],
                        *, model_name: str, reid_model: str, reid_weights: str | None,
                        device: str, imgsz: int, conf: float, match_cfg: MatchConfig,
                        reid_batch: int = 32) -> dict:
    from ultralytics import YOLO
    from .wildtrack import frame_sources

    if reid_model not in MODELS:
        raise ValueError(f"Unsupported ReID model: {reid_model}")
    if reid_batch < 1:
        raise ValueError("reid_batch must be positive")
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in this environment. Activate your GPU-enabled neuro_clean environment.")
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    if reid_weights is None:
        guess = Path('weights') / MODELS[reid_model]
        reid_weights = str(guess if guess.is_file() else download_weights(reid_model))
    if not Path(reid_weights).is_file():
        raise FileNotFoundError(f"ReID checkpoint not found: {reid_weights}")

    sources = frame_sources(root, cameras)
    output.mkdir(parents=True, exist_ok=True)
    detector = YOLO(model_name)
    encoder = ReID("osnet", device=device, weights=reid_weights, model_name=reid_model)
    all_tracks = []
    performance = []
    retained_embeddings, retained_meta = [], []
    frame_set = set(frame_ids)
    with (output / "observations.csv").open("w", encoding="utf-8", newline="") as tracked_file, \
         (output / "raw_detections.csv").open("w", encoding="utf-8", newline="") as raw_file:
        w_tracked = csv.DictWriter(tracked_file, fieldnames=FIELDS)
        w_raw = csv.DictWriter(raw_file, fieldnames=RAW_FIELDS)
        w_tracked.writeheader(); w_raw.writeheader()
        for camera in cameras:
            tracker = ReIDLocalTracker(match_cfg)
            reader = FrameSource(sources[camera], fps=2.0)
            raw_count = assigned_count = total_frames = 0
            t0 = time.perf_counter()
            try:
                for index, (fid, frame) in enumerate(reader.iter_frames(0, 1, frame_set)):
                    boxes = detector.predict(source=frame, classes=[0], conf=conf, imgsz=imgsz,
                                             device=device, verbose=False)[0].boxes
                    raw = _bbox_rows(boxes, frame.shape)
                    valid = []
                    crops = []
                    for det in raw:
                        x1,y1,x2,y2 = det["bbox"]
                        if x2-x1 < 4 or y2-y1 < 8:
                            det["decision"] = "tiny_box"
                            continue
                        crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        if crop.size == 0:
                            det["decision"] = "empty_crop"
                            continue
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        det["quality"] = crop_quality(rgb, det["confidence"],
                            x1 <= 0 or y1 <= 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0])
                        valid.append(det); crops.append(rgb)
                    embeddings = []
                    for start in range(0, len(crops), reid_batch):
                        embeddings.extend(encoder.encode(crops[start:start + reid_batch]))
                    if len(embeddings) != len(valid):
                        raise RuntimeError("ReID encoder returned wrong number of embeddings")
                    for det, vec in zip(valid, embeddings):
                        det["embedding"] = _normalize(vec)
                    tracker.step(fid, index, valid)
                    for det in raw:
                        bbox = det["bbox"]
                        w_raw.writerow({
                            "camera_id":camera, "frame_id":fid, "detection_index":det["detection_index"],
                            **dict(zip(["x1","y1","x2","y2"], [round(float(x),2) for x in bbox])),
                            "confidence":round(det["confidence"],4),
                            "assigned_local_id":det.get("local_id", "") if det.get("local_id") is not None else "",
                            "decision":det["decision"],
                            "cosine": round(det["cosine"],4) if det.get("cosine") is not None else "",
                            "motion_heights":round(det["motion_heights"],4) if det.get("motion_heights") is not None else "",
                        })
                        raw_count += 1
                        if det.get("local_id") is None:
                            continue
                        assigned_count += 1
                        local_id = int(det["local_id"])
                        w_tracked.writerow({
                            "camera_id":camera, "frame_id":fid,"local_id":local_id,
                            **dict(zip(["x1","y1","x2","y2"], [f"{float(x):.2f}" for x in bbox])),
                            "confidence":f"{det['confidence']:.4f}","world_x":"","world_y":"",
                        })
                        retained_embeddings.append(det["embedding"].copy())
                        retained_meta.append([camera, fid, local_id])
                    total_frames += 1
                    if total_frames == 1 or total_frames % 5 == 0:
                        print(f"  camera={camera} image={total_frames}/{len(frame_ids)} "
                              f"raw={len(raw)} tracked={sum(d.get('local_id') is not None for d in raw)} "
                              f"local_ids={len(tracker.tracks)}", flush=True)
            finally:
                reader.close()
            all_tracks.extend(tracker.summaries(camera))
            elapsed = time.perf_counter() - t0
            performance.append({"camera_id":camera,"frames":total_frames,"raw_yolo_boxes":raw_count,
                                "tracked_boxes":assigned_count,"local_tracklets":len(tracker.tracks),
                                "seconds":round(elapsed,2)})
            print(f"Camera {camera}: raw YOLO={raw_count}, exported={assigned_count}, "
                  f"local IDs={len(tracker.tracks)}", flush=True)

    (output / "local_tracklets.json").write_text(json.dumps(all_tracks), encoding="utf-8")
    (output / "performance.json").write_text(json.dumps(performance,indent=2), encoding="utf-8")
    if retained_embeddings:
        np.savez_compressed(output / "observation_embeddings.npz",
                            vectors=np.stack(retained_embeddings).astype(np.float32),
                            camera_frame_local=np.asarray(retained_meta,dtype=np.int32))
    dimensions = {len(t["embedding"]) for t in all_tracks}
    if len(dimensions)>1:
        raise ValueError(f"Inconsistent ReID dimensions: {dimensions}")
    write_manifest(output, scene="WILDTRACK", cameras=cameras,
                   inputs=[sources[c] for c in cameras], detector=model_name,
                   reid=reid_weights, feature_dimension=next(iter(dimensions), None),
                   config={"tracker":"reid_motion_v1", "imgsz":imgsz,"conf":conf,
                           "frame_ids":frame_ids,"fps_fallback":2.0,
                           "model":model_name,"reid_model":reid_model,
                           "match":match_cfg.__dict__})
    return {"cameras": performance, "local_tracklets":len(all_tracks),
            "raw_yolo_boxes":sum(x["raw_yolo_boxes"] for x in performance),
            "tracked_boxes":sum(x["tracked_boxes"] for x in performance)}
