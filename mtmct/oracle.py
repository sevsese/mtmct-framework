"""Oracle tracklets: GT boxes with camera-local IDs and real crop features."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .cache import write_config, write_manifest
from .export import FIELDS
from .nvidia import NvidiaDatasetAdapter
from .reid import ReID, normalized
from .features import MAX_SAMPLES, crop_quality


def generate(scene_dir: Path, output: Path, cameras: list[int], max_frames: int,
             device: str = "cpu", reid_method: str = "color",
             reid_weights: str | None = None, reid_model: str = "osnet_x1_0",
             sample_every: int = 5, analysis_labels: Path | None = None):
    if max_frames <= 0 or sample_every <= 0:
        raise ValueError("max_frames and sample_every must be positive")
    if reid_method == "osnet" and not reid_weights:
        from .weights import download_weights
        reid_weights = str(download_weights(reid_model))
    encoder = ReID(reid_method, device=device, weights=reid_weights, model_name=reid_model)

    selected = set(cameras)
    by_camera_frame = defaultdict(list)
    local_ids: dict[int, dict[int, int]] = defaultdict(dict)
    for observation in NvidiaDatasetAdapter(scene_dir / "ground_truth.json").observations(
            strict=False, max_frames=max_frames):
        if observation.camera_id not in selected:
            continue
        camera_map = local_ids[observation.camera_id]
        local_id = camera_map.setdefault(observation.gt_object_id, len(camera_map) + 1)
        by_camera_frame[(observation.camera_id, observation.frame_index)].append(
            (local_id, observation.bbox_xyxy))

    output.mkdir(parents=True, exist_ok=True)
    tracks = defaultdict(lambda: {
        "start_frame": 10**12, "end_frame": -1, "embedding_sum": None, "samples": 0,
        "feature_samples": [],
    })
    performance = []
    videos = [scene_dir / "videos" / f"Camera_{camera:04d}.mp4" for camera in cameras]
    with (output / "observations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for camera, video in zip(cameras, videos):
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot read video: {video}")
            started = time.perf_counter()
            processed = 0
            try:
                for frame_index in range(max_frames):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    processed += 1
                    height, width = frame.shape[:2]
                    crops, crop_keys = [], []
                    for local_id, raw_box in by_camera_frame[(camera, frame_index)]:
                        x1, y1, x2, y2 = raw_box
                        x1, x2 = max(0.0, min(x1, width - 1.0)), max(0.0, min(x2, float(width)))
                        y1, y2 = max(0.0, min(y1, height - 1.0)), max(0.0, min(y2, float(height)))
                        if x2 - x1 < 2 or y2 - y1 < 2:
                            continue
                        key = (camera, local_id)
                        track = tracks[key]
                        track["start_frame"] = min(track["start_frame"], frame_index)
                        track["end_frame"] = max(track["end_frame"], frame_index)
                        writer.writerow({
                            "camera_id": camera, "frame_id": frame_index, "local_id": local_id,
                            "x1": f"{x1:.2f}", "y1": f"{y1:.2f}",
                            "x2": f"{x2:.2f}", "y2": f"{y2:.2f}",
                            "confidence": "1.0000", "world_x": "", "world_y": "",
                        })
                        if track["samples"] == 0 or frame_index % sample_every == 0:
                            crop = frame[int(y1):int(y2), int(x1):int(x2)]
                            if crop.size:
                                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                clipped = x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height
                                crop_keys.append((key, crop_quality(crops[-1], 1.0, clipped)))
                    for (key, quality), embedding in zip(crop_keys, encoder.encode(crops)):
                        track = tracks[key]
                        if track["embedding_sum"] is None:
                            track["embedding_sum"] = np.zeros_like(embedding, dtype=np.float64)
                        track["embedding_sum"] += embedding
                        track["samples"] += 1
                        track["feature_samples"].append((quality, frame_index, embedding.tolist()))
                        track["feature_samples"] = sorted(
                            track["feature_samples"], key=lambda sample: -sample[0])[:MAX_SAMPLES]
            finally:
                capture.release()
            seconds = time.perf_counter() - started
            performance.append({"camera_id": camera, "processed_frames": processed,
                                "seconds": round(seconds, 2),
                                "fps": round(processed / max(seconds, 1e-9), 2)})

    summaries = []
    for (camera, local_id), track in sorted(tracks.items()):
        embedding = track["embedding_sum"]
        summaries.append({
            "camera_id": camera, "local_id": local_id,
            "start_frame": int(track["start_frame"]), "end_frame": int(track["end_frame"]),
            "positions": [],
            "embedding": normalized(embedding).tolist() if embedding is not None else [],
            "feature_quality": [sample[0] for sample in track["feature_samples"]],
            "feature_frames": [sample[1] for sample in track["feature_samples"]],
            "feature_vectors": [sample[2] for sample in track["feature_samples"]],
            "samples": track["samples"],
            "source": "oracle_gt_bbox",
        })
    (output / "local_tracklets.json").write_text(json.dumps(summaries), encoding="utf-8")
    (output / "performance.json").write_text(json.dumps(performance, indent=2), encoding="utf-8")
    if analysis_labels:
        analysis_labels.parent.mkdir(parents=True, exist_ok=True)
        analysis_labels.write_text(json.dumps([
            {"camera_id": camera, "local_id": local_id, "gt_object_id": gt_id}
            for camera, mapping in sorted(local_ids.items())
            for gt_id, local_id in sorted(mapping.items(), key=lambda item: item[1])
        ]), encoding="utf-8")
    dimensions = {len(row["embedding"]) for row in summaries if row["embedding"]}
    write_manifest(
        output, scene=scene_dir.name, cameras=cameras,
        inputs=[scene_dir / "ground_truth.json", *videos],
        detector="oracle_gt_boxes", reid=reid_weights or reid_method,
        feature_dimension=next(iter(dimensions), None),
        config={"max_frames": max_frames, "sample_every": sample_every,
                "reid_method": reid_method, "reid_model": reid_model,
                "gt_identity_exposed_to_matcher": False},
    )
    write_config(output, {"mode": "oracle_tracklets", "scene": scene_dir.name,
                          "cameras": cameras, "max_frames": max_frames,
                          "device": device, "reid": reid_method,
                          "reid_model": reid_model, "sample_every": sample_every,
                          "gt_identity_exposed_to_matcher": False})
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Create camera-local tracklets from GT boxes")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cameras", type=int, nargs="+", required=True)
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reid", choices=["osnet", "color"], default="osnet")
    parser.add_argument("--reid-weights")
    parser.add_argument("--reid-model", default="osnet_x1_0")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--analysis-labels", type=Path,
                        help="Optional GT labels for analysis only; matcher never reads this file")
    args = parser.parse_args()
    summaries = generate(args.scene_dir, args.out, args.cameras, args.max_frames,
                         args.device, args.reid, args.reid_weights,
                         args.reid_model, args.sample_every, args.analysis_labels)
    print(f"Wrote {len(summaries)} oracle tracklets to {args.out}")


if __name__ == "__main__":
    main()
