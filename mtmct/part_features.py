"""Cheap view-tolerant upper/lower-body colour features for office videos."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .frame_sources import FrameSource
from .features import crop_quality


def _histogram(image: np.ndarray) -> list[float]:
    if image.size == 0:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256]).reshape(-1)
    norm = float(np.linalg.norm(hist))
    return (hist / norm).tolist() if norm > 1e-9 else []


def body_parts(crop: np.ndarray) -> dict[str, list[float]]:
    """Ignore head/background margins and describe clothing in two bands."""
    height = crop.shape[0]
    width = crop.shape[1]
    result = {'part_quality':{}, 'schema':'colour-visibility-v2'}
    shape = min(1.,width/max(.25*height,1.))
    for name,top,bottom in [('upper',.12,.56),('lower',.56,.96)]:
        band = crop[int(top*height):int(bottom*height)]
        center = band[:,int(.15*width):max(int(.85*width),int(.15*width)+1)]
        edges = np.concatenate((band[:,:max(1,int(.15*width))],band[:,int(.85*width):]),axis=1)
        hist,bg = _histogram(center),_histogram(edges)
        result[name] = hist
        if center.size and hist and bg:
            gray = cv2.cvtColor(center,cv2.COLOR_BGR2GRAY)
            exposure = float(np.mean((gray>10)&(gray<245)))
            separation = max(0.,1-float(np.dot(hist,bg)))
            # Conservative proxy, not a segmentation/pose confidence: a
            # colour band indistinguishable from its margins is weak evidence.
            result['part_quality'][name] = shape*exposure*(.2+.8*separation)
        else:
            result['part_quality'][name] = 0.
    return result


def attach_body_parts(tracklets: list[dict], observations_path: str | Path,
                      camera_sources: dict[int, str | Path], limit: int = 6,
                      min_height: int = 48, min_width: int = 16) -> None:
    track_keys = {(int(row["camera_id"]), int(row["local_id"])) for row in tracklets}
    wanted = {
        (int(row["camera_id"]), int(row["local_id"]), int(frame))
        for row in tracklets for frame in row.get("feature_frames", [])[:limit]
    }
    observations = {}
    observation_frames = defaultdict(list)
    boundary_boxes = defaultdict(list)
    with open(observations_path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            track_key = (int(row["camera_id"]), int(row["local_id"]))
            frame_id = int(row["frame_id"])
            key = (*track_key, frame_id)
            if track_key in track_keys:
                observation_frames[track_key].append(frame_id)
                bbox = tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2"))
                boundary_boxes[track_key].append((frame_id, bbox))
            if key in wanted:
                observations[key] = tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2"))

    requests = defaultdict(lambda: defaultdict(list))
    by_key = {}
    for row in tracklets:
        key = (int(row["camera_id"]), int(row["local_id"]))
        by_key[key] = row
        row["body_parts"] = []
        row["observation_frames"] = observation_frames[key]
        row["observation_boxes"] = {
            str(frame): list(box) for frame, box in boundary_boxes[key]
        }
        for frame in row.get("feature_frames", [])[:limit]:
            if (*key, int(frame)) in observations:
                requests[key[0]][int(frame)].append(key)

    dimensions = {}
    for camera, frames in requests.items():
        reader = FrameSource(camera_sources[camera])
        try:
            dimensions[camera] = reader.dimensions()
            for frame_id in sorted(frames):
                frame = reader.get(frame_id)
                if frame is None:
                    continue
                height, width = frame.shape[:2]
                for key in frames[frame_id]:
                    x1, y1, x2, y2 = observations[(*key, frame_id)]
                    left, top = max(0, int(x1)), max(0, int(y1))
                    right, bottom = min(width, int(x2)), min(height, int(y2))
                    if right - left >= min_width and bottom - top >= min_height:
                        crop = frame[top:bottom, left:right]
                        sample = body_parts(crop)
                        sample['quality'] = crop_quality(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),1.,
                            left==0 or top==0 or right==width or bottom==height)
                        measured = dict(zip(by_key[key].get('feature_frames',[]),by_key[key].get('feature_quality',[])))
                        sample['quality'] = min(sample['quality'],float(measured.get(frame_id,sample['quality'])))
                        by_key[key]["body_parts"].append(sample)
        finally:
            reader.close()

    for key, row in by_key.items():
        if not boundary_boxes[key] or key[0] not in dimensions:
            continue
        width, height = dimensions[key[0]]
        ordered = sorted(boundary_boxes[key])
        normalize = lambda box: [box[0] / width, box[1] / height,
                                 box[2] / width, box[3] / height]
        row["start_bbox_norm"] = normalize(ordered[0][1])
        row["end_bbox_norm"] = normalize(ordered[-1][1])
