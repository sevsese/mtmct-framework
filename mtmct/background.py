"""Weak foreground evidence for static cameras; never suppress a detection."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .frame_sources import FrameSource


class ForegroundEvidence:
    def __init__(self, mode: str = 'mog2', background=None):
        if mode not in {'mog2', 'median_static'}:
            raise ValueError('Unsupported foreground mode')
        if mode == 'median_static' and background is None:
            raise ValueError('median_static requires a background')
        self.mode, self.background = mode, background
        self.model = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True) if mode == 'mog2' else None
        self.seen = 0

    def update(self, frame):
        small = cv2.resize(frame, (640, 360))
        self.seen += 1
        if self.model is not None:
            mask = self.model.apply(small)
            return None if self.seen < 30 else (mask == 255)
        delta = cv2.absdiff(small, self.background)
        return np.max(delta, axis=2) > 30

    @staticmethod
    def fraction(mask, box, shape):
        if mask is None:
            return None
        h, w = shape[:2]
        x1, y1, x2, y2 = box
        xa, xb = max(0, int(x1 * 640 / w)), min(640, int(np.ceil(x2 * 640 / w)))
        ya, yb = max(0, int(y1 * 360 / h)), min(360, int(np.ceil(y2 * 360 / h)))
        crop = mask[ya:yb, xa:xb]
        return float(crop.mean()) if crop.size else None


def annotate(tracklets, observation_path: Path, scene_dir: Path, start: int, end: int, mode='mog2'):
    if not 0 <= start < end:
        raise ValueError("Expected 0 <= start < end")
    rows = copy.deepcopy(tracklets)
    targets = {}
    for t in rows:
        t['foreground_ratio'] = [None] * len(t.get('feature_frames', []))
        for i, frame in enumerate(t.get('feature_frames', [])):
            targets[(int(t['camera_id']), int(t['local_id']), int(frame))] = (t, i)
    boxes = {}
    with observation_path.open(newline='', encoding='utf-8') as stream:
        for obs in csv.DictReader(stream):
            key = tuple(int(obs[k]) for k in ('camera_id', 'local_id', 'frame_id'))
            if key in targets:
                boxes[key] = [float(obs[k]) for k in ('x1', 'y1', 'x2', 'y2')]
    for cam in sorted({t['camera_id'] for t in rows}):
        video = scene_dir / 'videos' / f'Camera_{cam:04d}.mp4'
        background = None
        if mode == 'median_static':
            with FrameSource(video) as source:
                samples = [cv2.resize(source.get(int(f)), (640, 360)) for f in np.linspace(start, end-1, min(25, end-start))]
            background = np.median(samples, axis=0).astype(np.uint8)
        estimator = ForegroundEvidence(mode, background)
        by_frame = {}
        for key in boxes:
            if key[0] == cam:
                by_frame.setdefault(key[2], []).append(key)
        with FrameSource(video) as source:
            for fid in range(start, end):
                frame = source.get(fid)
                if frame is None:
                    raise RuntimeError(f"Missing camera {cam} frame {fid}")
                mask = estimator.update(frame)
                for key in by_frame.get(fid, []):
                    t, i = targets[key]
                    ratio = estimator.fraction(mask, boxes[key], frame.shape)
                    t['foreground_ratio'][i] = ratio
                    if ratio is not None:
                        # Stationary people remain eligible; foreground is weak evidence.
                        t['feature_quality'][i] *= 0.85 + 0.15 * ratio
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--scene-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=900)
    p.add_argument('--mode', choices=['mog2', 'median_static'], default='mog2')
    a = p.parse_args()
    rows = json.loads((a.run_dir/'local_tracklets.json').read_text())
    output = annotate(rows, a.run_dir/'observations.csv', a.scene_dir, a.start_frame, a.max_frames, a.mode)
    a.output.write_text(json.dumps(output), encoding='utf-8')


if __name__ == '__main__':
    main()
