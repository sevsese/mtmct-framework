"""Mosaic visualization for videos or numbered image-frame directories."""
from __future__ import annotations
from collections import defaultdict
from contextlib import ExitStack
from math import ceil, sqrt
from itertools import count
from pathlib import Path
import argparse
import csv
import cv2
import numpy as np
from .geometry import camera_number
from .frame_sources import FrameSource


def id_color(gid: int) -> tuple[int, int, int]:
    return (int((37 * gid + 105) % 180 + 60),
            int((67 * gid + 35) % 180 + 60),
            int((101 * gid + 55) % 180 + 60))


def render(scene_dir: Path, predictions: Path, output: Path,
           max_frames: int = 300, stride: int = 3, scale: float = 0.5,
           cameras: list[int] | None = None,
           camera_sources: dict[int, Path] | None = None, fps_fallback: float = 30.0):
    if stride < 1 or scale <= 0:
        raise ValueError('stride >= 1 and scale > 0 required')
    if camera_sources is None:
        files = sorted((scene_dir / 'videos').glob('Camera_*.mp4'))
        sources = {camera_number(p.stem): p for p in files}
    else:
        sources = {int(c): Path(p) for c, p in camera_sources.items()}
    selected = sorted(set(cameras)) if cameras is not None else sorted(sources)
    if len(selected) < 2 or any(cam not in sources for cam in selected):
        raise ValueError('Mosaic requires at least two available cameras')
    tracks = defaultdict(list)
    with Path(predictions).open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            fid = int(row['frame_id'])
            if max_frames > 0 and fid >= max_frames:
                continue
            tracks[(int(row['camera_id']), fid)].append(row)
    with ExitStack() as stack:
        readers = {cam: stack.enter_context(FrameSource(sources[cam], fps_fallback))
                   for cam in selected}
        base_w, base_h = next(iter(readers.values())).dimensions()
        w, h = max(1, int(base_w * scale)), max(1, int(base_h * scale))
        cols = ceil(sqrt(len(selected)))
        rows = ceil(len(selected) / cols)
        source_fps = next(iter(readers.values())).fps
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'),
                                 max(1.0, source_fps / stride), (w * cols, h * rows))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot create mosaic video: {output}')
        try:
            if all(reader.images is not None for reader in readers.values()):
                all_ids = sorted(set().union(*(set(reader.images) for reader in readers.values())))
                timeline = (fid for fid in all_ids if max_frames <= 0 or fid < max_frames)
            else:
                timeline = range(max_frames) if max_frames > 0 else count()
            for frame_id in timeline:
                if frame_id % stride != 0:
                    continue
                images = []
                any_frame = False
                for cam, reader in readers.items():
                    frame = reader.get(frame_id)
                    if frame is not None:
                        any_frame = True
                        for r in tracks.get((cam, frame_id), []):
                            x1, y1, x2, y2 = (int(float(r[k])) for k in ('x1', 'y1', 'x2', 'y2'))
                            gid = int(r['global_id'])
                            color = id_color(gid)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(frame, f'GLOBAL {gid}', (x1, max(20, y1-8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2)
                        cv2.putText(frame, f'Camera {cam} frame {frame_id}', (20, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    else:
                        resized = np.zeros((h, w, 3), dtype=np.uint8)
                    images.append(resized)
                if not any_frame:
                    if all(reader.images is not None for reader in readers.values()):
                        continue
                    break
                while len(images) < cols * rows:
                    images.append(np.zeros((h, w, 3), dtype=np.uint8))
                tiles = [np.concatenate(images[i * cols:(i + 1) * cols], axis=1)
                         for i in range(rows)]
                writer.write(np.concatenate(tiles, axis=0))
        finally:
            writer.release()
    print('Created mosaic:', output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene-dir', type=Path, required=True)
    p.add_argument('--pred', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('runs/mosaic.mp4'))
    p.add_argument('--max-frames', type=int, default=300)
    p.add_argument('--stride', type=int, default=3)
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--cameras', type=int, nargs='+')
    a = p.parse_args()
    render(a.scene_dir, a.pred, a.out, a.max_frames, a.stride, a.scale, a.cameras)

if __name__ == '__main__':
    main()
