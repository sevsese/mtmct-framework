"""Lightweight frame sources for both MP4 cameras and SCOUT image sequences.

Preserves original 0-based frame numbers. No video re-encoding is necessary.
"""
from __future__ import annotations
from pathlib import Path
import re
import cv2
import numpy as np

_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def _read_image(path: Path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def image_number(path: Path) -> int:
    m = re.search(r'(\d+)$', path.stem)
    if not m:
        raise ValueError(f'Image frame filename must end in a frame number: {path.name}')
    return int(m.group(1))


class FrameSource:
    def __init__(self, path: str | Path, fps: float = 30.0):
        self.path = Path(path)
        self.video = None
        self.fps = float(fps)
        if self.path.is_dir():
            self.images = {}
            for file in self.path.iterdir():
                if file.suffix.lower() in _EXTENSIONS:
                    num = image_number(file)
                    if num in self.images:
                        raise ValueError(f'Duplicate frame {num}: {file}')
                    self.images[num] = file
            if not self.images:
                raise FileNotFoundError(f'No images in {self.path}')
        elif self.path.is_file():
            self.images = None
            self.video = cv2.VideoCapture(str(self.path))
            if not self.video.isOpened():
                raise RuntimeError(f'Cannot open video: {self.path}')
            measured = self.video.get(cv2.CAP_PROP_FPS)
            if measured and measured > 0:
                self.fps = measured
        else:
            raise FileNotFoundError(f'Frame source not found: {self.path}')
        self._cursor = 0

    def get(self, frame_id: int):
        if self.images is not None:
            file = self.images.get(frame_id)
            return _read_image(file) if file else None
        if frame_id < self._cursor:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            self._cursor = frame_id
        while self._cursor < frame_id:
            ok, _ = self.video.read()
            if not ok:
                return None
            self._cursor += 1
        ok, frame = self.video.read()
        if ok:
            self._cursor += 1
        return frame if ok else None

    def iter_frames(self, max_frames: int, stride: int = 1,
                    frame_ids: set[int] | None = None):
        """Yield frames without renumbering image sequences.

        ``frame_ids`` is for sparse annotated datasets such as WILDTRACK.  It
        selects real filenames, rather than treating their numeric suffixes as
        contiguous video indices.
        """
        if stride < 1:
            raise ValueError('stride must be >= 1')
        if frame_ids is not None and max_frames > 0:
            raise ValueError('Use max_frames=0 when selecting explicit frame_ids')
        if self.images is not None:
            for fid in sorted(self.images):
                if (max_frames > 0 and fid >= max_frames) or fid % stride != 0:
                    continue
                if frame_ids is not None and fid not in frame_ids:
                    continue
                frame = self.get(fid)
                if frame is None:
                    raise RuntimeError(f'Unreadable image: {self.images[fid]}')
                yield fid, frame
            return
        fid = 0
        while max_frames <= 0 or fid < max_frames:
            frame = self.get(fid)
            if frame is None:
                break
            if fid % stride == 0:
                yield fid, frame
            fid += 1

    def dimensions(self):
        if self.images is None:
            return int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = _read_image(self.images[min(self.images)])
        if f is None:
            raise RuntimeError('Cannot read first image')
        return f.shape[1], f.shape[0]

    def close(self):
        if self.video is not None:
            self.video.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
