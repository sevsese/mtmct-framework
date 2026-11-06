"""Dataset-neutral access to MTMCT video/frame inputs and ground truth.

No synthetic bbox fabrication: CITIC ground truth is world positions, not
per-camera bounding boxes. SCOUT has bboxes but annotation quality varies.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import csv
import json
import re

from .frame_sources import image_number

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv'}


@dataclass(frozen=True)
class GTObservation:
    frame_id: int
    camera_id: int | None
    identity: int
    bbox_xyxy: tuple[float, float, float, float] | None = None
    world_xy: tuple[float, float] | None = None
    timestamp_s: float | None = None


class MTMCTDataset(ABC):
    kind: str
    fps: float

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @abstractmethod
    def list_scenes(self) -> list[str]: ...

    @abstractmethod
    def list_cameras(self, scene: str) -> list[int]: ...

    @abstractmethod
    def get_video(self, scene: str, camera: int) -> Path:
        """An MP4/AVI file, or a folder containing image_N.jpg frames."""
        ...

    @abstractmethod
    def get_ground_truth(self, scene: str) -> Path | None: ...

    @abstractmethod
    def get_calibration(self, scene: str, camera: int | None = None) -> Path | None: ...

    def get_timestamps(self, scene: str, camera: int) -> Path | None:
        return None

    @abstractmethod
    def observations(self, scene: str, max_frames: int = 0, cameras: list[int] | None = None) -> Iterator[GTObservation]: ...

    def frame_sources(self, scene: str, cameras: list[int]) -> dict[int, Path]:
        return {int(camera): self.get_video(scene, int(camera)) for camera in cameras}

    def supports_bbox_evaluation(self, scene: str, cameras: list[int]) -> bool:
        """Explicit capability flag: never confuse world GT with 2D bbox GT."""
        return False


class NvidiaDataset(MTMCTDataset):
    kind = 'nvidia'
    fps = 30.0

    def __init__(self, root: str | Path, split: str = 'train'):
        super().__init__(root)
        self.split = split

    def _dir(self, scene: str) -> Path:
        p = self.root / self.split / scene
        if not p.exists() and (self.root / scene).is_dir():
            p = self.root / scene
        return p

    def list_scenes(self) -> list[str]:
        directory = self.root / self.split
        return sorted(p.name for p in directory.iterdir() if p.is_dir()) if directory.is_dir() else []

    def list_cameras(self, scene: str) -> list[int]:
        return sorted(int(p.stem.split('_')[-1]) for p in (self._dir(scene) / 'videos').glob('Camera_*.mp4'))

    def get_video(self, scene: str, camera: int) -> Path:
        path = self._dir(scene) / 'videos' / f'Camera_{camera:04d}.mp4'
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def get_ground_truth(self, scene: str) -> Path | None:
        path = self._dir(scene) / 'ground_truth.json'
        return path if path.is_file() else None

    def get_calibration(self, scene: str, camera: int | None = None) -> Path | None:
        path = self._dir(scene) / 'calibration.json'
        return path if path.is_file() else None

    def supports_bbox_evaluation(self, scene: str, cameras: list[int]) -> bool:
        return self.get_ground_truth(scene) is not None

    def observations(self, scene: str, max_frames: int = 0, cameras: list[int] | None = None):
        from .nvidia import NvidiaDatasetAdapter
        path = self.get_ground_truth(scene)
        if not path:
            return
        allowed = set(cameras) if cameras is not None else None
        for obs in NvidiaDatasetAdapter(path).observations(max_frames=max_frames if max_frames > 0 else None, strict=False):
            if allowed is None or obs.camera_id in allowed:
                yield GTObservation(obs.frame_index, obs.camera_id, obs.gt_object_id,
                                    tuple(obs.bbox_xyxy), timestamp_s=obs.timestamp_s)


class CITICDataset(MTMCTDataset):
    kind = 'citic'
    fps = 25.0
    # Stable internal numeric IDs. Names correspond to official camera names.
    camera_names = {0: 'HallWide', 1: 'HallSeg', 2: 'PF', 3: 'PC', 4: 'Showroom'}

    def __init__(self, root: str | Path, video_map: dict[int, str | Path] | None = None):
        super().__init__(root)
        self.video_map = {int(k): Path(v) for k, v in (video_map or {}).items()}

    def list_scenes(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and re.fullmatch(r'capture\d+', p.name, re.I)) if self.root.is_dir() else []

    def _video_map(self, scene: str) -> dict[int, Path]:
        folder = self.root / scene
        if not folder.is_dir():
            raise FileNotFoundError(f'CITIC capture folder not found: {folder}')
        found = {}
        for cam, path in self.video_map.items():
            resolved = path if path.is_absolute() else self.root / path
            if resolved.is_file():
                found[cam] = resolved
            else:
                raise FileNotFoundError(f'Configured CITIC camera file missing: {resolved}')
        for p in folder.rglob('*'):
            if p.suffix.lower() not in VIDEO_EXTS or not p.is_file():
                continue
            # A directory with one video per named camera is supported.
            matches = [c for c, name in self.camera_names.items()
                       if re.search(r'(?<![a-z])' + re.escape(name.lower()) + r'(?![a-z])', p.stem.lower())
                       or re.search(r'(?<![a-z])' + re.escape(name.lower()) + r'(?![a-z])', p.parent.name.lower())]
            if len(matches) == 1 and matches[0] not in found:
                found[matches[0]] = p
        return found

    def list_cameras(self, scene: str) -> list[int]:
        return sorted(self._video_map(scene))

    def get_video(self, scene: str, camera: int) -> Path:
        result = self._video_map(scene)
        if camera not in result:
            raise FileNotFoundError(f'CITIC camera {camera} ({self.camera_names.get(camera)}) not found in {self.root / scene}. '
                                    f'Found: {result}. Supply VIDEO_MAP for nonstandard filenames.')
        return result[camera]

    def get_ground_truth(self, scene: str) -> Path | None:
        p = self.root / f'{scene}_gt.mot.txt'
        return p if p.is_file() else None

    def get_timestamps(self, scene: str, camera: int) -> Path | None:
        # Raw synchronization metadata; not silently applied as frame offsets.
        path = self.root / 'timesyncs.csv'
        return path if path.is_file() else None

    def get_calibration(self, scene: str, camera: int | None = None) -> Path | None:
        if camera is None:
            return None  # Node JSON is not a NVIDIA cameraMatrix file.
        name = self.camera_names.get(camera)
        if not name:
            return None
        files = list((self.root / 'map_config').glob(f'Node*{name}*.json'))
        return files[0] if files else None

    def observations(self, scene: str, max_frames: int = 0, cameras: list[int] | None = None):
        """CITIC GT contains world paths. Its bbox values are -1 placeholders."""
        path = self.get_ground_truth(scene)
        if path is None:
            return
        with path.open(newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                if not row or row[0].startswith('#'):
                    continue
                if len(row) < 10:
                    raise ValueError(f'Malformed CITIC GT line: {row}')
                frame = int(float(row[0]))
                if max_frames > 0 and frame >= max_frames:
                    continue
                yield GTObservation(frame, None, int(float(row[1])),
                                    world_xy=(float(row[7]), float(row[8])), timestamp_s=frame / self.fps)


class SCOUTDataset(MTMCTDataset):
    kind = 'scout'
    fps = 10.0

    def __init__(self, root: str | Path, allow_pseudo_labels: bool = False):
        super().__init__(root)
        self.allow_pseudo_labels = allow_pseudo_labels

    def list_scenes(self) -> list[str]:
        base = self.root / 'images'
        return sorted(p.name for p in base.glob('sequence_*') if p.is_dir()) if base.is_dir() else []

    def _seq(self, scene: str) -> str:
        return scene if scene.startswith('sequence_') else f'sequence_{scene}'

    def list_cameras(self, scene: str) -> list[int]:
        directory = self.root / 'images' / self._seq(scene)
        return sorted(int(p.name.split('_')[-1]) for p in directory.glob('cam_*') if p.is_dir())

    def get_video(self, scene: str, camera: int) -> Path:
        path = self.root / 'images' / self._seq(scene) / f'cam_{camera}'
        if not path.is_dir() or not any(path.glob('image_*.jpg')):
            raise FileNotFoundError(f'SCOUT image sequence not found: {path}')
        return path

    def get_ground_truth(self, scene: str) -> Path | None:
        directory = self.root / 'annotations' / 'mot_format' / self._seq(scene)
        return directory if directory.is_dir() else None

    def get_calibration(self, scene: str, camera: int | None = None) -> Path | None:
        if camera is None:
            return None
        path = self.root / 'calibrations' / f'cam_{camera}.json'
        return path if path.is_file() else None

    def get_timestamps(self, scene: str, camera: int) -> Path | None:
        path = self.root / 'timestamps' / f'timestamps_{self._seq(scene)}.json'
        return path if path.is_file() else None

    def supports_bbox_evaluation(self, scene: str, cameras: list[int]) -> bool:
        directory = self.get_ground_truth(scene)
        if directory is None:
            return False
        # Official first release: first eight cameras of sequence 1 manually annotated.
        if self._seq(scene) != 'sequence_1' and not self.allow_pseudo_labels:
            return False
        if not self.allow_pseudo_labels and any(c >= 8 for c in cameras):
            return False
        return all((directory / f'cam_{c}.txt').is_file() for c in cameras)

    def observations(self, scene: str, max_frames: int = 0, cameras: list[int] | None = None):
        directory = self.get_ground_truth(scene)
        if directory is None:
            return
        selected = cameras if cameras is not None else self.list_cameras(scene)
        for cam in selected:
            if not self.allow_pseudo_labels and (self._seq(scene) != 'sequence_1' or cam >= 8):
                raise ValueError('SCOUT pseudolabel GT is disabled. Set ALLOW_SCOUT_PSEUDO_LABELS=True only for exploratory metrics.')
            p = directory / f'cam_{cam}.txt'
            if not p.is_file():
                raise FileNotFoundError(f'Missing SCOUT MOT file: {p}')
            with p.open(newline='', encoding='utf-8') as f:
                for row in csv.reader(f):
                    if not row or row[0].startswith('#'):
                        continue
                    if len(row) < 10:
                        raise ValueError(f'Malformed SCOUT MOT line: {row}')
                    fid, pid = int(float(row[0])), int(float(row[1]))
                    if max_frames > 0 and fid >= max_frames:
                        continue
                    x, y, w, h = map(float, row[2:6])
                    if w <= 0 or h <= 0:
                        continue
                    yield GTObservation(fid, cam, pid, (x, y, x + w, y + h),
                                        timestamp_s=fid / self.fps)


def open_dataset(kind: str, root: str | Path, *, split: str = 'train',
                 allow_pseudo_labels: bool = False,
                 video_map: dict[int, str | Path] | None = None) -> MTMCTDataset:
    kind = kind.lower()
    if kind == 'nvidia':
        return NvidiaDataset(root, split)
    if kind == 'citic':
        return CITICDataset(root, video_map=video_map)
    if kind == 'scout':
        return SCOUTDataset(root, allow_pseudo_labels=allow_pseudo_labels)
    raise ValueError(f'Unknown dataset {kind!r}: choose nvidia, citic, scout')
