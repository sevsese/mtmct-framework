"""Streaming NVIDIA GT validation and scene inspection (analysis only)."""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from pathlib import Path
from typing import Any

from .geometry import camera_number
from .gt import value


@dataclass(frozen=True)
class CameraTiming:
    fps: float = 30.0
    frame_index_offset: int = 0
    time_offset_s: float = 0.0
    pts_s: dict[int, float] | None = None

    def timestamp_s(self, frame_index: int) -> float:
        if self.pts_s and frame_index in self.pts_s:
            return float(self.pts_s[frame_index])
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        return self.time_offset_s + (frame_index - self.frame_index_offset) / self.fps


@dataclass(frozen=True)
class NvidiaGTObservation:
    frame_index: int
    timestamp_s: float
    camera_id: int
    bbox_xyxy: tuple[float, float, float, float]
    gt_object_id: int


class NvidiaDatasetAdapter:
    """Read NVIDIA ground truth without loading the full JSON into memory.

    Bad individual annotations are skipped by observations(strict=False),
    which is the default for batch experiments. Use strict=True to fail fast.
    The original ground-truth file is never modified.
    """

    def __init__(self, gt_path: str | Path, timing: CameraTiming | None = None):
        self.gt_path = Path(gt_path)
        self.timing = timing or CameraTiming()
        self.observation_stats: dict[str, Any] = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "total_bboxes": 0,
            "valid_bboxes": 0,
            "invalid_bboxes": 0,
            "invalid_bboxes_by_camera": {},
            "invalid_bbox_examples": [],
            "invalid_object_ids": 0,
            "invalid_camera_ids": 0,
        }

    def frames(self):
        import ijson

        with self.gt_path.open("rb") as stream:
            for raw_frame, objects in ijson.kvitems(stream, ""):
                try:
                    frame = int(raw_frame)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid frame index {raw_frame!r}") from exc
                if not isinstance(objects, list):
                    raise ValueError(f"Frame {frame} must contain a list of objects")
                yield frame, objects

    @staticmethod
    def _bbox(raw, frame: int, camera: str) -> tuple[float, float, float, float]:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValueError(f"Frame {frame}, {camera}: bbox must contain four values")
        try:
            box = tuple(float(x) for x in raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Frame {frame}, {camera}: non-numeric bbox {raw!r}") from exc
        if not all(isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Frame {frame}, {camera}: invalid bbox {raw!r}")
        return box

    def validate(self) -> list[str]:
        """List GT problems without changing or silently fixing annotations."""
        errors: list[str] = []
        for frame, objects in self.frames():
            seen: set[int] = set()
            for obj in objects:
                if str(value(obj, "object_type", "")).lower() not in {"person", "people"}:
                    continue
                raw_id = value(obj, "object_id")
                if raw_id is None:
                    errors.append(f"Frame {frame}: Person has no object_id")
                    continue
                try:
                    ident = int(raw_id)
                except (TypeError, ValueError, OverflowError):
                    errors.append(f"Frame {frame}: invalid object_id {raw_id!r}")
                    continue
                if ident in seen:
                    errors.append(f"Frame {frame}: duplicate object_id {ident}")
                seen.add(ident)
                boxes = value(obj, "2d_bounding_box_visible")
                if not isinstance(boxes, dict) or not boxes:
                    errors.append(f"Frame {frame}, object {ident}: missing visible bbox")
                    continue
                for camera, raw_box in boxes.items():
                    try:
                        camera_number(camera)
                        self._bbox(raw_box, frame, str(camera))
                    except (TypeError, ValueError, OverflowError) as exc:
                        errors.append(str(exc))
        return errors

    def observations(self, strict: bool = False, max_frames: int | None = None):
        """Yield valid observations, optionally failing fast on bad entries.

        strict=False is intentional: a zero-area bounding box is a bad GT
        annotation, not a reason to abort a whole optimization run.
        Check observation_stats after consuming the iterator.
        """
        stats = self._empty_stats()
        self.observation_stats = stats

        for frame, objects in self.frames():
            if max_frames is not None and frame >= max_frames:
                break
            for obj in objects:
                if str(value(obj, "object_type", "")).lower() not in {"person", "people"}:
                    continue
                raw_id = value(obj, "object_id")
                boxes = value(obj, "2d_bounding_box_visible")
                if raw_id is None or not isinstance(boxes, dict):
                    continue

                try:
                    gt_object_id = int(raw_id)
                except (TypeError, ValueError, OverflowError) as exc:
                    if strict:
                        raise ValueError(f"Frame {frame}: invalid object_id {raw_id!r}") from exc
                    stats["invalid_object_ids"] += 1
                    continue

                for camera, raw_box in boxes.items():
                    try:
                        camera_id = camera_number(camera)
                    except (TypeError, ValueError, OverflowError):
                        if strict:
                            raise
                        stats["invalid_camera_ids"] += 1
                        continue

                    stats["total_bboxes"] += 1
                    try:
                        bbox = self._bbox(raw_box, frame, str(camera))
                    except (TypeError, ValueError, OverflowError) as exc:
                        if strict:
                            raise
                        stats["invalid_bboxes"] += 1
                        counts = stats["invalid_bboxes_by_camera"]
                        key = str(camera)
                        counts[key] = counts.get(key, 0) + 1
                        if len(stats["invalid_bbox_examples"]) < 10:
                            stats["invalid_bbox_examples"].append(str(exc))
                        continue

                    stats["valid_bboxes"] += 1
                    yield NvidiaGTObservation(
                        frame_index=frame,
                        timestamp_s=self.timing.timestamp_s(frame),
                        camera_id=camera_id,
                        bbox_xyxy=bbox,
                        gt_object_id=gt_object_id,
                    )

        if not strict and stats["invalid_bboxes"]:
            bad, total = stats["invalid_bboxes"], stats["total_bboxes"]
            fraction = bad / total if total else 0.0
            severity = RuntimeWarning if fraction > 0.05 else UserWarning
            warnings.warn(
                f"{self.gt_path}: skipped {bad}/{total} invalid GT bboxes "
                f"({fraction:.2%}); examples: {stats['invalid_bbox_examples'][:3]}",
                severity,
                stacklevel=2,
            )


def inspect_scene(scene_dir: str | Path, output: str | Path, timing: CameraTiming | None = None):
    scene_dir, output = Path(scene_dir), Path(output)
    adapter = NvidiaDatasetAdapter(scene_dir / "ground_truth.json", timing)
    errors = adapter.validate()

    camera_ids: dict[int, set[int]] = defaultdict(set)
    camera_frames: dict[int, set[int]] = defaultdict(set)
    camera_observations: dict[int, int] = defaultdict(int)
    camera_span: dict[int, list[float]] = {}
    frame_people: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    people: set[int] = set()
    frames: set[int] = set()
    for ob in adapter.observations(strict=False):
        frames.add(ob.frame_index)
        people.add(ob.gt_object_id)
        camera_ids[ob.camera_id].add(ob.gt_object_id)
        camera_frames[ob.camera_id].add(ob.frame_index)
        camera_observations[ob.camera_id] += 1
        frame_people[ob.frame_index][ob.camera_id].add(ob.gt_object_id)
        span = camera_span.setdefault(ob.camera_id, [ob.timestamp_s, ob.timestamp_s])
        span[0], span[1] = min(span[0], ob.timestamp_s), max(span[1], ob.timestamp_s)

    cameras = sorted(camera_ids)
    pair_rows = []
    for a, b in combinations(cameras, 2):
        common_frames = camera_frames[a] & camera_frames[b]
        positives = 0
        negatives = 0
        for frame in common_frames:
            left, right = frame_people[frame][a], frame_people[frame][b]
            shared = len(left & right)
            positives += shared
            negatives += len(left) * len(right) - shared
        pair_rows.append({
            "camera_a": a,
            "camera_b": b,
            "common_ids": len(camera_ids[a] & camera_ids[b]),
            "co_visible_frames": len(common_frames),
            "positive_pairs": positives,
            "negative_pairs": negatives,
        })
    pair_rows.sort(key=lambda row: (-row["positive_pairs"], -row["common_ids"], row["camera_a"], row["camera_b"]))

    local_videos = {camera_number(path.stem) for path in (scene_dir / "videos").glob("Camera_*.mp4")}
    report = {
        "schema_version": "nvidia-scene-report-v1",
        "scene": scene_dir.name,
        "gt_path": str(adapter.gt_path),
        "timestamp_formula": "PTS[frame] if available, else time_offset_s + (frame_index - frame_index_offset) / fps",
        "fps": adapter.timing.fps,
        "frame_index_offset": adapter.timing.frame_index_offset,
        "time_offset_s": adapter.timing.time_offset_s,
        "validation": {"issue_count": len(errors), "samples": errors[:10]},
        "observation_filtering": adapter.observation_stats,
        "frame_count": len(frames),
        "first_frame": min(frames) if frames else None,
        "last_frame": max(frames) if frames else None,
        "unique_people": len(people),
        "cameras": [{
            "camera_id": camera,
            "observations": camera_observations[camera],
            "frames": len(camera_frames[camera]),
            "unique_people": len(camera_ids[camera]),
            "start_s": camera_span[camera][0],
            "end_s": camera_span[camera][1],
        } for camera in cameras],
        "video_validation": {
            "missing_videos": sorted(set(cameras) - local_videos),
            "videos_without_gt": sorted(local_videos - set(cameras)),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "scene_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "camera_pair_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "camera_a", "camera_b", "common_ids", "co_visible_frames", "positive_pairs", "negative_pairs"
        ])
        writer.writeheader()
        writer.writerows(pair_rows)
    return report, pair_rows


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Validate and inspect an NVIDIA MTMCT scene")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--time-offset", type=float, default=0.0)
    args = parser.parse_args(argv)
    report, pairs = inspect_scene(
        args.scene_dir,
        args.out,
        CameraTiming(args.fps, args.frame_offset, args.time_offset),
    )
    print(json.dumps(report, indent=2))
    print(f"Camera pairs: {len(pairs)} -> {args.out / 'camera_pair_report.csv'}")


if __name__ == "__main__":
    main()
