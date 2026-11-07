"""Camera clock offsets and optional measured transition topology."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TransitionResult:
    allowed: bool
    penalty: float
    status: str


class CameraTransitionModel:
    def __init__(self, routes: dict[tuple[int, int], dict] | None = None,
                 overlap_pairs: set[frozenset[int]] | None = None,
                 disjoint_pairs: set[frozenset[int]] | None = None):
        self.routes = routes or {}
        self.overlap_pairs = overlap_pairs or set()
        self.disjoint_pairs = disjoint_pairs or set()

    @classmethod
    def from_json(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        routes = {}
        for row in data.get("routes", []):
            routes[(int(row["from"]), int(row["to"]))] = row
        overlap_pairs = {
            frozenset((int(pair[0]), int(pair[1])))
            for pair in data.get("overlap_pairs", [])
        }
        disjoint = {frozenset(map(int, pair)) for pair in data.get("disjoint_pairs", [])}
        if disjoint & overlap_pairs:
            raise ValueError("Camera pairs cannot be both overlapping and disjoint")
        return cls(routes, overlap_pairs, disjoint)

    def allows_overlap(self, camera_a: int, camera_b: int) -> bool:
        return frozenset((int(camera_a), int(camera_b))) in self.overlap_pairs

    def evaluate(self, source: int, destination: int, gap_s: float) -> TransitionResult:
        route = self.routes.get((int(source), int(destination)))
        if route is None or route.get("status", "unknown") == "unknown":
            return TransitionResult(True, 0.0, "unknown")
        if route.get("status") == "forbidden":
            return TransitionResult(False, 1.0, "forbidden")
        minimum = float(route.get("min_s", 0.0))
        maximum = float(route.get("max_s", float("inf")))
        if gap_s < minimum or gap_s > maximum:
            return TransitionResult(False, 1.0, "outside_measured_route")
        if maximum == float("inf") or maximum <= minimum:
            return TransitionResult(True, 0.0, "measured")
        # An allowed interval is not a learned travel-time distribution.
        # Do not invent a preference for its midpoint.
        return TransitionResult(True, 0.0, "measured")


def parse_offsets(values: list[str] | None) -> dict[int, float]:
    offsets = {}
    for value in values or []:
        camera, seconds = value.split("=", 1)
        offsets[int(camera)] = float(seconds)
    return offsets


def track_times(tracklet: dict, fps: float | dict[int, float],
                offsets: dict[int, float] | None = None):
    camera = int(tracklet["camera_id"])
    offset = (offsets or {}).get(camera, 0.0)
    if "start_s" in tracklet and "end_s" in tracklet:
        return float(tracklet["start_s"]) + offset, float(tracklet["end_s"]) + offset
    camera_fps = float(fps.get(camera, 0.0) if isinstance(fps, dict) else fps)
    if camera_fps <= 0:
        raise ValueError(f"fps must be positive for camera {camera}")
    return (float(tracklet["start_frame"]) / camera_fps + offset,
            float(tracklet["end_frame"]) / camera_fps + offset)


def observation_time(tracklet, frame, fps, offsets=None, starts=None):
    """Local PTS takes precedence over frame/FPS. Offsets are applied once."""
    camera = int(tracklet["camera_id"])
    timestamp = (tracklet.get("frame_timestamps") or {}).get(str(frame))
    if timestamp is None and frame == int(tracklet["end_frame"]):
        timestamp = tracklet.get("end_s")
    if timestamp is None and tracklet.get("start_s") is not None:
        timestamp = float(tracklet["start_s"]) + (frame - int(tracklet["start_frame"])) / fps
    if timestamp is None:
        timestamp = frame / fps
    return float(timestamp) + (starts or {}).get(camera, 0.) + (offsets or {}).get(camera, 0.)
