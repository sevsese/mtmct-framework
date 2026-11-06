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
    def __init__(self, routes: dict[tuple[int, int], dict] | None = None):
        self.routes = routes or {}

    @classmethod
    def from_json(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        routes = {}
        for row in data.get("routes", []):
            routes[(int(row["from"]), int(row["to"]))] = row
        return cls(routes)

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
        midpoint = (minimum + maximum) / 2
        penalty = min(abs(gap_s - midpoint) / ((maximum - minimum) / 2), 1.0)
        return TransitionResult(True, penalty, "measured")


def parse_offsets(values: list[str] | None) -> dict[int, float]:
    offsets = {}
    for value in values or []:
        camera, seconds = value.split("=", 1)
        offsets[int(camera)] = float(seconds)
    return offsets


def track_times(tracklet: dict, fps: float, offsets: dict[int, float] | None = None):
    offset = (offsets or {}).get(int(tracklet["camera_id"]), 0.0)
    if "start_s" in tracklet and "end_s" in tracklet:
        return float(tracklet["start_s"]) + offset, float(tracklet["end_s"]) + offset
    fps = float(tracklet.get("fps", fps))
    return (float(tracklet["start_frame"]) / fps + offset,
            float(tracklet["end_frame"]) / fps + offset)
