"""Human-readable global-track summaries for datasets without ground truth."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .time_sync import track_times


def build_global_track_report(tracklets: list[dict], mapping: dict[tuple[int, int], int],
                              fps: float | dict[int, float],
                              offsets: dict[int, float] | None = None) -> list[dict]:
    groups = defaultdict(list)
    for row in tracklets:
        key = (int(row["camera_id"]), int(row["local_id"]))
        if key not in mapping:
            continue  # Association may suppress unconfirmed short tracklets.
        start, end = track_times(row, fps, offsets)
        groups[int(mapping[key])].append({
            "camera_id": key[0], "local_id": key[1],
            "start_s": round(start, 3), "end_s": round(end, 3),
        })

    report = []
    for global_id, segments in sorted(groups.items()):
        segments.sort(key=lambda row: (row["start_s"], row["end_s"], row["camera_id"]))
        sequence = []
        for row in segments:
            if not sequence or sequence[-1] != row["camera_id"]:
                sequence.append(row["camera_id"])
        report.append({
            "global_id": global_id,
            "start_s": min(row["start_s"] for row in segments),
            "end_s": max(row["end_s"] for row in segments),
            "camera_sequence": sequence,
            "segments": segments,
        })
    return report


def write_global_track_report(path: str | Path, tracklets: list[dict],
                              mapping: dict[tuple[int, int], int],
                              fps: float | dict[int, float],
                              offsets: dict[int, float] | None = None) -> list[dict]:
    report = build_global_track_report(tracklets, mapping, fps, offsets)
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
