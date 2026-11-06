"""Optional, auditable post-tracking repair of duplicate and fragmented tracklets.

Never edits the cached observations or embeddings. Suppressed keys are excluded
only from exported predictions; original data can be reused with repair off.
This is an experimental, conservative heuristic, not a trained MOT algorithm.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .association import Config, groups_compatible, frames, unit

Key = tuple[int, int]


@dataclass(frozen=True)
class RepairConfig:
    duplicate_max_frames: int = 3
    duplicate_min_long_frames: int = 10
    duplicate_min_iou: float = 0.45
    duplicate_max_distance_m: float = 0.50
    duplicate_min_similarity: float = 0.72
    max_gap_frames: int = 30
    # The lower threshold is allowed ONLY with independent cross-camera
    # geometric corroboration, never merely with close image coordinates.
    min_witnessed_similarity: float = 0.72
    min_unwitnessed_similarity: float = 0.92
    max_witness_distance_m: float = 0.75
    max_speed_mps: float = 3.5
    position_slack_m: float = 0.80
    max_pixel_displacement_heights: float = 0.90
    # A very short duplicate can have severely corrupted OSNet features.
    # Only apply this fallback with floor geometry and strong bbox overlap.
    duplicate_low_quality_max_frames: int = 2
    duplicate_low_quality_max_quality: float = 0.20
    duplicate_low_quality_min_iou: float = 0.47
    duplicate_low_quality_max_distance_m: float = 0.35
    duplicate_low_quality_min_similarity: float = 0.55
    duplicate_low_quality_min_long_frames: int = 30
    # Lower-ReID rescue is allowed ONLY with unique cross-camera geometry.
    strong_witness_min_similarity: float = 0.65
    strong_witness_max_distance_m: float = 0.65
    strong_witness_min_margin_m: float = 1.5
    strong_witness_max_gap_frames: int = 15
    strong_witness_max_boundary_distance_m: float = 1.50
    strong_witness_max_pixel_displacement_heights: float = 0.60


def read_observations(path: str | Path) -> dict[Key, dict[int, dict]]:
    result: dict[Key, dict[int, dict]] = defaultdict(dict)
    with open(path, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            key = (int(row["camera_id"]), int(row["local_id"]))
            frame = int(row["frame_id"])
            if frame in result[key]:
                raise ValueError(f"Duplicate observation for tracklet {key} frame {frame}")
            result[key][frame] = row
    return result


def _bbox(row: dict) -> tuple[float, float, float, float]:
    return tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))


def _iou(row_a: dict, row_b: dict) -> float:
    x1, y1, x2, y2 = _bbox(row_a)
    u1, v1, u2, v2 = _bbox(row_b)
    w = max(0., min(x2, u2) - max(x1, u1))
    h = max(0., min(y2, v2) - max(y1, v1))
    union = (x2-x1)*(y2-y1) + (u2-u1)*(v2-v1) - w*h
    return w*h/union if union > 0 else 0.


def _world(row: dict) -> np.ndarray | None:
    if row.get("world_x", "") in (None, "") or row.get("world_y", "") in (None, ""):
        return None
    point = np.asarray([float(row["world_x"]), float(row["world_y"])])
    return point if np.isfinite(point).all() else None


def _similarity(left: dict, right: dict) -> float | None:
    a, b = unit(left.get("embedding", [])), unit(right.get("embedding", []))
    if not a.size or a.shape != b.shape or np.linalg.norm(a) < 1e-10 or np.linalg.norm(b) < 1e-10:
        return None
    return float(np.clip(a @ b, -1., 1.))


def find_duplicate_tracklets(tracklets: list[dict], observations: dict[Key, dict[int, dict]],
                             settings: RepairConfig = RepairConfig(), *, use_geometry: bool = True):
    """Suppress ONLY a very short duplicate fully shadowed by a long-lived track.

    Two overlapping people should not be suppressed: require matching ReID,
    repeated substantial box overlap, and (when known) nearby floor points.
    """
    by_key = {(int(t["camera_id"]), int(t["local_id"])): t for t in tracklets}
    if len(by_key) != len(tracklets):
        raise ValueError("Duplicate (camera_id, local_id) among cached tracklets")
    suppressed: set[Key] = set()
    log = []
    candidates = []
    for short_key, short in observations.items():
        if short_key not in by_key or not 2 <= len(short) <= settings.duplicate_max_frames:
            continue
        for long_key, long in observations.items():
            if (long_key == short_key or long_key[0] != short_key[0] or
                    long_key not in by_key or len(long) < settings.duplicate_min_long_frames or
                    len(long) <= 3 * len(short) or not set(short).issubset(long)):
                continue
            overlaps = [_iou(short[f], long[f]) for f in short]
            if min(overlaps) < settings.duplicate_min_iou * 0.85 or float(np.mean(overlaps)) < settings.duplicate_min_iou:
                continue
            distances = []
            for frame in short if use_geometry else []:
                p, q = _world(short[frame]), _world(long[frame])
                if p is not None and q is not None:
                    distances.append(float(np.linalg.norm(p-q)))
            if distances and (len(distances) != len(short) or max(distances) > settings.duplicate_max_distance_m):
                continue
            similarity = _similarity(by_key[short_key], by_key[long_key])
            # Without floor coordinates, use stronger box and appearance gates.
            if similarity is None:
                continue
            mean_iou = float(np.mean(overlaps))
            quality = [float(x) for x in by_key[short_key].get("feature_quality", [])]
            # Low-quality embeddings are unreliable, so treat them as weaker
            # evidence only when very short boxes co-locate across all frames.
            low_quality_duplicate = (
                len(short) <= settings.duplicate_low_quality_max_frames
                and len(long) >= settings.duplicate_low_quality_min_long_frames
                and quality and max(quality) <= settings.duplicate_low_quality_max_quality
                and distances and max(distances) <= settings.duplicate_low_quality_max_distance_m
                and min(overlaps) >= settings.duplicate_low_quality_min_iou
                and similarity >= settings.duplicate_low_quality_min_similarity
            )
            normal_duplicate = (
                similarity >= (settings.duplicate_min_similarity if distances else 0.90)
                and (bool(distances) or min(overlaps) >= 0.70)
            )
            if not normal_duplicate and not low_quality_duplicate:
                continue
            candidates.append((mean_iou, similarity, short_key, long_key,
                               "short_low_quality_duplicate" if low_quality_duplicate and not normal_duplicate
                               else "short_duplicate_of_long_track"))
    # Stable winner if there are multiple possible long-lived tracks.
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    for overlap, similarity, short_key, long_key, reason in candidates:
        if short_key in suppressed or long_key in suppressed:
            continue
        suppressed.add(short_key)
        log.append({"suppressed": list(short_key), "kept": list(long_key),
                    "mean_iou": round(overlap, 4), "cosine_similarity": round(similarity, 4),
                    "reason": reason})
    remaining = [t for t in tracklets if (int(t["camera_id"]), int(t["local_id"])) not in suppressed]
    return remaining, suppressed, log


def _boundary_compatible(earlier: dict, later: dict,
                         observations: dict[Key, dict[int, dict]], settings: RepairConfig,
                         fps: float, use_geometry: bool = True) -> tuple[bool, str, float | None]:
    gap = int(later["start_frame"]) - int(earlier["end_frame"])
    if gap < 1 or gap > settings.max_gap_frames:
        return False, "gap_outside_window", None
    cam = int(earlier["camera_id"])
    first_key = (cam, int(earlier["local_id"]))
    second_key = (cam, int(later["local_id"]))
    before = observations.get(first_key, {})
    after = observations.get(second_key, {})
    if not before or not after:
        return False, "missing_boundary_observations", None
    a = before[max(before)]; b = after[min(after)]
    p, q = (_world(a), _world(b)) if use_geometry else (None, None)
    if p is not None and q is not None:
        distance = float(np.linalg.norm(p-q))
        limit = settings.max_speed_mps * (gap / fps) + settings.position_slack_m
        return distance <= limit, "world_distance", distance
    x1,y1,x2,y2 = _bbox(a)
    u1,v1,u2,v2 = _bbox(b)
    # Compare bbox bottom centres in the SAME camera (no camera calibration needed).
    distance = float(np.hypot((x1+x2-u1-u2)/2, y2-v2))
    height = max(y2-y1, v2-v1, 1.)
    return distance <= settings.max_pixel_displacement_heights*height, "pixel_distance", distance/height


def _cross_camera_witness(older_group: set[Key], newcomer: dict,
                          by_key: dict[Key, dict], settings: RepairConfig,
                          cfg: Config) -> float | None:
    """Independent corroboration by ANOTHER camera at the newcomer's start."""
    if not cfg.use_geometry:
        return None
    newcomer_cam = int(newcomer["camera_id"])
    offsets = cfg.camera_offsets_s or {}
    positions = frames(newcomer)
    if not positions:
        return None
    first_frames = sorted(positions)[:5]
    best = None
    for key in sorted(older_group):
        if key[0] == newcomer_cam:
            continue
        other = frames(by_key[key])
        # Compare synchronized times, not raw frame indices, when cameras
        # have a measured clock offset.
        shift = round(cfg.fps * (offsets.get(newcomer_cam, 0.0) - offsets.get(key[0], 0.0)))
        distances = [float(np.linalg.norm(np.subtract(positions[f], other[f + shift])))
                     for f in first_frames if f + shift in other]
        # Two observations are safer than a chance intersection at one frame.
        if len(distances) >= 2:
            candidate = float(np.median(distances))
            best = candidate if best is None else min(best, candidate)
    return best if best is not None and best <= settings.max_witness_distance_m else None


def _unique_cross_camera_witness(older_group: set[Key], newcomer: dict,
                                 by_key: dict[Key, dict], cfg: Config,
                                 minimum_observations: int = 3) -> tuple[float | None, float | None]:
    """(Best witness distance, separation from nearest alternative track).

    Compare all other-camera tracks, not just the selected global group.
    This guards against merging a nearby second person in crowded scenes.
    Requires synchronized projected positions; no geometry => no rescue.
    """
    if not cfg.use_geometry:
        return None, None
    newcomer_cam = int(newcomer["camera_id"])
    positions = frames(newcomer)
    first_frames = sorted(positions)[:5]
    if len(first_frames) < minimum_observations:
        return None, None
    offsets = cfg.camera_offsets_s or {}
    positive, competitors = [], []
    for key, candidate in by_key.items():
        if key[0] == newcomer_cam:
            continue
        other = frames(candidate)
        shift = round(cfg.fps * (offsets.get(newcomer_cam, 0.0) - offsets.get(key[0], 0.0)))
        distances = [float(np.linalg.norm(np.subtract(positions[f], other[f + shift])))
                     for f in first_frames if f + shift in other]
        if len(distances) < minimum_observations:
            continue
        d = float(np.median(distances))
        (positive if key in older_group else competitors).append(d)
    if not positive:
        return None, None
    closest = min(positive)
    margin = min(competitors) - closest if competitors else float("inf")
    return closest, margin


def relink_after_occlusion(tracklets: list[dict], mapping: dict[Key, int], cfg: Config,
                           observations: dict[Key, dict[int, dict]],
                           settings: RepairConfig = RepairConfig()):
    """Repair short gaps only when motion + appearance + group consistency agree.

    A weaker appearance gate requires a DIFFERENT camera to corroborate the
    same floor position. Otherwise a strict 0.92 cosine gate is applied.
    """
    result = dict(mapping)
    by_key = {(int(t["camera_id"]), int(t["local_id"])): t for t in tracklets}
    candidates = []
    for i, left in enumerate(tracklets):
        for right in tracklets[i+1:]:
            if int(left["camera_id"]) != int(right["camera_id"]):
                continue
            earlier, later = (left, right) if int(left["end_frame"]) < int(right["start_frame"]) else (right, left)
            if int(earlier["end_frame"]) >= int(later["start_frame"]):
                continue
            sim = _similarity(earlier, later)
            if sim is None or sim < min(settings.min_witnessed_similarity,
                                        settings.strong_witness_min_similarity):
                continue
            spatial_ok, kind, distance = _boundary_compatible(earlier, later, observations, settings, cfg.fps, cfg.use_geometry)
            if not spatial_ok:
                continue
            a = (int(earlier["camera_id"]), int(earlier["local_id"]))
            b = (int(later["camera_id"]), int(later["local_id"]))
            candidates.append((sim, a, b, kind, distance))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    journal = []
    for sim, a, b, kind, distance in candidates:
        if result[a] == result[b]:
            continue
        old_group = {key for key, gid in result.items() if gid == result[a]}
        new_group = {key for key, gid in result.items() if gid == result[b]}
        older, later = by_key[a], by_key[b]
        witness = _cross_camera_witness(old_group, later, by_key, settings, cfg)
        strong_distance, separation = _unique_cross_camera_witness(old_group, later, by_key, cfg)
        strong_witness = (
            strong_distance is not None
            and strong_distance <= settings.strong_witness_max_distance_m
            and separation is not None
            and separation >= settings.strong_witness_min_margin_m
            and int(later["start_frame"]) - int(older["end_frame"]) <= settings.strong_witness_max_gap_frames
            and distance is not None
            and ((kind == "world_distance" and distance <= settings.strong_witness_max_boundary_distance_m)
                 or (kind == "pixel_distance" and distance <= settings.strong_witness_max_pixel_displacement_heights))
        )
        if sim < settings.min_witnessed_similarity and not strong_witness:
            journal.append({"older": list(a), "later": list(b), "decision": "rejected",
                            "reason": "insufficient_unique_witness", "similarity": round(sim, 4),
                            "witness_distance_m": round(strong_distance, 4) if strong_distance is not None else None,
                            "witness_margin_m": round(separation, 4) if separation is not None and np.isfinite(separation) else None})
            continue
        if sim >= settings.min_witnessed_similarity and sim < settings.min_unwitnessed_similarity and witness is None:
            journal.append({"older": list(a), "later": list(b), "decision": "rejected",
                            "reason": "no_independent_witness", "similarity": round(sim, 4)})
            continue
        # Check every pair, not just the proposed edge, to avoid transitive
        # same-camera conflicts and physically impossible camera transitions.
        if not groups_compatible([by_key[x] for x in old_group], [by_key[y] for y in new_group], cfg):
            journal.append({"older": list(a), "later": list(b), "decision": "rejected",
                            "reason": "component_conflict", "similarity": round(sim, 4)})
            continue
        previous_id, merged_id = result[a], result[b]
        for key in new_group:
            result[key] = previous_id
        journal.append({"older": list(a), "later": list(b), "decision": "accepted",
                        "old_global_id": previous_id, "merged_global_id": merged_id,
                        "similarity": round(sim, 4), "boundary_type": kind,
                        "boundary_distance": round(distance, 4) if distance is not None else None,
                        "witness_distance_m": round(strong_distance if strong_witness else witness, 4)
                                              if (strong_witness or witness is not None) else None,
                        "witness_margin_m": round(separation, 4) if strong_witness and np.isfinite(separation) else None,
                        "rescue": "unique_cross_camera_geometry" if strong_witness and sim < settings.min_witnessed_similarity else "normal"})
    return result, journal
