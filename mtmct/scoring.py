"""Separate proven conflicts from available soft evidence for MTMCT 2.0."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np

from .association import Config, unit
from .time_sync import observation_time


@dataclass
class Evidence:
    row: dict
    cfg: Config
    anchor_quality: float = .30

    @cached_property
    def key(self):
        return int(self.row["camera_id"]), int(self.row["local_id"])

    @cached_property
    def uid(self):
        return str(self.row.get("tracklet_uid") or
                   f"{self.key[0]}:{self.key[1]}:{self.row['start_frame']}")

    @cached_property
    def fps(self):
        fps = (self.cfg.camera_fps or {}).get(self.key[0]) or self.row.get("source_fps") or self.row.get("fps") or self.cfg.fps
        if not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError(f"Invalid FPS for tracklet {self.key}")
        return float(fps)

    def time(self, frame):
        return observation_time(self.row, frame, self.fps, self.cfg.camera_offsets_s,
                                self.cfg.camera_start_s)

    @cached_property
    def interval(self):
        start, end = (self.time(int(self.row[k])) for k in ("start_frame", "end_frame"))
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError(f"Invalid interval for tracklet {self.key}")
        return start, end

    @cached_property
    def frames(self):
        return frozenset(map(int, self.row.get("observation_frames") or ()))

    @cached_property
    def mean(self):
        vector = unit(self.row.get("embedding") or [])
        return vector if vector.ndim == 1 and np.isfinite(vector).all() and np.any(vector) else np.empty(0)

    @cached_property
    def prototypes(self):
        samples = []
        vectors = self.row.get("feature_vectors") or []
        qualities = self.row.get("feature_quality") or []
        frames = self.row.get("feature_frames") or []
        details = self.row.get("feature_quality_details") or []
        for i, (vector, quality, frame) in enumerate(zip(vectors, qualities, frames)):
            e = unit(vector)
            q = float(quality)
            if i < len(details):
                q *= 1 - .5 * details[i].get("occlusion_proxy", 0.)
            if e.ndim != 1 or not e.size or not np.isfinite(e).all() or not np.any(e) or not 0 < q <= 1:
                continue
            if samples and e.shape != samples[0][0].shape:
                continue
            samples.append((e, q, int(frame), i))
        if not samples:
            return []
        best = max(sample[1] for sample in samples)
        pool = [sample for sample in samples if sample[1] >= .6 * best]
        chosen = []
        while pool and len(chosen) < 4:
            def priority(sample):
                diversity = min(((1 - float(sample[0] @ old[0])) / 2 for old in chosen), default=0.)
                return .75 * sample[1] + .25 * diversity, -sample[3]
            chosen.append(pool.pop(max(range(len(pool)), key=lambda i: priority(pool[i]))))
        foreground = self.row.get("foreground_ratio") or []
        return [{"e": e.tolist(), "q": q, "camera_id": self.key[0], "time_s": self.time(frame),
                 "frame_id": frame, "source": f"{self.uid}:{frame}", "support": 1,
                 "quality_details": {**(details[i] if i < len(details) else {}),
                                     "foreground": foreground[i] if i < len(foreground) else None}}
                for e, q, frame, i in chosen]

    @cached_property
    def quality(self):
        if self.prototypes:
            return float(np.mean(sorted((p["q"] for p in self.prototypes), reverse=True)[:3]))
        # Legacy caches may have only one aggregate. It remains usable, with
        # lower confidence than a measured gallery.
        return .35 if self.mean.size else 0.

    @cached_property
    def anchor(self):
        count = len(self.frames) or int(self.row.get("samples", 0))
        if self.quality < self.anchor_quality or count < 5 or self.row.get("suspect_switch"):
            return False
        if len(self.prototypes) > 1 and self.mean.size:
            return all(len(p["e"]) == len(self.mean) and np.dot(p["e"], self.mean) >= .55
                       for p in self.prototypes)
        return bool(self.mean.size or self.prototypes)

    @cached_property
    def positions(self):
        if not self.cfg.use_geometry:
            return []
        return sorted((self.time(int(frame)), int(frame), np.array((x, y), dtype=float))
                      for frame, x, y in self.row.get("positions", ()) if math.isfinite(x) and math.isfinite(y))


def appearance(a, b, cfg, gallery="topk"):
    values = []
    if a.mean.size and a.mean.shape == b.mean.shape:
        values.append(("mean", float(a.mean @ b.mean), 1 - cfg.gallery_weight))
    if gallery != "mean" and a.prototypes and b.prototypes:
        left, right = np.array([p["e"] for p in a.prototypes]), np.array([p["e"] for p in b.prototypes])
        if left.shape[1] == right.shape[1]:
            scores = (left @ right.T).ravel()
            similarity = float(scores.max() if gallery == "max" else np.sort(scores)[-min(3, len(scores)):].mean())
            values.append(("gallery", similarity, cfg.gallery_weight))
    if not values:
        return None, {}
    reid_weight = sum(weight for _, _, weight in values)
    similarity = (sum(score * weight for _, score, weight in values) / reid_weight
                  if reid_weight else values[0][1])
    parts = []
    if cfg.part_weight:
        for left in a.row.get("body_parts", []):
            for right in b.row.get("body_parts", []):
                shared = []
                for part in ("upper", "lower"):
                    u, v = unit(left.get(part, [])), unit(right.get(part, []))
                    if u.size and u.shape == v.shape and np.isfinite(u).all() and np.isfinite(v).all():
                        shared.append(float(u @ v))
                if shared:
                    parts.append(float(np.mean(shared)))
    details = {name: score for name, score, _ in values}
    if parts:
        details["parts"] = float(np.mean(sorted(parts)[-3:]))
        similarity = (1 - cfg.part_weight) * similarity + cfg.part_weight * details["parts"]
    return float(np.clip(similarity, -1, 1)), details


def _geometry(a, b, overlapping, cfg):
    if not a.positions or not b.positions:
        return None, 0., 0, None
    if not overlapping:
        dt = b.positions[0][0] - a.positions[-1][0]
        if dt < 0:
            return None, 0., 0, None
        distance = float(np.linalg.norm(a.positions[-1][2] - b.positions[0][2]))
        limit = cfg.max_speed_mps * dt + cfg.spatial_slack_m
        return min(distance / max(limit, 1e-9), 1.), .75, 1, "speed" if distance > limit else None
    times = [x[0] for x in b.positions]
    tolerance = max(cfg.sync_tolerance_s, .5 / min(a.fps, b.fps))
    nearest = {}
    for t, frame, point in a.positions:
        j = bisect_left(times, t)
        j = min((k for k in (j - 1, j) if 0 <= k < len(times)), key=lambda k: abs(times[k] - t))
        error = abs(times[j] - t)
        if error <= tolerance and (j not in nearest or error < nearest[j][0]):
            nearest[j] = (error, frame, point)
    if not nearest:
        return None, 0., 0, None
    distances, costs, reliabilities = [], [], []
    for j, (_, frame, point) in nearest.items():
        _, other_frame, other = b.positions[j]
        delta = point - other
        distance = float(np.linalg.norm(delta))
        distances.append(distance)
        raw_a = a.row.get("position_covariances", {}).get(str(frame))
        raw_b = b.row.get("position_covariances", {}).get(str(other_frame))
        if raw_a is not None and raw_b is not None:
            matrices = [np.asarray(raw, dtype=float) for raw in (raw_a, raw_b)]
            if all(m.shape == (2, 2) and np.isfinite(m).all() and np.allclose(m, m.T)
                   and np.linalg.eigvalsh(m).min() >= 0 for m in matrices):
                cov = sum(matrices)
                md = float(delta @ np.linalg.solve(cov + 1e-6 * np.eye(2), delta))
                costs.append(min(math.sqrt(max(0., md)) / 3, 1.))
                reliabilities.append(math.exp(-float(np.trace(cov)) / (2 * cfg.geometry_sigma_m ** 2)))
                continue
        costs.append(min(distance / cfg.overlap_distance_m, 1.))
        reliabilities.append(.8)
    conflict = "simultaneous_floor_positions" if np.median(distances) > cfg.overlap_distance_m else None
    return float(np.median(costs)), float(np.median(reliabilities)), len(nearest), conflict


def physical(a, b, cfg):
    """A long gap or an unknown topology is never proof of incompatibility."""
    if a.interval[0] > b.interval[0]:
        a, b = b, a
    same_camera = a.key[0] == b.key[0]
    gap = b.interval[0] - a.interval[1]
    overlap = gap <= 1e-8
    topology = cfg.transition_model if cfg.use_time else None
    pair = frozenset((a.key[0], b.key[0]))
    conflict = None
    dt_cost, dt_reliability, status = None, 0., "unknown"
    if same_camera and (a.frames & b.frames or (overlap and not (a.frames and b.frames))):
        conflict = "same_camera_same_observation_time"
    elif topology and not same_camera:
        if overlap and pair in topology.disjoint_pairs:
            conflict = "disjoint_cameras_simultaneous"
        elif overlap and topology.allows_overlap(*pair):
            dt_cost, dt_reliability, status = 0., .75, "overlap"
        elif not overlap:
            transition = topology.evaluate(a.key[0], b.key[0], gap)
            if transition.status == "forbidden":
                conflict = "forbidden_transition"
            elif transition.status == "measured":
                dt_cost, dt_reliability, status = transition.penalty, .8, "measured"
            elif transition.status == "outside_measured_route":
                route = topology.routes[a.key[0], b.key[0]]
                if route.get("hard_min") and gap < float(route.get("min_s", 0.)):
                    conflict = "minimum_travel_time"
                else:
                    dt_cost, dt_reliability, status = 1., .4, "outside_measured_route"
    if not conflict and cfg.use_time and same_camera and 0 < gap <= 1.:
        dt_cost, dt_reliability, status = gap, .5, "short_same_camera_gap"
    distance, geo_reliability, sync_samples, geo_conflict = _geometry(a, b, overlap, cfg) if not conflict else (None, 0., 0, None)
    return {
        "conflict": conflict or geo_conflict, "gap_s": max(0., gap), "topology": status,
        "spatial": distance, "spatial_reliability": geo_reliability, "sync_samples": sync_samples,
        "temporal": dt_cost, "temporal_reliability": dt_reliability,
    }


def score(a, b, cfg, context, gallery="topk"):
    similarity, details = appearance(a, b, cfg, gallery)
    distances = {
        "appearance": None if similarity is None else (1 - similarity) / 2,
        "spatial": context["spatial"], "temporal": context["temporal"],
    }
    reliability = {
        "appearance": math.sqrt(a.quality * b.quality) if similarity is not None else 0.,
        "spatial": context["spatial_reliability"], "temporal": context["temporal_reliability"],
    }
    base = dict(zip(distances, (cfg.appearance_weight, cfg.spatial_weight, cfg.temporal_weight)))
    raw = {k: base[k] * reliability[k] if distances[k] is not None else 0. for k in base}
    denominator = sum(raw.values())
    weights = {k: v / denominator for k, v in raw.items()} if denominator else {}
    cost = sum(weights[k] * distances[k] for k in weights if distances[k] is not None) if denominator else None
    return {
        **context, "similarity": similarity, "appearance_components": details,
        "distances": distances, "weights": weights, "modality_reliability": reliability,
        "reliability": denominator / sum(base.values()), "cost": cost,
        "probability": None,
    }
