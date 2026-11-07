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
    def usable_samples(self):
        """One quality policy for prototypes, gallery scoring and evidence gates."""
        samples = []
        details = self.row.get('feature_quality_details') or []
        for i, (v, q, f) in enumerate(zip(self.row.get('feature_vectors') or [],
                                        self.row.get('feature_quality') or [],
                                        self.row.get('feature_frames') or [])):
            e = unit(v)
            q = float(q) * (1 - .5 * (details[i].get('occlusion_proxy', 0.) if i < len(details) else 0.))
            if (e.ndim == 1 and e.size and np.isfinite(e).all() and np.any(e)
                    and 0 < q <= 1 and (not samples or e.shape == samples[0][0].shape)):
                samples.append((e, q, int(f), i))
        best = max((s[1] for s in samples), default=0.)
        return [s for s in samples if s[1] >= max(.2, .35 * best)]

    @cached_property
    def temporal_gallery(self):
        samples = sorted({s[2]:s for s in self.usable_samples}.values(), key=lambda s:s[2])
        result = {}
        for name, indexes in zip(('early','middle','late'), np.array_split(np.arange(len(samples)),3)):
            if not len(indexes):
                continue
            group = [samples[i] for i in indexes]
            vector = unit(np.average([s[0] for s in group],axis=0,weights=[s[1] for s in group]))
            result[name] = dict(vector=vector.tolist(),quality=float(np.mean([s[1] for s in group])),
                independent_frames=len(group), spread=float(np.mean([1-np.dot(s[0],vector) for s in group])),
                start_s=self.time(group[0][2]),end_s=self.time(group[-1][2]))
        return result

    @cached_property
    def prototypes(self):
        details = self.row.get("feature_quality_details") or []
        samples = self.usable_samples
        if not samples:
            return []
        pool = list(samples)
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


def appearance(a, b, cfg, gallery="topk", temporal=False):
    values = []
    top3 = None
    if a.mean.size and a.mean.shape == b.mean.shape:
        values.append(("mean", float(a.mean @ b.mean), 1 - cfg.gallery_weight))
    if gallery != "mean" and a.prototypes and b.prototypes:
        def usable(track):
            return np.array([s[0] for s in track.usable_samples])
        left, right = usable(a), usable(b)
        if left.shape[1] == right.shape[1]:
            matrix = left @ right.T
            scores = matrix.ravel()
            if temporal:
                used_left, used_right, selected = set(), set(), []
                for flat in np.argsort(-scores):
                    i,j = np.unravel_index(flat, matrix.shape)
                    fi,fj = a.usable_samples[i][2],b.usable_samples[j][2]
                    if fi in used_left or fj in used_right:
                        continue
                    used_left.add(fi); used_right.add(fj)
                    selected.append(float(matrix[i,j]))
                scores = np.asarray(selected)
            top3 = float(np.sort(scores)[-min(3, len(scores)):].mean())
            similarity = float(scores.max()) if gallery == "max" else top3
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
    if top3 is not None:
        details["gallery_top3"] = top3
        if temporal:
            details['independent_support_frames'] = min(3,len(scores))
            comparisons = {}
            for na,ga in a.temporal_gallery.items():
                for nb,gb in b.temporal_gallery.items():
                    if len(ga['vector']) == len(gb['vector']):
                        comparisons[na+'_'+nb] = float(np.dot(ga['vector'],gb['vector']))
            details['temporal_parts'] = comparisons
            if comparisons:
                details['temporal_similarity'] = max(comparisons.values())
            details['early_similarity'] = comparisons.get('early_early')
    if parts:
        details["parts"] = float(np.mean(sorted(parts)[-3:]))
        similarity = (1 - cfg.part_weight) * similarity + cfg.part_weight * details["parts"]
    return float(np.clip(similarity, -1, 1)), details


def same_view_continuity(a, b, mean):
    """Recover a short handoff using image boxes, without floor geometry."""
    if a.key[0] != b.key[0] or mean is None:
        return None
    if a.interval[0] > b.interval[0]:
        a, b = b, a
    left, right = (t.row.get("observation_boxes", {}) for t in (a, b))
    common = sorted(a.frames & b.frames)
    if common:
        # A persistent overlap between two full people is never a handoff.
        # Only a short seam or a nearly fully shadowed brief duplicate qualifies.
        seam = a.time(common[-1]) - a.time(common[0]) <= .20
        duplicate = (min(len(a.frames), len(b.frames)) <= 20
                     and len(common) >= .85 * min(len(a.frames), len(b.frames)))
        if len(common) < 2 or not (seam or duplicate) or mean < (.70 if duplicate else .82):
            return None
        for frame in common:
            if str(frame) not in left or str(frame) not in right:
                return None
            x, y = (np.asarray(boxes[str(frame)]) for boxes in (left, right))
            sizes = np.maximum(0, np.array([x[2:]-x[:2], y[2:]-y[:2]]))
            area = sizes.prod(axis=1)
            overlap = np.maximum(0, np.minimum(x[2:],y[2:])-np.maximum(x[:2],y[:2])).prod()
            near_identical = overlap / max(float(area.sum()-overlap), 1e-9) >= .85
            if (min(area) <= 0 or overlap < .90 * min(area) or (min(area) > .75 * max(area) and not near_identical)
                    or abs(x[1]-y[1]) > .15 * max(sizes[:,1])):
                return None
        return "shadowed_duplicate" if duplicate else "overlapping_handoff"
    gap = b.interval[0] - a.interval[1]
    if not 0 < gap <= 1.8 or mean < (.62 if gap <= .50 else .70) or not left or not right:
        return None
    x, y = np.array(left[max(left, key=int)]), np.array(right[min(right, key=int)])
    h = max(x[3]-x[1], y[3]-y[1], 1.)
    center_x, center_y = (x[:2]+x[2:])/2, (y[:2]+y[2:])/2
    # A longer seam needs a tighter displacement bound, not a global ReID relaxation.
    if np.linalg.norm(center_x-center_y) <= h * (.30 + .20*gap):
        return "short_gap_recovery"
    return None


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
            # Permission to co-occur is not a measurement of correspondence.
            status = "overlap"
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


def score(a, b, cfg, context, gallery="topk", temporal=False):
    similarity, details = appearance(a, b, cfg, gallery, temporal=temporal)
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
