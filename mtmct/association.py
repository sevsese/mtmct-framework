"""Offline global ID association over local tracklets, without model training.

Edges combine appearance, floor position, and time. Hard constraints are
re-checked *across all members* before any union, preventing contradictory
transitive merges (same camera / time, implausible physical separation).
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np
from .time_sync import track_times


@dataclass
class Config:
    appearance_weight: float = 0.75
    spatial_weight: float = 0.20
    temporal_weight: float = 0.05
    max_cost: float = 0.35
    min_similarity: float = 0.80
    max_gap_seconds: float = 12.0
    max_speed_mps: float = 3.5
    spatial_slack_m: float = 1.5
    overlap_distance_m: float = 2.0
    fps: float = 30.0
    camera_fps: dict[int, float] | None = None
    require_geometry: bool = False
    use_geometry: bool = True
    use_time: bool = True
    camera_offsets_s: dict[int, float] | None = None
    transition_model: object | None = None
    gallery_weight: float = 0.0
    gallery_size: int = 6
    part_weight: float = 0.0
    max_interior_gap_seconds: float = 30.0
    max_interior_motion_heights: float = 4.0
    duplicate_overlap_iou: float = 0.5


def unit(v: list[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(a))
    return a / norm if norm > 1e-10 else a


def appearance_similarity(a: dict, b: dict, cfg: Config) -> float | None:
    """Blend track means with the best pair of quality-selected observations."""
    v, w = unit(a.get("embedding", [])), unit(b.get("embedding", []))
    if v.size == 0 or w.size == 0 or v.shape != w.shape:
        return None
    mean_similarity = float(np.clip(v @ w, -1.0, 1.0))
    weight = float(cfg.gallery_weight)
    if not 0 <= weight <= 1:
        raise ValueError("gallery_weight must be between 0 and 1")
    limit = max(1, int(cfg.gallery_size))
    left = [unit(row) for row in (a.get("feature_vectors") or [])[:limit]]
    right = [unit(row) for row in (b.get("feature_vectors") or [])[:limit]]
    scores = [float(x @ y) for x in left for y in right
              if x.size and x.shape == y.shape]
    reid_similarity = (mean_similarity if weight <= 0 or not scores else
                       (1.0 - weight) * mean_similarity + weight * max(scores))
    part_weight = float(cfg.part_weight)
    if not 0 <= part_weight <= 1:
        raise ValueError("part_weight must be between 0 and 1")
    part_scores = []
    for left_parts in a.get("body_parts", []):
        for right_parts in b.get("body_parts", []):
            shared = []
            for name in ("upper", "lower"):
                x, y = unit(left_parts.get(name, [])), unit(right_parts.get(name, []))
                if x.size and x.shape == y.shape:
                    shared.append(float(x @ y))
            if shared:
                part_scores.append(sum(shared) / len(shared))
    if not part_scores or part_weight <= 0:
        return float(np.clip(reid_similarity, -1.0, 1.0))
    return float(np.clip((1.0 - part_weight) * reid_similarity +
                         part_weight * max(part_scores), -1.0, 1.0))


def frames(t: dict) -> dict[int, tuple[float,float]]:
    return {int(p[0]): (float(p[1]), float(p[2])) for p in t.get("positions", [])}


def _interior_reconnect(earlier: dict, later: dict, gap: float, cfg: Config) -> bool:
    if int(earlier["camera_id"]) != int(later["camera_id"]):
        return False
    if gap > cfg.max_interior_gap_seconds:
        return False
    a, b = earlier.get("end_bbox_norm"), later.get("start_bbox_norm")
    if not a or not b:
        return False
    def interior(box):
        cx, bottom = (box[0] + box[2]) / 2, box[3]
        return .04 < cx < .96 and bottom < .97
    if not interior(a) or not interior(b):
        return False
    height = max(((a[3] - a[1]) + (b[3] - b[1])) / 2, 1e-6)
    center_a = np.array(((a[0] + a[2]) / 2, (a[1] + a[3]) / 2))
    center_b = np.array(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    return float(np.linalg.norm(center_a - center_b) / height) <= cfg.max_interior_motion_heights


def _same_camera_conflict(a: dict, b: dict, cfg: Config) -> bool:
    shared = set(a.get("observation_frames", [])) & set(b.get("observation_frames", []))
    if not a.get("observation_frames") or not b.get("observation_frames"):
        return True
    boxes_a, boxes_b = a.get("observation_boxes", {}), b.get("observation_boxes", {})
    for frame in shared:
        left, right = boxes_a.get(str(frame)), boxes_b.get(str(frame))
        if not left or not right:
            return True
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        area_b = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        iou = intersection / max(area_a + area_b - intersection, 1e-9)
        if iou < cfg.duplicate_overlap_iou:
            return True
    return False


def _component_conflict(a: dict, b: dict, cfg: Config) -> bool:
    """A chain may span > max_gap; only simultaneous contradictions are global."""
    fps = cfg.camera_fps or cfg.fps
    start_a, end_a = track_times(a, fps, cfg.camera_offsets_s)
    start_b, end_b = track_times(b, fps, cfg.camera_offsets_s)
    if max(start_a, start_b) > min(end_a, end_b):
        return False
    if cfg.use_geometry:
        positions_a, positions_b = frames(a), frames(b)
        common = set(positions_a) & set(positions_b)
        if common:
            distance = float(np.median([
                np.linalg.norm(np.subtract(positions_a[frame], positions_b[frame]))
                for frame in common
            ]))
            if distance > cfg.overlap_distance_m:
                return True
    if int(a["camera_id"]) == int(b["camera_id"]):
        return _same_camera_conflict(a, b, cfg)
    return bool(cfg.transition_model and not cfg.transition_model.allows_overlap(
        int(a["camera_id"]), int(b["camera_id"])))


def _space_constraints(a: dict, b: dict, cfg: Config) -> tuple[bool, float, float]:
    """(allowed, normalized spatial penalty, temporal penalty)."""
    start_a_frame, end_a_frame = int(a["start_frame"]), int(a["end_frame"])
    start_b_frame, end_b_frame = int(b["start_frame"]), int(b["end_frame"])
    fps = cfg.camera_fps or cfg.fps
    start_a, end_a = track_times(a, fps, cfg.camera_offsets_s)
    start_b, end_b = track_times(b, fps, cfg.camera_offsets_s)
    overlapping = max(start_a, start_b) <= min(end_a, end_b)
    if overlapping and int(a["camera_id"]) == int(b["camera_id"]):
        if _same_camera_conflict(a, b, cfg):
            return False, 1., 1.
        return True, 0.0, 0.0
    if not cfg.use_geometry and not cfg.use_time:
        return True, 0.0, 0.0
    pa, pb = frames(a), frames(b)
    if not cfg.use_geometry:
        if overlapping:
            if cfg.transition_model and not cfg.transition_model.allows_overlap(
                    int(a["camera_id"]), int(b["camera_id"])):
                return False, 0.0, 1.0
            return True, 0.0, 0.0
        gap = max(start_a, start_b) - min(end_a, end_b)
        earlier, later = (a,b) if end_a < start_b else (b,a)
        if cfg.transition_model:
            transition = cfg.transition_model.evaluate(
                int(earlier["camera_id"]), int(later["camera_id"]), gap)
            if not transition.allowed:
                return False, 0.0, 1.0
        if gap > cfg.max_gap_seconds and not _interior_reconnect(earlier, later, gap, cfg):
            return False, 0.0, 1.0
        if cfg.transition_model:
            if transition.status == "unknown":
                return True, 0.0, min(gap / max(cfg.max_gap_seconds, 1e-10), 1.0)
            return True, 0.0, transition.penalty
        return True, 0.0, min(gap / cfg.max_gap_seconds, 1.0)
    if overlapping:
        common = sorted(set(pa).intersection(pb))
        if common:
            distances = [float(np.linalg.norm(np.subtract(pa[f], pb[f]))) for f in common]
            # A single valid synchronized sample is already informative, but
            # robust median suppresses a bad foot-point estimate.
            d = float(np.median(distances))
            if d > cfg.overlap_distance_m:
                return False, 1., 1.
            return True, min(d / cfg.overlap_distance_m, 1.), 0.
        # No simultaneous projected samples: cannot use overlap geometry.
        if cfg.require_geometry:
            return False, 1., 1.
        return True, 0.5, 0.

    # Choose chronological order for transition test.
    earlier, later = (a,b) if end_a < start_b else (b,a)
    gap = max(start_a, start_b) - min(end_a, end_b)
    transition_penalty = None
    if cfg.transition_model:
        transition = cfg.transition_model.evaluate(
            int(earlier["camera_id"]), int(later["camera_id"]), gap)
        if not transition.allowed:
            return False, 1., 1.
        transition_penalty = transition.penalty if transition.status != "unknown" else None
    if gap > cfg.max_gap_seconds and not _interior_reconnect(earlier, later, gap, cfg):
        return False, 1., 1.
    p, q = frames(earlier), frames(later)
    if p and q:
        p_last = p[max(p)]; q_first = q[min(q)]
        dist = float(np.linalg.norm(np.subtract(p_last, q_first)))
        allowed = cfg.max_speed_mps * max(gap, 1.0 / cfg.fps) + cfg.spatial_slack_m
        if dist > allowed:
            return False, 1., 1.
        spatial = min(dist / max(allowed, 1e-10), 1.)
    elif cfg.require_geometry:
        return False, 1., 1.
    else:
        spatial = 0.5
    temporal = transition_penalty if transition_penalty is not None else min(
        gap / max(cfg.max_gap_seconds, 1e-10), 1.)
    return True, spatial, temporal


def pair_cost(a: dict, b: dict, cfg: Config) -> float | None:
    """Lower is better; None means physically impossible."""
    score = pair_score(a, b, cfg)
    return None if score["cost"] is None else float(score["cost"])


def pair_score(a: dict, b: dict, cfg: Config) -> dict:
    """Explain a candidate score without exposing any GT information."""
    feasible, spatial, temporal = _space_constraints(a, b, cfg)
    if not feasible:
        return {"cost": None, "similarity": None, "spatial": spatial,
                "temporal": temporal, "reason": "hard_constraint"}
    similarity = appearance_similarity(a, b, cfg)
    if similarity is None:
        return {"cost": None, "similarity": None, "spatial": spatial,
                "temporal": temporal, "reason": "missing_reid"}
    if similarity < cfg.min_similarity:
        return {"cost": None, "similarity": similarity, "spatial": spatial,
                "temporal": temporal, "reason": "weak_appearance"}
    # Cosine distance, in [0,2]; normalize to [0,1].
    appearance = (1.0 - similarity) / 2.0
    weights = [cfg.appearance_weight, cfg.spatial_weight, cfg.temporal_weight]
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("weights must be nonnegative and at least one positive")
    cost = (weights[0]*appearance + weights[1]*spatial + weights[2]*temporal) / sum(weights)
    return {"cost": float(cost), "similarity": similarity, "spatial": spatial,
            "temporal": temporal, "reason": "candidate"}


def associate_graph(tracklets: list[dict], cfg: Config,
                    allow_same_camera_fragments: bool = False):
    """Constrained agglomeration with a decision journal for every pair."""
    keys = [(int(row["camera_id"]), int(row["local_id"])) for row in tracklets]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (camera_id, local_id) in tracklets")
    parent = list(range(len(tracklets)))
    members = {index: {index} for index in range(len(tracklets))}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    journal = []
    candidates = []
    for left in range(len(tracklets)):
        for right in range(left + 1, len(tracklets)):
            if (tracklets[left]["camera_id"] == tracklets[right]["camera_id"]
                    and not allow_same_camera_fragments):
                journal.append({"i": left, "j": right, "decision": "rejected",
                                "status": "rejected", "reason": "same_camera_disabled"})
                continue
            score = pair_score(tracklets[left], tracklets[right], cfg)
            edge = {"i": left, "j": right, **score}
            if score["cost"] is None:
                edge.update(decision="rejected",status="rejected")
                journal.append(edge)
            elif score["cost"] > cfg.max_cost:
                edge.update(decision="rejected", status="rejected", reason="max_cost")
                journal.append(edge)
            else:
                candidates.append(edge)
    candidates.sort(key=lambda edge: edge["cost"])
    for edge in candidates:
        left_root, right_root = find(edge["i"]), find(edge["j"])
        if left_root == right_root:
            edge.update(decision="rejected", status="rejected", reason="already_connected")
            journal.append(edge)
            continue
        compatible = True
        for left in members[left_root]:
            for right in members[right_root]:
                same_camera = tracklets[left]["camera_id"] == tracklets[right]["camera_id"]
                if same_camera and not allow_same_camera_fragments:
                    compatible = False
                elif _component_conflict(tracklets[left], tracklets[right], cfg):
                    compatible = False
                if not compatible:
                    break
            if not compatible:
                break
        if not compatible:
            edge.update(decision="rejected", status="rejected", reason="component_conflict")
            journal.append(edge)
            continue
        parent[right_root] = left_root
        members[left_root].update(members.pop(right_root))
        edge.update(decision="accepted", status="provisional", reason="best_compatible_edge")
        journal.append(edge)
    roots = sorted({find(index) for index in range(len(tracklets))}, key=lambda root: (
        min(int(tracklets[index]["start_frame"]) for index in members[root]),
        min(keys[index] for index in members[root]),
    ))
    global_ids = {root: index + 1 for index, root in enumerate(roots)}
    mapping = {keys[index]: global_ids[find(index)] for index in range(len(tracklets))}
    return mapping, journal


def associate(tracklets: list[dict], cfg: Config) -> tuple[dict[tuple[int,int], int], list[dict]]:
    """Return global IDs while allowing non-overlapping tracker fragments.

    Complexity O(N^2), intentionally for small-scene hackathon baseline.
    Identical local IDs in different cameras are independent keys.
    """
    keys = [(int(t["camera_id"]), int(t["local_id"])) for t in tracklets]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (camera_id, local_id) in tracklets")
    n = len(tracklets)
    parent = list(range(n))
    members = {i: {i} for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges=[]
    for i in range(n):
        for j in range(i+1,n):
            cost=pair_cost(tracklets[i],tracklets[j],cfg)
            if cost is not None and cost<=cfg.max_cost:
                edges.append({"i":i,"j":j,"cost":round(float(cost),6)})
    edges.sort(key=lambda e:e["cost"])

    for edge in edges:
        r1 = find(edge["i"]); r2 = find(edge["j"])
        if r1 == r2:
            continue
        # Complete-link feasibility avoids inconsistent chains A~B, B~C,
        # while A and C are physically impossible.
        compatible = not any(
            _component_conflict(tracklets[i], tracklets[j], cfg)
            for i in members[r1] for j in members[r2]
        )
        if not compatible:
            continue
        parent[r2] = r1
        members[r1].update(members.pop(r2))
    roots = sorted({find(i) for i in range(n)}, key=lambda r: (
        min(int(tracklets[i]["start_frame"]) for i in members[r]),
        min(keys[i] for i in members[r])
    ))
    root_to_global = {r: k+1 for k,r in enumerate(roots)}
    mapping = {keys[i]: root_to_global[find(i)] for i in range(n)}
    return mapping, edges


def load_tracklets(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_mapping(mapping: dict[tuple[int,int],int], path: str | Path):
    rows = [{"camera_id":c,"local_id":l,"global_id":g}
            for (c,l),g in sorted(mapping.items())]
    with open(path,"w",encoding="utf-8") as f:
        json.dump(rows,f,indent=2)
