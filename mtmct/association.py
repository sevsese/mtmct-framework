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
    require_geometry: bool = False
    use_geometry: bool = True


def unit(v: list[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(a))
    return a / norm if norm > 1e-10 else a


def frames(t: dict) -> dict[int, tuple[float,float]]:
    return {int(p[0]): (float(p[1]), float(p[2])) for p in t.get("positions", [])}


def _space_constraints(a: dict, b: dict, cfg: Config) -> tuple[bool, float, float]:
    """(allowed, normalized spatial penalty, temporal penalty)."""
    start_a, end_a = int(a["start_frame"]), int(a["end_frame"])
    start_b, end_b = int(b["start_frame"]), int(b["end_frame"])
    overlapping = max(start_a, start_b) <= min(end_a, end_b)
    if overlapping and int(a["camera_id"]) == int(b["camera_id"]):
        return False, 1., 1.
    pa, pb = frames(a), frames(b)
    if not cfg.use_geometry:
        if overlapping:
            return True, 0.0, 0.0
        gap = (max(start_a, start_b) - min(end_a, end_b)) / cfg.fps
        if gap > cfg.max_gap_seconds:
            return False, 0.0, 1.0
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
    gap_frames = int(later["start_frame"]) - int(earlier["end_frame"])
    gap = gap_frames / cfg.fps
    if gap > cfg.max_gap_seconds:
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
    return True, spatial, min(gap / cfg.max_gap_seconds, 1.)


def pair_cost(a: dict, b: dict, cfg: Config) -> float | None:
    """Lower is better; None means physically impossible."""
    feasible, spatial, temporal = _space_constraints(a, b, cfg)
    if not feasible:
        return None
    v, w = unit(a.get("embedding", [])), unit(b.get("embedding", []))
    if v.size == 0 or w.size == 0 or v.shape != w.shape:
        return None
    similarity = float(np.clip(v @ w, -1.0, 1.0))
    if similarity < cfg.min_similarity:
        return None
    # Cosine distance, in [0,2]; normalize to [0,1].
    appearance = (1.0 - similarity) / 2.0
    weights = [cfg.appearance_weight, cfg.spatial_weight, cfg.temporal_weight]
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("weights must be nonnegative and at least one positive")
    return (weights[0]*appearance + weights[1]*spatial + weights[2]*temporal) / sum(weights)


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
        compatible = all(
            _space_constraints(tracklets[i], tracklets[j], cfg)[0]
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
