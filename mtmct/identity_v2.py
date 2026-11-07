"""Reliable-first offline clustering with reversible edges and shared vetoes."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
from math import isfinite

import numpy as np

from .scoring import Evidence, physical, same_view_continuity, score


@dataclass(frozen=True)
class Settings:
    top_k: int = 12
    margin: float = .006
    min_reliability: float = .12
    gallery: str = "max"
    reentry_similarity: float = .88
    anchor_quality: float = .25
    single_attach_margin: float = .03
    unsupported_similarity: float = .55
    min_lifetime_s: float = 0.
    cross_camera_mean: float | None = None
    cross_camera_gallery: float = .67
    cross_camera_quality: float = .30
    recover_same_view: bool = False


OFFICE_SETTINGS = Settings(min_lifetime_s=.75, cross_camera_mean=.72, recover_same_view=True)


def observed_lifetime(tracks):
    intervals = []
    for track in tracks:
        samples = sorted(track.time(f) for f in track.frames)
        for i, t in enumerate(samples):
            end = samples[i+1] if i+1 < len(samples) and samples[i+1]-t <= .24 else t+1/track.fps
            intervals.append((t, end))
    total = 0.
    end = float("-inf")
    for start, stop in sorted(intervals):
        total += max(0., stop-max(start, end))
        end = max(end, stop)
    return total


def associate_v2(tracklets, cfg, *, same_camera=True, revoked=(), settings=Settings()):
    if settings.top_k < 1 or settings.gallery not in {"mean", "max", "topk"}:
        raise ValueError("Invalid v2 candidate settings")
    weights = (cfg.appearance_weight, cfg.spatial_weight, cfg.temporal_weight)
    if not all(isfinite(w) and w >= 0 for w in weights) or not sum(weights):
        raise ValueError("Association weights must be finite, nonnegative and nonzero")
    if not 0 <= cfg.gallery_weight <= 1 or not 0 <= cfg.part_weight <= 1:
        raise ValueError("Gallery and body-part weights must be in [0,1]")
    tracks = [Evidence(row, cfg, settings.anchor_quality) for row in tracklets]
    if len({t.key for t in tracks}) != len(tracks) or len({t.uid for t in tracks}) != len(tracks):
        raise ValueError("Duplicate tracklet key or UID")
    for track in tracks:
        _ = track.interval
    revoked = {frozenset(pair) for pair in revoked}
    parent = list(range(len(tracks)))
    members = {i: {i} for i in parent}
    anchors = {i for i, t in enumerate(tracks) if t.anchor and
               (not settings.min_lifetime_s or observed_lifetime([t]) >= settings.min_lifetime_s)}
    # Fixed for this offline pass. Provisional attachments cannot reinforce
    # future decisions through the appearance memory.
    trusted = set(anchors)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    @lru_cache(maxsize=None)
    def context(i, j):
        return physical(tracks[i], tracks[j], cfg)

    @lru_cache(maxsize=None)
    def recovery(i, j):
        a, b = tracks[i], tracks[j]
        if not settings.recover_same_view or a.key[0] != b.key[0]:
            return None
        mean = float(a.mean @ b.mean) if a.mean.size and a.mean.shape == b.mean.shape else None
        return same_view_continuity(a, b, mean)

    def veto(i, j):
        if frozenset((tracks[i].uid, tracks[j].uid)) in revoked:
            return "revoked_by_review"
        if not same_camera and tracks[i].key[0] == tracks[j].key[0]:
            return "same_camera_disabled"
        i, j = min(i, j), max(i, j)
        conflict = context(i, j)["conflict"]
        if conflict == "same_camera_same_observation_time" and recovery(i, j):
            return None
        return conflict

    def compatible(a, b):
        return all(not veto(i, j) for i in members[a] for j in members[b])

    def unite(a, b):
        parent[b] = a
        members[a].update(members.pop(b))

    edges, neighbors = [], defaultdict(list)
    candidates = {}
    # ponytail: pair enumeration is quadratic in offline tracklets; add an
    # indexed ReID shortlist when profiling shows this dominates long runs.
    for i, left in enumerate(tracks):
        for j in range(i + 1, len(tracks)):
            right = tracks[j]
            reason = veto(i, j)
            edge = {"i": i, "j": j, "left_uid": left.uid, "right_uid": right.uid,
                    "state": "rejected", "decision": "rejected", "probability": None}
            edges.append(edge)
            if reason:
                state = "revoked" if reason == "revoked_by_review" else "rejected" if reason == "same_camera_disabled" else "conflict"
                edge.update(state=state, reason=reason)
                continue
            edge.update(score(left, right, cfg, context(i, j), settings.gallery))
            seam = recovery(i, j)
            if seam:
                edge.update(conflict=None, continuity=seam, state="candidate", decision="deferred",
                            reason="verified_same_view_continuity")
                continue
            same_view = left.key[0] == right.key[0]
            if same_view and settings.recover_same_view:
                edge["reason"] = "no_same_view_continuity"
                continue
            if not same_view and settings.cross_camera_mean is not None:
                parts = edge["appearance_components"]
                if (min(left.quality, right.quality) < settings.cross_camera_quality
                        or min(len(left.prototypes), len(right.prototypes)) < 2):
                    edge["reason"] = "insufficient_cross_camera_crop_quality"
                    continue
                if (parts.get("mean", -1.) < settings.cross_camera_mean
                        or parts.get("gallery_top3", -1.) < settings.cross_camera_gallery):
                    edge["reason"] = "weak_cross_camera_reid"
                    continue
            similarity = edge["similarity"]
            independent_geometry = edge["sync_samples"] >= 2 and edge["spatial"] <= .25
            long_gap = cfg.use_time and edge["gap_s"] > cfg.max_gap_seconds
            if edge["cost"] is None or edge["reliability"] < settings.min_reliability:
                edge["reason"] = "insufficient_evidence"
            elif not independent_geometry and (similarity is None or similarity < cfg.min_similarity):
                edge["reason"] = "weak_or_missing_appearance"
            elif (not independent_geometry and edge["topology"] in {"unknown", "outside_measured_route"}
                  and similarity < settings.unsupported_similarity):
                edge["reason"] = "unconfirmed_without_topology"
            elif long_gap and (similarity is None or similarity < settings.reentry_similarity):
                edge["reason"] = "unconfirmed_reentry"
            elif edge["cost"] > cfg.max_cost:
                edge["reason"] = "high_cost"
            else:
                edge.update(state="candidate", decision="deferred", reason="scored")
                candidates[i, j] = edge
                neighbors[i].append((edge["cost"], j))
                neighbors[j].append((edge["cost"], i))

    continuity_edges = sorted((e for e in edges if e.get("continuity")),
                              key=lambda e: (-e["appearance_components"]["mean"], e["i"], e["j"]))
    for edge in continuity_edges:
        a, b = root(edge["i"]), root(edge["j"])
        if a != b and compatible(a, b):
            unite(a, b)
            edge.update(state="confirmed", decision="accepted", reason=edge["continuity"], stage="continuity")
        elif a != b:
            edge.update(reason="component_conflict")

    shortlist = {(min(i, j), max(i, j)) for i, items in neighbors.items()
                 for _, j in sorted(items)[:settings.top_k]}
    for pair in candidates.keys() - shortlist:
        candidates[pair]["reason"] = "outside_top_k"
    ordered = sorted((candidates[pair] for pair in shortlist),
                     key=lambda e: (e["cost"], -e["reliability"], e["left_uid"], e["right_uid"]))
    for edge in ordered:
        i, j = edge["i"], edge["j"]
        similarity = edge["similarity"]
        if i not in anchors or j not in anchors or similarity is None:
            continue
        # Compatible alternatives may be successive views of the same
        # person. Only mutually exclusive alternatives compete for a seed.
        alternative = min((cost for k, other in ((i, j), (j, i))
                           for cost, rival in neighbors[k] if rival != other and veto(other, rival)),
                          default=float("inf"))
        if alternative - edge["cost"] < settings.margin:
            edge["reason"] = "ambiguous_seed"
            continue
        a, b = root(i), root(j)
        if a == b:
            continue
        if not compatible(a, b):
            edge.update(reason="component_conflict")
            continue
        unite(a, b)
        edge.update(state="confirmed", decision="accepted", reason="anchor_edge", stage="seed")

    confirmed_roots = {root(i) for i in anchors}
    for i in sorted(parent, key=lambda i: (-tracks[i].quality, tracks[i].uid)):
        if root(i) != i or len(members[i]) != 1:
            continue
        groups = defaultdict(dict)
        for _, j in neighbors[i]:
            if j not in trusted or root(j) not in confirmed_roots:
                continue
            edge = candidates[min(i, j), max(i, j)]
            if (min(i, j), max(i, j)) not in shortlist:
                continue
            similarity = edge["similarity"]
            if similarity is None or similarity < cfg.min_similarity:
                continue
            group, camera = root(j), tracks[j].key[0]
            if i in anchors and len(members[group] & trusted) < 2:
                continue
            old = groups[group].get(camera)
            if old is None or edge["cost"] < old["cost"]:
                groups[group][camera] = edge
        ranked = []
        for group, per_camera in groups.items():
            if not compatible(i, group):
                continue
            links = sorted(per_camera.values(), key=lambda e: e["cost"])[:2]
            references = [j for j in members[group] if j in trusted]
            prototypes = [np.asarray(p["e"]) for j in references for p in tracks[j].prototypes]
            vector = tracks[i].mean
            similarities = sorted((float(vector @ p) for p in prototypes if p.shape == vector.shape),
                                  reverse=True)[:3]
            edge_support = 1 - float(np.mean([e["cost"] for e in links]))
            prototype_support = (1 + float(np.mean(similarities))) / 2 if similarities else edge_support
            support = .4 * prototype_support + .6 * edge_support
            single_threshold = min(1., cfg.min_similarity + settings.single_attach_margin)
            if support >= 1 - cfg.max_cost and (len(links) > 1 or links[0]["similarity"] >= single_threshold
                                                or links[0]["topology"] == "measured"):
                ranked.append((support, group, links))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < settings.margin):
            continue
        support, group, links = ranked[0]
        unite(group, i)
        trusted.discard(i)
        links[0].update(state="provisional", decision="accepted", reason="gallery_and_edges",
                        stage="attachment", support=support, supporting_cameras=len(links))

    groups = sorted((sorted(group) for group in members.values()),
                    key=lambda group: min((tracks[i].interval[0], tracks[i].uid) for i in group))
    return tracks, groups, edges


def link_v2(run_dir, manifest, tracklets, cfg, *, scene_id=0, same_camera=True,
            min_observations=1, settings=Settings(), revocations=None, repair_config=None):
    from .association import save_mapping
    from .export import export_predictions
    from .identity_store import prior_revocations, register
    from .occlusion_repair import find_duplicate_tracklets, read_observations
    from .refinement import rewrite_observations, split_sustained

    tracking = manifest.get("tracking_config", {})
    camera_fps = {int(c): float(fps) for c, fps in tracking.get("source_fps", {}).items()}
    offsets = {int(c): float(t) for c, t in tracking.get("camera_offsets_s", {}).items()}
    cfg = replace(cfg, camera_fps=camera_fps | (cfg.camera_fps or {}),
                  camera_offsets_s=offsets | (cfg.camera_offsets_s or {}),
                  spatial_weight=cfg.spatial_weight if cfg.use_geometry else 0.)
    source = run_dir / "observations.csv"
    observations = read_observations(source)
    for row in tracklets:
        key = int(row["camera_id"]), int(row["local_id"])
        recorded = observations.get(key, {})
        if recorded:
            row["observation_frames"] = sorted(recorded)
            row["observation_boxes"] = {str(f): [float(o[k]) for k in ("x1","y1","x2","y2")]
                                        for f, o in recorded.items()}
            timestamps = {str(f): float(o["timestamp_s"]) for f, o in recorded.items()
                          if o.get("timestamp_s") not in (None, "")}
            if timestamps:
                row["frame_timestamps"] = timestamps
        if not cfg.use_geometry:
            for obs in recorded.values():
                obs.pop("world_x", None)
                obs.pop("world_y", None)
    duplicates = []
    if repair_config is not None:
        tracklets, _, duplicates = find_duplicate_tracklets(tracklets, observations, repair_config)
    tracklets, remap, local_diagnostics = split_sustained(tracklets, cfg)
    tracklets = [t for t in tracklets if len(t.get("observation_frames", [])) >= min_observations]
    registry = run_dir/"identity_registry.sqlite"
    if revocations is None:
        revocations = prior_revocations(registry)
    tracks, groups, edges = associate_v2(tracklets, cfg, same_camera=same_camera,
                                         revoked=revocations, settings=settings)
    short = []
    if settings.min_lifetime_s:
        retained = []
        for group in groups:
            lifetime = observed_lifetime([tracks[i] for i in group])
            if lifetime < settings.min_lifetime_s:
                short.append({"tracklets": [tracks[i].uid for i in group],
                              "observed_seconds": round(lifetime, 3)})
            else:
                retained.append(group)
        groups = retained
    provenance = {"scene": manifest.get("scene"), "inputs": manifest.get("inputs"),
                  "models": manifest.get("models")}
    scope = hashlib.sha256(json.dumps(provenance if provenance["inputs"] else str(run_dir.resolve()),
                                       sort_keys=True).encode()).hexdigest()
    mapping, state = register(registry, scope, tracks, groups, edges)
    state["config"] = {"geometry": cfg.use_geometry, "settings": asdict(settings),
                       "weights": [cfg.appearance_weight, cfg.spatial_weight, cfg.temporal_weight],
                       "camera_fps": cfg.camera_fps, "camera_offsets_s": cfg.camera_offsets_s,
                       "max_cost": cfg.max_cost, "min_similarity": cfg.min_similarity,
                       "long_gap_s": cfg.max_gap_seconds}
    state["local_corrections"] = {"switches": local_diagnostics, "duplicates": duplicates}
    state["short_unconfirmed"] = short
    for filename, content in (("identity_state.json", state), ("association_edges.json", edges),
                              ("refined_tracklets.json", tracklets)):
        (run_dir/filename).write_text(json.dumps(content, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    save_mapping(mapping, run_dir/"global_ids.json")
    if remap:
        source = run_dir/"refined_observations.csv"
        rewrite_observations(run_dir/"observations.csv", source, remap)
    suppressed_boxes = set()
    if settings.recover_same_view:
        for group in groups:
            best = {}
            for i in group:
                track = tracks[i]
                cam, local = track.key
                for frame, (x1,y1,x2,y2) in track.row.get("observation_boxes", {}).items():
                    key = cam, int(frame)
                    box = (cam, local, int(frame))
                    area = (x2-x1)*(y2-y1)
                    previous = best.get(key)
                    if previous is None or area > previous[0]:
                        if previous is not None:
                            suppressed_boxes.add(previous[1])
                        best[key] = (area, box)
                    else:
                        suppressed_boxes.add(box)
    out_csv, out_txt = export_predictions(source, run_dir/"global_ids.json", run_dir, scene_id=scene_id,
                                         excluded_observations=suppressed_boxes)
    return tracklets, mapping, edges, out_csv, out_txt
