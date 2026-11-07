"""Reliable-first offline clustering with reversible edges and shared vetoes."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
from math import isfinite

import numpy as np

from .scoring import Evidence, physical, same_view_continuity, occlusion_evidence, score
from .candidates_v2 import candidate_pairs


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
    calibration_path: str | None = None
    optimizer: str = 'agglomerative'
    identity_bank_path: str | None = None
    temporal_gallery_enabled: bool = False
    overlap_matching_enabled: bool = False
    overlap_bonus_enabled: bool = False
    overlap_bonus_weight: float = .05
    mutual_matching_enabled: bool = False
    separate_hypotheses: bool = False


OFFICE_SETTINGS = Settings(min_lifetime_s=1., cross_camera_mean=.72, recover_same_view=True,
                           identity_bank_path='runs/office_identity_bank.sqlite')
PLAN_SETTINGS = replace(OFFICE_SETTINGS, temporal_gallery_enabled=True, overlap_matching_enabled=True,
                        overlap_bonus_enabled=True, mutual_matching_enabled=True, separate_hypotheses=True)


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


def group_artifacts(tracks):
    """Quarantine short wide crops born near two people during an occlusion.

    This is a suspicion flag, not a claim that every wide/short box is false.
    No hair/pose attributes are inferred from a rectangular crop.
    """
    suspects = {}
    for i, track in enumerate(tracks):
        boxes = track.row.get('observation_boxes') or {}
        if not boxes or not 0 < observed_lifetime([track]) <= 4.:
            continue
        rects = np.array(list(boxes.values()), dtype=float)
        sizes = rects[:,2:] - rects[:,:2]
        if np.median(sizes[:,0] / np.maximum(1., sizes[:,1])) < .55:
            continue
        first = min(boxes, key=int)
        box = np.array(boxes[first])
        height = max(1., box[3]-box[1])
        neighbors, occluded_frames = [], set()
        for j, other in enumerate(tracks):
            previous = other.row.get('observation_boxes') or {}
            if i == j or track.key[0] != other.key[0] or not previous:
                continue
            nearest = min(previous, key=lambda f: abs(other.time(int(f))-track.time(int(first))))
            old = np.array(previous[nearest])
            if (abs(other.time(int(nearest))-track.time(int(first))) <= 2.
                    and other.interval[0] < track.interval[0]
                    and np.linalg.norm((box[:2]+box[2:]-old[:2]-old[2:])/2) <= .6*height):
                neighbors.append(other.uid)
            for f in boxes.keys() & previous.keys():
                a, b = np.array(boxes[f]), np.array(previous[f])
                inter = np.maximum(0., np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
                if inter >= .35 * max(1., (a[2:]-a[:2]).prod()):
                    occluded_frames.add(f)
        if len(neighbors) >= 2 and len(occluded_frames) >= .2 * len(boxes):
            suspects[i] = {'state':'artifact_candidate', 'nearby_tracklets':neighbors,
                           'occluded_fraction':len(occluded_frames)/len(boxes)}
    return suspects


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
        track.row.pop('artifact_candidate', None)
    artifacts = group_artifacts(tracks) if settings.recover_same_view else {}
    for i, details in artifacts.items():
        tracks[i].row['artifact_candidate'] = details
    revoked = {frozenset(pair) for pair in revoked}
    parent = list(range(len(tracks)))
    members = {i: {i} for i in parent}
    anchors = {i for i, t in enumerate(tracks) if i not in artifacts and t.anchor and
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

    @lru_cache(maxsize=None)
    def pair_score(i, j):
        return score(tracks[i], tracks[j], cfg, context(i, j), settings.gallery,
                     temporal=settings.temporal_gallery_enabled)

    def veto(i, j):
        if i in artifacts or j in artifacts:
            return 'artifact_candidate'
        if frozenset((tracks[i].uid, tracks[j].uid)) in revoked:
            return "revoked_by_review"
        if not same_camera and tracks[i].key[0] == tracks[j].key[0]:
            return "same_camera_disabled"
        i, j = min(i, j), max(i, j)
        conflict = context(i, j)["conflict"]
        if conflict == "same_camera_same_observation_time" and recovery(i, j):
            return None
        return conflict

    def compatible(a, b, route_support=False):
        costs = []
        for i in members[a]:
            for j in members[b]:
                if veto(i, j):
                    return False
                if settings.cross_camera_mean is None or recovery(i, j):
                    continue
                edge = pair_score(min(i,j), max(i,j))
                parts = edge['appearance_components']
                if settings.separate_hypotheses:
                    # Missing or weak pairwise evidence is neutral. Aggregate
                    # the available evidence instead of imposing an overlap gate
                    # on every pair in an otherwise supported component.
                    if edge['cost'] is not None:
                        costs.append((edge['cost'],edge['modality_reliability']['appearance']))
                    continue
                # Insufficient evidence is not a cannot-link; contradictory
                # appearance or a failed short local seam cannot be bypassed.
                if (not settings.separate_hypotheses and parts.get('mean', 1.) < .55 and parts.get('gallery_top3', 1.) < .55):
                    return False
                if (not route_support and tracks[i].key[0] != tracks[j].key[0]
                        and edge['topology'] == 'overlap' and edge['spatial_reliability'] == 0
                        and parts.get('mean', -1.) < settings.cross_camera_mean):
                    return False
                if (tracks[i].key[0] == tracks[j].key[0]
                        and 0 < edge['gap_s'] <= 1.8):
                    return False
        if costs and sum(w for _,w in costs)>0:
            return sum(c*w for c,w in costs)/sum(w for _,w in costs) <= cfg.max_cost
        return True

    def unite(a, b):
        parent[b] = a
        members[a].update(members.pop(b))

    edges, neighbors = [], defaultdict(list)
    candidates = {}
    pairs = set(candidate_pairs(tracks, settings.top_k))
    if settings.overlap_matching_enabled and cfg.use_time and cfg.transition_model:
        pairs.update((i,j) for i,a in enumerate(tracks) for j,b in enumerate(tracks) if i<j
                     and cfg.transition_model.allows_overlap(a.key[0],b.key[0])
                     and min(a.interval[1],b.interval[1])>max(a.interval[0],b.interval[0]))
    for i, j in sorted(pairs):
        left = tracks[i]
        for j in (j,):
            right = tracks[j]
            reason = veto(i, j)
            edge = {"i": i, "j": j, "left_uid": left.uid, "right_uid": right.uid,
                    "state": "rejected", "decision": "rejected", "probability": None,
                    "overlap": {"enabled":False}}
            edges.append(edge)
            if reason:
                state = "revoked" if reason == "revoked_by_review" else "rejected" if reason == "same_camera_disabled" else "conflict"
                edge.update(state=state, reason=reason)
                continue
            edge.update(pair_score(i, j))
            seam = recovery(i, j)
            if seam:
                if seam == 'occlusion_gap_recovery':
                    edge['continuity_evidence'] = occlusion_evidence(left,right,edge['appearance_components']['mean'])
                edge.update(conflict=None, continuity=seam, state="candidate", decision="deferred",
                            reason="verified_same_view_continuity")
                continue
            same_view = left.key[0] == right.key[0]
            if same_view and settings.recover_same_view:
                edge["reason"] = "no_same_view_continuity"
                continue
            if not same_view and settings.cross_camera_mean is not None:
                parts = edge["appearance_components"]
                sufficient = lambda t: (t.quality >= settings.cross_camera_quality
                                        and (len(t.usable_samples) >= 2 or t.quality >= .75))
                if not all(sufficient(t) for t in (left, right)):
                    edge["reason"] = "insufficient_cross_camera_crop_quality"
                    continue
                mean, gallery = parts.get('mean', -1.), parts.get('gallery_top3', -1.)
                # Repeated gallery evidence may support a sequential passage;
                # uncalibrated simultaneous views retain the stricter gate.
                sequential = (left.interval[1] < right.interval[0] or right.interval[1] < left.interval[0])
                repeated_support = (sequential and mean >= settings.cross_camera_gallery
                                    and gallery >= settings.cross_camera_mean)
                if not repeated_support and (mean < settings.cross_camera_mean
                        or gallery < settings.cross_camera_gallery):
                    edge["reason"] = "weak_cross_camera_reid"
                    continue
            similarity = edge["similarity"]
            independent_geometry = edge["sync_samples"] >= 2 and edge["spatial"] <= .25
            long_gap = cfg.use_time and edge["gap_s"] > cfg.max_gap_seconds
            gap_fraction = min(1.,edge['gap_s']/max(cfg.max_gap_seconds,1e-9)) if cfg.use_time else 0.
            reentry_floor = (settings.unsupported_similarity + gap_fraction *
                             (settings.reentry_similarity-settings.unsupported_similarity))
            if edge["cost"] is None or edge["reliability"] < settings.min_reliability:
                edge["reason"] = "insufficient_evidence"
            elif not independent_geometry and (similarity is None or similarity < cfg.min_similarity):
                edge["reason"] = "weak_or_missing_appearance"
            elif (not independent_geometry and edge["topology"] in {"unknown", "outside_measured_route"}
                  and similarity < settings.unsupported_similarity):
                edge["reason"] = "unconfirmed_without_topology"
            elif long_gap and (similarity is None or similarity < settings.reentry_similarity):
                edge["reason"] = "unconfirmed_reentry"
            elif (settings.separate_hypotheses and edge['topology']=='unknown'
                  and edge['gap_s']>0 and similarity < reentry_floor):
                edge.update(state='provisional',decision='deferred',reason='insufficient_reentry_support',
                            required_similarity=reentry_floor)
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
        i, j = edge['i'], edge['j']
        rivals = [other for other in continuity_edges if other is not edge
                  and ({i,j} & {other['i'],other['j']})
                  and any(veto(a,b) for a in {i,j}-{other['i'],other['j']}
                          for b in {other['i'],other['j']}-{i,j})]
        if any(abs(other['appearance_components']['mean']-edge['appearance_components']['mean'])
               < settings.margin for other in rivals):
            edge.update(state='rejected', decision='deferred', reason='ambiguous_local_continuity')
            continue
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
        if settings.overlap_matching_enabled and edge.get('topology') == 'overlap':
            continue
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

    if settings.overlap_matching_enabled and cfg.use_time and cfg.transition_model:
        from .overlap_matching import match_overlaps, synchronous_score
        lookup = {(e['i'],e['j']):e for e in edges}
        matches = match_overlaps(tracks,cfg.transition_model.overlap_pairs,
            lambda i,j: pair_score(min(i,j),max(i,j)),veto,settings,
            no_match_similarity=cfg.min_similarity,margin=max(.02,settings.margin),
            window_score_fn=lambda i,j,start,end:synchronous_score(tracks[i],tracks[j],
                pair_score(min(i,j),max(i,j)),start,end))
        for match in matches:
            i,j = sorted((match['i'],match['j']))
            edge = lookup[i,j]
            edge['initial_reason'] = edge['reason']
            edge['overlap'] = dict(match,enabled=True)
            edge['evidence_sources'] = ['appearance','temporal_assignment']
            edge.update(state='provisional',decision='deferred',reason='overlap_ambiguous' if match['ambiguous'] else 'overlap_mutual_best')
            p = edge['appearance_components']
            strong = (p.get('mean',-1.) >= (settings.cross_camera_mean or .72)
                      and p.get('gallery_top3',-1.) >= settings.cross_camera_gallery)
            temporal_strong = (match.get('independent_observations',0) >= 3
                               and (match.get('synchronous_mean') or -1.) >= .72
                               and match.get('temporal_agreement',0.) >= .8)
            a,b = root(i),root(j)
            if ((strong or temporal_strong) and not match['ambiguous'] and i in anchors and j in anchors
                    and edge['cost'] <= cfg.max_cost and compatible(a,b,route_support=temporal_strong)):
                if a != b:
                    unite(a,b)
                edge.update(state='confirmed',decision='accepted',stage='overlap')

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
        if not settings.separate_hypotheses:
            unite(group, i)
        trusted.discard(i)
        links[0].update(state="provisional", decision="deferred" if settings.separate_hypotheses else "accepted", reason="gallery_and_edges",
                        stage="attachment", support=support, supporting_cameras=len(links))

    # A weak intermediate view needs two already confirmed cameras, a
    # measured route to each, and no incompatible competing component.
    if settings.cross_camera_mean is not None:
        for i in sorted(parent, key=lambda k: tracks[k].uid):
            if root(i) != i or len(members[i]) != 1:
                continue
            proposals = []
            for group in list(members):
                if group == i or not compatible(i, group, route_support=True):
                    continue
                support = {}
                for j in members[group] & trusted:
                    if tracks[i].key[0] == tracks[j].key[0]:
                        continue
                    edge = next((e for e in edges if (e['i'], e['j']) == (min(i,j),max(i,j))), None)
                    if not edge or edge.get('cost') is None:
                        continue
                    p = edge['appearance_components']
                    allowed_topology = {'overlap', 'measured', 'unknown'} if settings.separate_hypotheses else {'overlap', 'measured'}
                    if (edge['topology'] in allowed_topology and edge['gap_s'] <= 10.
                            and tracks[i].quality >= settings.cross_camera_quality
                            and p.get('mean',-1.) >= .62 and p.get('gallery_top3',-1.) >= .55
                            and edge['similarity'] >= cfg.min_similarity and edge['cost'] <= cfg.max_cost):
                        camera = tracks[j].key[0]
                        if camera not in support or edge['cost'] < support[camera]['cost']:
                            support[camera] = edge
                if len(support) >= 2:
                    proposals.append((float(np.mean([e['cost'] for e in support.values()])), group, support))
            proposals.sort(key=lambda p: (p[0], p[1]))
            if not proposals or (len(proposals)>1 and proposals[1][0]-proposals[0][0] < settings.margin):
                continue
            _, group, support = proposals[0]
            unite(group, i)
            for edge in support.values():
                edge.update(state='confirmed' if settings.separate_hypotheses else 'provisional', decision='accepted', reason='two_camera_route_support',
                            stage='route_attachment', supporting_cameras=len(support))

    groups = sorted((sorted(group) for group in members.values()),
                    key=lambda group: min((tracks[i].interval[0], tracks[i].uid) for i in group))
    if settings.calibration_path:
        from .probability import predict
        calibration = json.loads(__import__('pathlib').Path(settings.calibration_path).read_text(encoding='utf-8'))
        for edge in edges:
            if edge.get('cost') is not None:
                edge['probability'] = predict(edge['cost'], calibration)
    if settings.optimizer == 'correlation':
        from .correlation import solve
        if not settings.calibration_path:
            raise ValueError('Correlation optimizer requires a supervised calibration model')
        probabilities={(e['i'],e['j']):e['probability'] for e in edges
                       if e.get('probability') is not None and (e['i'],e['j']) in candidates}
        conflicts=[(i,j) for i in range(len(tracks)) for j in range(i+1,len(tracks)) if veto(i,j)]
        groups,_ = solve(len(tracks),probabilities,conflicts)
        owners={i:k for k,g in enumerate(groups) for i in g}
        for edge in edges:
            if edge.get('probability') is not None and (edge['i'],edge['j']) in candidates:
                accepted=owners[edge['i']]==owners[edge['j']]
                edge.update(state='confirmed' if accepted else 'rejected',decision='accepted' if accepted else 'rejected',reason='correlation_optimizer')
    elif settings.optimizer != 'agglomerative':
        raise ValueError('Unknown graph optimizer')
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
    artifacts = [tracks[g[0]].row['artifact_candidate'] | {'tracklet':tracks[g[0]].uid}
                 for g in groups if len(g) == 1 and tracks[g[0]].row.get('artifact_candidate')]
    groups = [g for g in groups if not (len(g) == 1 and tracks[g[0]].row.get('artifact_candidate'))]
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
    if settings.identity_bank_path:
        from .identity_bank import update_bank
        model_scope=hashlib.sha256(json.dumps(manifest.get('models'),sort_keys=True).encode()).hexdigest()
        state['persistent_person_ids']=update_bank(settings.identity_bank_path,model_scope,scope,tracks,groups,edges,mapping)
    state["config"] = {"geometry": cfg.use_geometry, "settings": asdict(settings),
                       "weights": [cfg.appearance_weight, cfg.spatial_weight, cfg.temporal_weight],
                       "camera_fps": cfg.camera_fps, "camera_offsets_s": cfg.camera_offsets_s,
                       "max_cost": cfg.max_cost, "min_similarity": cfg.min_similarity,
                       "long_gap_s": cfg.max_gap_seconds}
    state["local_corrections"] = {"switches": local_diagnostics, "duplicates": duplicates}
    state["short_unconfirmed"] = short
    state['artifact_candidates'] = artifacts
    (run_dir/'provisional_links.json').write_text(json.dumps([e for e in edges if e['state']=='provisional'],ensure_ascii=False),encoding='utf-8')
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
    from .episodes import export_episodes
    export_episodes(out_csv, {t.key[0]: t.fps for t in tracks}, cfg.camera_offsets_s or {}, cfg.max_gap_seconds)
    return tracklets, mapping, edges, out_csv, out_txt
