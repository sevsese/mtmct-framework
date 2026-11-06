"""Revisable, auditable identity graph; no learned parameters or GT input."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from .association import Config, groups_compatible, hard_compatible, pair_score
from .features import gallery_similarity
from .features import reliable_gallery
from .time_sync import track_times


class IdentityGraph:
    """Keep evidence edges so withdrawing an edge can actually split a component."""

    def __init__(self, tracklets: list[dict], cfg: Config):
        self.tracklets = tracklets
        self.cfg = cfg
        self.edges = []
        self.keys = [(int(t['camera_id']), int(t['local_id'])) for t in tracklets]
        if len(set(self.keys)) != len(self.keys):
            raise ValueError('Duplicate tracklet keys')

    def revoke(self, i: int, j: int, reason: str):
        for edge in self.edges:
            if {edge['i'], edge['j']} == {i, j}:
                edge.update(status='revoked', decision='rejected', reason=reason)

    def resolve(self):
        parent = list(range(len(self.keys)))
        members = {i: {i} for i in parent}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for edge in sorted(self.edges, key=lambda e: (e.get('cost', 1.), e['i'], e['j'])):
            if edge['status'] not in {'provisional', 'confirmed'}:
                continue
            a, b = find(edge['i']), find(edge['j'])
            if a == b:
                continue
            if not groups_compatible([self.tracklets[i] for i in members[a]],
                                     [self.tracklets[i] for i in members[b]], self.cfg):
                edge.update(status='revoked', decision='rejected', reason='component_conflict')
                continue
            parent[b] = a
            members[a].update(members.pop(b))
            edge.update(status='confirmed', decision='accepted')
        roots = sorted(members, key=lambda r: min((self.tracklets[i]['start_frame'], self.keys[i])
                                                 for i in members[r]))
        ids = {root: k + 1 for k, root in enumerate(roots)}
        return {key: ids[find(i)] for i, key in enumerate(self.keys)}


def associate_adaptive(tracklets: list[dict], cfg: Config, top_k: int = 10,
                       margin: float = 0.02, window_seconds: float = 30.):
    """Reliable-first matching with ambiguous competitors retained as candidates.

    Time windows bound candidate search; memory handles longer reappearances.
    Unknown geometry/time carries no invented negative evidence.
    """
    if top_k < 1 or margin < 0 or window_seconds <= 0:
        raise ValueError('top_k/window must be positive and margin nonnegative')
    graph = IdentityGraph(tracklets, cfg)
    candidates = []
    # ponytail: O(N^2) enumeration suits this scene; use time buckets for long recordings.
    for i, a in enumerate(tracklets):
        sa, ea = track_times(a, cfg.fps, cfg.camera_offsets_s)
        for j in range(i + 1, len(tracklets)):
            b = tracklets[j]
            sb, eb = track_times(b, cfg.fps, cfg.camera_offsets_s)
            if max(sa, sb) - min(ea, eb) > window_seconds:
                continue
            base = {'i': i, 'j': j, 'status': 'rejected', 'decision': 'rejected'}
            if not hard_compatible(a, b, cfg):
                graph.edges.append({**base, 'reason': 'hard_constraint'})
                continue
            similarity, reliability = gallery_similarity(a, b)
            if similarity is None or similarity < cfg.min_similarity:
                graph.edges.append({**base, 'reason': 'missing_or_weak_reid', 'similarity': similarity})
                continue
            # Penalize time only with a measured route; unknown isn't impossible.
            scoring = replace(cfg, max_gap_seconds=float('inf'),
                              temporal_weight=cfg.temporal_weight if cfg.transition_model else 0,
                              spatial_weight=cfg.spatial_weight if cfg.use_geometry and a.get('positions') and b.get('positions') else 0)
            score = pair_score(a, b, scoring)
            if score['cost'] is None:
                continue
            appearance_delta = (float(score['similarity']) - similarity) / 2
            appearance_fraction = scoring.appearance_weight / (scoring.appearance_weight + scoring.spatial_weight + scoring.temporal_weight)
            cost = max(0., float(score['cost']) + appearance_fraction * appearance_delta)
            if cost > cfg.max_cost:
                continue
            candidates.append({**base, 'similarity': similarity, 'reliability': reliability,
                               'cost': cost, 'reason': 'appearance_and_available_evidence'})
    neighbors = {i: [] for i in range(len(tracklets))}
    for edge in candidates:
        for endpoint in (edge['i'], edge['j']):
            neighbors[endpoint].append(edge)
    kept = set()
    for edges in neighbors.values():
        for edge in sorted(edges, key=lambda e: e['cost'])[:top_k]:
            kept.add((edge['i'], edge['j']))
    # Two passes: establish reliable identities before attaching weak tracklets.
    for edge in sorted(candidates, key=lambda e: (e['reliability'] < 0.4, e['cost'], e['i'], e['j'])):
        if (edge['i'], edge['j']) not in kept:
            continue
        competitors = []
        for endpoint, partner in ((edge['i'], edge['j']), (edge['j'], edge['i'])):
            for other in neighbors[endpoint]:
                target = other['j'] if other['i'] == endpoint else other['i']
                if target != partner and not hard_compatible(tracklets[partner], tracklets[target], cfg):
                    competitors.append(other['similarity'])
        required = margin * (2 - edge['reliability'])
        ambiguous = competitors and edge['similarity'] - max(competitors) < required
        weak = edge['reliability'] < 0.4 and edge['similarity'] < max(cfg.min_similarity, 0.9)
        if ambiguous or weak:
            edge.update(status='candidate', decision='deferred', reason='ambiguous' if ambiguous else 'low_reliability')
        else:
            edge.update(status='provisional', decision='accepted')
        graph.edges.append(edge)
    # resolve respects the reliability pass order via an explicit priority cost.
    for edge in graph.edges:
        if edge.get('reliability', 1.) < 0.4 and 'cost' in edge:
            edge['raw_cost'] = edge['cost']
            edge['cost'] += 1.
    mapping = graph.resolve()
    for edge in graph.edges:
        if 'raw_cost' in edge:
            edge['cost'] = edge.pop('raw_cost')
    return mapping, graph.edges


def save_identity_state(path: Path, tracklets: list[dict], mapping: dict, *,
                        scope: str, now_s: float, missing_after: float = 5.,
                        archive_after: float = 300., cfg: Config | None = None):
    """Persist stable IDs for known tracklet UIDs with aliases and lifecycle events.

    New, nonoverlapping batches must first be associated against prior records;
    persistence alone never invents a cross-batch ReID match.
    """
    cfg = cfg or Config()
    previous = json.loads(path.read_text()) if path.exists() else {'scope': scope, 'next_id': 1, 'groups': [], 'aliases': {}}
    if previous['scope'] != scope:
        raise ValueError('Memory scope mismatch: scene/model/time basis must match')
    owner = {uid: g['global_id'] for g in previous['groups'] for uid in g['tracklet_uids']}
    aliases = dict(previous.get('aliases', {}))
    grouped = {}
    for t in tracklets:
        gid = mapping[(int(t['camera_id']), int(t['local_id']))]
        grouped.setdefault(gid, []).append(t)
    used = set()
    records = []
    events = []
    for source_id, rows in sorted(grouped.items()):
        uids = [t.get('tracklet_uid', f"{scope}:{t['camera_id']}:{t['local_id']}:{t['start_frame']}") for t in rows]
        known = sorted({owner[u] for u in uids if u in owner} - used)
        gid = known[0] if known else previous['next_id']
        if not known:
            previous['next_id'] += 1
        used.add(gid)
        for old in known[1:]:
            aliases[str(old)] = gid
            events.append({'event': 'alias', 'from': old, 'to': gid})
        last = max(track_times(t, cfg.fps, cfg.camera_offsets_s)[1] for t in rows)
        status = 'archived' if now_s - last > archive_after else 'missing' if now_s - last > missing_after else 'confirmed' if len(rows) > 1 else 'tentative'
        old_record = next((g for g in previous['groups'] if g['global_id'] == gid), None)
        if old_record and (old_record['status'] in {'missing', 'archived'} or now_s - old_record['last_seen_s'] > missing_after) and last > old_record['last_seen_s'] and status in {'confirmed', 'tentative'}:
            events.append({'event': 'reactivated', 'global_id': gid})
        records.append({'global_id': gid, 'source_global_id': source_id, 'status': status,
                        'last_seen_s': last, 'tracklet_uids': uids,
                        'tracklets': [{k: v for k, v in t.items() if k in {
                            'camera_id', 'local_id', 'tracklet_uid', 'start_frame', 'end_frame',
                            'start_s', 'end_s', 'fps', 'positions', 'embedding', 'samples'}}
                            | {'tracklet_uid': uid, 'prototypes': reliable_gallery(t)[0],
                               'gallery_reliability': reliable_gallery(t)[1]}
                            for t, uid in zip(rows, uids)]})
    for old in previous['groups']:
        if old['global_id'] not in used and str(old['global_id']) not in aliases:
            old['status'] = 'archived' if now_s - old['last_seen_s'] > archive_after else 'missing'
            records.append(old)
    state = {**previous, 'groups': records, 'aliases': aliases, 'events': events}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(path)
    with path.with_suffix('.jsonl').open('a', encoding='utf-8') as stream:
        for event in events:
            stream.write(json.dumps({'time_s': now_s, **event}) + '\n')
    return state


def resume_identity_memory(path: Path, tracklets: list[dict], mapping: dict, *,
                           scope: str, now_s: float, cfg: Config, threshold: float = .9):
    """Reassociate an incoming batch with saved galleries, then keep stable IDs.

    Tracklet UIDs, not local IDs (which restart in each batch), identify records.
    The full histories participate in physical vetoes before any reactivation.
    """
    from .memory import apply_memory
    current = [dict(t, tracklet_uid=t.get('tracklet_uid',
               f"{scope}:{t['camera_id']}:{t['local_id']}:{t['start_frame']}")) for t in tracklets]
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous and previous['scope'] != scope:
        raise ValueError('Memory scope mismatch: scene/model/time basis must match')
    current_uids = {t['tracklet_uid'] for t in current}
    historical = []
    seed = {}
    next_group = 1
    if previous:
        for group in previous['groups']:
            for t in group.get('tracklets', []):
                if t['tracklet_uid'] not in current_uids:
                    historical.append((t, group['global_id']))
            next_group = max(next_group, group['global_id'] + 1)
    all_rows = []
    current_keys = []
    for t, gid in historical:
        synthetic = dict(t, local_id=len(all_rows) + 1)
        all_rows.append(synthetic)
        seed[(int(synthetic['camera_id']), synthetic['local_id'])] = gid
    for t in current:
        synthetic = dict(t, local_id=len(all_rows) + 1)
        all_rows.append(synthetic)
        key = (int(synthetic['camera_id']), synthetic['local_id'])
        current_keys.append(key)
        seed[key] = next_group + mapping[(int(t['camera_id']), int(t['local_id']))]
    resumed, _, decisions = apply_memory(all_rows, seed, cfg=cfg, mode='gallery',
                                         similarity_threshold=threshold)
    state = save_identity_state(path, all_rows, resumed, scope=scope, now_s=now_s, cfg=cfg)
    stable = {g['source_global_id']: g['global_id'] for g in state['groups'] if 'source_global_id' in g}
    output = {(int(t['camera_id']), int(t['local_id'])): stable[resumed[key]]
              for t, key in zip(current, current_keys)}
    return output, decisions
