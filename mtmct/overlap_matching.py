"""Mutually exclusive assignment of simultaneous tracks on overlapping camera pairs.

Simultaneous visibility on a verified overlapping pair is a *candidate*, never
proof: the score matrix blends visual similarity with a bounded overlap bonus,
then solves a maximum-weight assignment with an explicit "no pair" option.
Nearly tied alternatives are returned flagged as ambiguous so the caller can
keep the uncertainty instead of forcing a merge.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def simultaneous_duration(a, b):
    """Time both tracks are observed together on the common clock."""
    return max(0., min(a.interval[1], b.interval[1]) - max(a.interval[0], b.interval[0]))


def overlap_bonus(duration_s, quality, settings):
    """Bounded bonus; scales with co-observation time and crop quality."""
    if not settings.overlap_bonus_enabled:
        return 0.
    return settings.overlap_bonus_weight * min(1., duration_s / 3.) * quality


def _match_window(tracks, overlap_pairs, score_fn, veto_fn, settings,
                   no_match_similarity=.45, margin=.02):
    """One consistent assignment per overlapping camera pair.

    score_fn(i, j) returns the scored edge dict for a track pair; veto_fn(i, j)
    returns a non-None reason for pairs that must not be matched. Returns a list
    of match records; records with ambiguous=True are near-ties and should be
    kept as open hypotheses, not merged.
    """
    matches = []
    for pair in sorted({tuple(sorted(map(int, p))) for p in overlap_pairs}):
        first, second = pair
        left = [i for i, t in enumerate(tracks) if t.key[0] == first]
        right = [j for j, t in enumerate(tracks) if t.key[0] == second]
        if not left or not right:
            continue
        scores = np.full((len(left), len(right)), -np.inf)
        durations = np.zeros((len(left), len(right)))
        similarities = np.full((len(left), len(right)), np.nan)
        for x, i in enumerate(left):
            for y, j in enumerate(right):
                tau = simultaneous_duration(tracks[i], tracks[j])
                if tau <= 0 or veto_fn(i, j):
                    continue
                similarity = score_fn(i, j).get("similarity")
                if similarity is None or similarity < no_match_similarity:
                    continue
                quality = float(np.sqrt(tracks[i].quality * tracks[j].quality))
                bonus = overlap_bonus(tau, quality, settings)
                scores[x, y] = similarity + bonus
                durations[x, y] = tau
                similarities[x, y] = similarity
        row_best = np.max(scores, axis=1)
        col_best = np.max(scores, axis=0)
        # Rows/columns without a single admissible pair would make the
        # assignment infeasible; they simply stay unmatched.
        finite_rows = np.where(np.isfinite(row_best))[0]
        finite_cols = np.where(np.isfinite(col_best))[0]
        if not len(finite_rows) or not len(finite_cols):
            continue
        # Hungarian on cost = -score, padded to a square matrix with dummy rows
        # and a dummy column. Any pair scoring below the no-match floor keeps
        # its row unmatched; real columns that find no partner are absorbed by
        # dummy rows, so the assignment is always feasible.
        sub = scores[np.ix_(finite_rows, finite_cols)]
        n_rows, n_cols = sub.shape
        cost = np.full((n_rows + n_cols, n_rows + n_cols), np.inf)
        cost[:n_rows, :n_cols] = np.where(np.isfinite(sub), -sub, np.inf)
        cost[n_rows:, n_cols:] = 0.
        np.fill_diagonal(cost[:n_rows, n_cols:], -no_match_similarity)
        np.fill_diagonal(cost[n_rows:, :n_cols], 0.)
        assigned_rows, assigned_cols = linear_sum_assignment(cost)
        for sx, sy in zip(assigned_rows, assigned_cols):
            if sx >= n_rows or sy >= n_cols or not np.isfinite(sub[sx, sy]):
                continue
            x, y = int(finite_rows[sx]), int(finite_cols[sy])
            i, j = left[x], right[y]
            rivals = [value for yy, value in enumerate(scores[x]) if yy != y and np.isfinite(value)]
            rivals += [value for xx, value in enumerate(scores[:, y]) if xx != x and np.isfinite(value)]
            rival = max([no_match_similarity, *rivals])
            ambiguous = bool(np.isfinite(rival) and scores[x, y] - rival < margin)
            mutual = bool(scores[x, y] >= row_best[x] and scores[x, y] >= col_best[y])
            if settings.mutual_matching_enabled and not mutual:
                continue
            matches.append({"i": i, "j": j, "left_uid": tracks[i].uid, "right_uid": tracks[j].uid,
                            "score": float(scores[x, y]), "similarity": float(similarities[x, y]),
                            "duration_s": float(durations[x, y]),
                            "bonus_applied": float(scores[x, y] - similarities[x, y]),
                            "mutual_best": mutual, "ambiguous": ambiguous})
    return matches


def match_overlaps(tracks, overlap_pairs, score_fn, veto_fn, settings,
                   no_match_similarity=.45, margin=.02):
    """Assignment only competes with people present in the same time interval."""
    results = {}
    for pair in overlap_pairs:
        indexes = [i for i,t in enumerate(tracks) if t.key[0] in pair]
        boundaries = sorted({v for i in indexes for v in tracks[i].interval})
        for start, end in zip(boundaries, boundaries[1:]):
            if end <= start:
                continue
            active = [i for i in indexes if tracks[i].interval[0] < end and tracks[i].interval[1] > start]
            matches = _match_window([tracks[i] for i in active], [pair],
                lambda i,j: score_fn(active[i],active[j]),
                lambda i,j: veto_fn(active[i],active[j]), settings, no_match_similarity, margin)
            for m in matches:
                i,j = active[m['i']],active[m['j']]
                key = tuple(sorted((i,j)))
                record = results.setdefault(key, dict(m, i=i, j=j, windows=[], duration_s=0.))
                record['windows'].append({'start_s':start,'end_s':end,'ambiguous':m['ambiguous']})
                record['duration_s'] += end-start
                record['ambiguous'] |= m['ambiguous']
    return list(results.values())
