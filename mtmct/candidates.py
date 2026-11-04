"""Inspect best cross-camera ReID matches without looking at ground-truth IDs."""
from __future__ import annotations
import argparse
from pathlib import Path
from .association import Config, load_tracklets, pair_cost, unit


def rank(run_dir: Path, max_cost: float, limit: int, use_geometry: bool,
         min_similarity: float = 0.80):
    import numpy as np
    ts = load_tracklets(run_dir / 'local_tracklets.json')
    cfg = Config(max_cost=max_cost, min_similarity=min_similarity,
                 use_geometry=use_geometry, spatial_weight=0.2 if use_geometry else 0.0)
    pairs = []
    for i, a in enumerate(ts):
        for b in ts[i+1:]:
            if a['camera_id'] == b['camera_id']:
                continue
            score = pair_cost(a, b, cfg)
            if score is None:
                continue
            cosine = float(np.dot(unit(a['embedding']), unit(b['embedding'])))
            pairs.append((score, cosine, a, b))
    for score, cosine, a, b in sorted(pairs, key=lambda r:r[0])[:limit]:
        print(f"C={score:.3f}  cosine={cosine:.3f}   cam{a['camera_id']}:ID{a['local_id']}"
              f"  <->  cam{b['camera_id']}:ID{b['local_id']}"
              f"   frames {a['start_frame']}..{a['end_frame']} / {b['start_frame']}..{b['end_frame']}")
    print(f"Valid inter-camera candidates: {len(pairs)} | below max-cost {sum(x[0]<=max_cost for x in pairs)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--max-cost', type=float, default=0.35)
    p.add_argument('--min-similarity', type=float, default=0.80)
    p.add_argument('--limit', type=int, default=25)
    p.add_argument('--with-geometry', action='store_true')
    a=p.parse_args()
    rank(a.run_dir, a.max_cost, a.limit, a.with_geometry, a.min_similarity)

if __name__=='__main__':
    main()
