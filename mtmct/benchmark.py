"""Fixed ablations against 1920 + baseline memory, using immutable inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

from .association import Config
from .cache import fingerprint, validate_cache
from .evaluate import eval_2d, save_report
from .link import run


VARIANTS = {
    'baseline_memory': {},
    'constrained_memory': {'memory_mode': 'constrained'},
    'gallery_memory': {'memory_mode': 'gallery'},
    'adaptive': {'matcher': 'adaptive', 'memory': False},
    'adaptive_memory': {'matcher': 'adaptive', 'memory_mode': 'gallery'},
    'sustained_split': {'split_tracklets': True, 'memory_mode': 'constrained'},
    'repair_memory': {'repair_occlusions': True, 'memory_mode': 'constrained'},
    'full': {'matcher': 'adaptive', 'memory_mode': 'gallery', 'split_tracklets': True, 'repair_occlusions': True},
}


def benchmark(source: Path, out: Path, gt: Path, start: int, end: int,
              cameras: list[int], foreground: Path | None = None, selected: list[str] | None = None,
              tracklets_override: Path | None = None):
    manifest = validate_cache(source)
    if manifest.get('legacy'):
        raise ValueError('Benchmark requires a source manifest')
    performance = json.loads((source/'performance.json').read_text())
    if {p['camera_id']: p['processed_frames'] for p in performance} != {c: end-start for c in cameras}:
        raise ValueError('Incomplete inference or wrong evaluation range')
    if manifest['tracking_config'].get('start_frame', 0) != start or manifest['tracking_config']['max_frames'] != end:
        raise ValueError('Manifest interval mismatch')
    out.mkdir(parents=True, exist_ok=True)
    provenance = {'source': str(source.resolve()), 'observations': fingerprint(source/'observations.csv'),
                  'tracklets': fingerprint(source/'local_tracklets.json'),
                  'interval': [start, end], 'cameras': cameras,
                  'implementation': [fingerprint(p) for p in sorted(Path(__file__).parent.glob('*.py'))]}
    if tracklets_override:
        provenance['tracklets_override'] = fingerprint(tracklets_override)
    if foreground:
        provenance['foreground'] = fingerprint(foreground)
    (out/'experiment.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    variants = dict(VARIANTS)
    if foreground:
        variants['foreground_gallery'] = {'memory_mode': 'gallery'}
        variants['foreground_full'] = dict(VARIANTS['full'])
    if selected:
        variants = {name: variants[name] for name in dict.fromkeys(['baseline_memory', *selected])}
    results = {}
    for name, options in variants.items():
        folder = out/name
        folder.mkdir(exist_ok=True)
        shutil.copy2(source/'observations.csv', folder/'observations.csv')
        shutil.copy2(foreground if name.startswith('foreground_') else tracklets_override or source/'local_tracklets.json', folder/'local_tracklets.json')
        derived = {**manifest, 'derivation': {'source': str(source.resolve()), 'variant': name, 'options': options}}
        (folder/'manifest.json').write_text(json.dumps(derived, indent=2), encoding='utf-8')
        cfg = Config(appearance_weight=.95, spatial_weight=0, temporal_weight=.05,
                     max_cost=.125, min_similarity=.8, use_geometry=False)
        params = {'memory': True, **options}
        tick = time.perf_counter()
        *_, predictions, _ = run(folder, cfg, **params)
        seconds = time.perf_counter() - tick
        report = eval_2d(predictions, gt, cameras, end, start_frame=start)
        report['association_seconds'] = seconds
        save_report(report, folder/'metrics.json')
        results[name] = report
        print(name, 'IDF1', report['global']['IDF1'], 'pairwise F1', report['pairwise_association']['F1'], flush=True)
        (out/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (out/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variant', 'IDF1', 'delta_pp', 'recall', 'precision', 'pairwise_F1', 'association_seconds'])
        for name, r in results.items():
            writer.writerow([name, r['global']['IDF1'], 100*(r['global']['IDF1']-results['baseline_memory']['global']['IDF1']),
                             r['detection']['recall'], r['detection']['precision'], r['pairwise_association']['F1'], r['association_seconds']])
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gt', type=Path, required=True)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=900)
    p.add_argument('--cameras', nargs='+', type=int, default=[3, 15])
    p.add_argument('--foreground', type=Path)
    p.add_argument('--variants', nargs='+')
    p.add_argument('--tracklets', type=Path)
    a = p.parse_args()
    benchmark(a.source, a.out, a.gt, a.start_frame, a.max_frames, a.cameras, a.foreground, a.variants, a.tracklets)


if __name__ == '__main__':
    main()
