"""Compare association variants on the same frozen detector output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from .association import Config
from .cache import fingerprint, validate_cache
from .evaluate import eval_2d, save_report
from .identity_v2 import Settings
from .link import run


def benchmark(source, out, gt, start, end, cameras, galleries=("max",), features=None):
    manifest = validate_cache(source)
    tracking = manifest.get("tracking_config", {})
    performance = json.loads((source/"performance.json").read_text())
    if tracking.get("start_frame", 0) != start or tracking.get("max_frames") != end or (
        {p["camera_id"]: p["processed_frames"] for p in performance} != {c: end-start for c in cameras}
    ):
        raise ValueError("Benchmark interval does not match completed inference")
    out.mkdir(parents=True, exist_ok=True)
    artifacts = ("observations.csv", "local_tracklets.json", "manifest.json")
    provenance = {"interval": [start, end], "cameras": cameras,
                  "source": [fingerprint(source/name) for name in artifacts],
                  "ground_truth": fingerprint(gt),
                  "features": fingerprint(features) if features else None}
    (out/"experiment.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    variants = {"baseline_memory": {"memory": True}, **{
        f"v2_{gallery}": {"matcher": "v2", "allow_same_camera_fragments": True,
                          "v2_settings": Settings(gallery=gallery),
                          "v2_features": features}
        for gallery in galleries
    }}
    reports = {}
    for name, options in variants.items():
        folder = out/name
        folder.mkdir(exist_ok=True)
        for file in artifacts:
            shutil.copyfile(source/file, folder/file)
        cfg = Config(appearance_weight=.95, spatial_weight=0, temporal_weight=.05,
                     max_cost=.125, min_similarity=.8, use_geometry=False,
                     gallery_weight=.25 if name.startswith("v2_") else 0.)
        tick = time.perf_counter()
        _, _, _, predictions, _ = run(folder, cfg, **options)
        seconds = time.perf_counter() - tick
        report = eval_2d(predictions, gt, cameras, end, start_frame=start)
        report["association_seconds"] = seconds
        save_report(report, folder/"metrics.json")
        reports[name] = report
        print(name, report["global"]["IDF1"], report["pairwise_association"]["F1"],
              report["identity_diagnostics"], flush=True)
    (out/"results.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=900)
    p.add_argument("--cameras", type=int, nargs="+", default=[3, 15])
    p.add_argument("--gallery", choices=["mean", "max", "topk"], nargs="+", default=["max"])
    p.add_argument("--features", type=Path)
    a = p.parse_args()
    benchmark(a.source, a.out, a.gt, a.start_frame, a.end_frame, a.cameras, a.gallery, a.features)


if __name__ == "__main__":
    main()
