"""Ablation of plan 2.0 association mechanisms on a frozen feature cache.

Re-runs only the association (no YOLO/OSNet) on copies of a saved run with
four configurations:
  1. baseline  - previous v2 behavior (all new flags off)
  2. temporal  - early/middle/late temporal galleries in scoring
  3. overlap   - mutual matching on verified overlapping camera pairs
  4. full      - all mechanisms together

Example:
  python scripts/association_ablation.py \
      --source runs/hackaton_5cam_60s_corridor_v13 \
      --out runs/ablation_hackaton_60s_plan_v1 \
      --cases configs/ablation_cases_hackaton_60s.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtmct.association import Config
from mtmct.identity_v2 import Settings
from mtmct.link import run as link_run
from mtmct.time_sync import CameraTransitionModel, parse_offsets

ARTIFACTS = ("manifest.json", "local_tracklets.json", "observations.csv", "performance.json")

BASE_SETTINGS = dict(min_lifetime_s=1., cross_camera_mean=.72, cross_camera_gallery=.67,
                     recover_same_view=True, gallery="max")

VARIANTS = {
    "1_baseline": {},
    "2_temporal": {"temporal_gallery_enabled": True},
    "3_overlap": {"overlap_matching_enabled": True, "overlap_bonus_enabled": True,
                  "mutual_matching_enabled": True},
    "4_full": {"temporal_gallery_enabled": True, "overlap_matching_enabled": True,
               "overlap_bonus_enabled": True, "mutual_matching_enabled": True, "separate_hypotheses": True},
}


def evaluate_cases(mapping, hypotheses, cases):
    results = []
    for case in cases:
        first, second = (tuple(key) for key in case["pair"])
        gid_first, gid_second = mapping.get(first), mapping.get(second)
        together = gid_first is not None and gid_first == gid_second
        hyp = next((e for e in hypotheses if _edge_matches(e, first, second)), None)
        results.append({"name": case["name"], "expect": case["expect"],
                        "same_global_id": together,
                        "global_ids": [gid_first, gid_second],
                        "hypothesis": bool(hyp),
                        "hypothesis_reason": hyp.get("reason") if hyp else None})
    return results


def _edge_matches(edge, first, second):
    def key(uid):
        parts = str(uid).split(":")
        return (int(parts[0]), int(parts[1])) if len(parts) >= 2 else None
    return {key(edge["left_uid"]), key(edge["right_uid"])} == {first, second}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Finished run with features")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--topology", type=Path, default=Path("configs/camera_topology.json"))
    parser.add_argument("--cases", type=Path, help="JSON list of {name, pair, expect}")
    parser.add_argument("--camera-offset", action="append", default=[], metavar="CAMERA=SECONDS")
    parser.add_argument("--min-observations", type=int, default=5)
    args = parser.parse_args()

    source = args.source
    for name in ARTIFACTS:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    tracking = manifest.get("tracking_config", {})
    camera_fps = {int(c): float(f) for c, f in tracking.get("source_fps", {}).items()}
    offsets = parse_offsets(args.camera_offset)

    cases = json.loads(args.cases.read_text(encoding="utf-8")) if args.cases else []
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"source": str(source), "topology": str(args.topology), "variants": {}}
    for variant, flags in VARIANTS.items():
        folder = args.out / variant
        folder.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACTS:
            shutil.copyfile(source / name, folder / name)
        cfg = Config(appearance_weight=.92, spatial_weight=0., temporal_weight=.08,
                     min_similarity=.45, max_cost=.40, max_gap_seconds=60.0,
                     max_interior_gap_seconds=60.0,
                     camera_fps=camera_fps, camera_offsets_s=offsets,
                     transition_model=CameraTransitionModel.from_json(args.topology),
                     gallery_weight=.25, part_weight=.30, use_geometry=False, use_time=True)
        settings = Settings(**{**BASE_SETTINGS, **flags})
        _, mapping, edges, _, _ = link_run(folder, cfg, matcher="v2",
                                           allow_same_camera_fragments=True,
                                           v2_settings=settings, repair_occlusions=True,
                                           min_observations=args.min_observations)
        hypotheses = json.loads((folder / "provisional_links.json").read_text(encoding='utf-8'))
        states = Counter((e.get("state"), e.get("reason")) for e in edges)
        report["variants"][variant] = {
            "flags": flags,
            "global_ids": len(set(mapping.values())),
            "edge_states": {f"{state}:{reason}": count for (state, reason), count
                            in sorted(states.items(), key=lambda item: str(item[0]))},
            "hypotheses": len(hypotheses),
            "cases": evaluate_cases(mapping, hypotheses, cases),
        }
        print(f"[{variant}] global IDs: {report['variants'][variant]['global_ids']}, "
              f"hypotheses: {len(hypotheses)}")
    (args.out / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Report:", args.out / "comparison.json")


if __name__ == "__main__":
    main()
