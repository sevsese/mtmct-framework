"""Run WILDTRACK image subsets through the existing MTMCT pipeline.

The official archive is expected below ``data/WILDTRACK``.  This script does
not download 6.8 GB implicitly: download/extract it once, then use
``--check-only`` to validate the layout before spending GPU time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mtmct.association import Config
from mtmct.cache import write_config
from mtmct.link import run as link
from mtmct.tracking import run as track
from mtmct.wildtrack import ANNOTATED_FPS, evaluate, frame_sources, inspect_dataset, select_frame_ids


ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MTMCT on WILDTRACK annotated image moments")
    p.add_argument("--root", type=Path, default=ROOT / "data" / "WILDTRACK")
    p.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--max-frames", type=int, default=40, help="0 uses all annotated moments")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "wildtrack_4cam_40f")
    p.add_argument("--model", default="yolo26l.pt")
    p.add_argument("--reid-model", default="osnet_ibn_x1_0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imgsz", type=int, default=1536)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--check-only", action="store_true", help="Validate data only; do not start YOLO")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cameras = sorted(set(args.cameras))
    if not cameras or min(cameras) < 0 or max(cameras) > 6:
        raise ValueError("WILDTRACK camera IDs must be between 0 and 6")
    sources = frame_sources(args.root, cameras)
    frame_ids = select_frame_ids(args.root, cameras, args.max_frames)
    report = inspect_dataset(args.root)
    report.update({"selected_cameras": cameras, "selected_frame_ids": frame_ids,
                   "selected_frames": len(frame_ids)})
    if args.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output already exists: {out}. Choose another --out path.")
    out.mkdir(parents=True, exist_ok=True)
    write_config(out, {"dataset": "wildtrack", "root": str(args.root), "cameras": cameras,
                       "frame_ids": frame_ids, "annotated_fps": ANNOTATED_FPS,
                       "geometry": False, "temporal_association": False})
    # Frame IDs stay raw for evaluation.  ByteTrack sees consecutive calls, but
    # association deliberately has no timing model for this sparse benchmark.
    track(args.root, out, args.model, "osnet", None, args.device, 0, 1, 5,
          args.imgsz, args.conf, reid_model=args.reid_model, cameras=cameras,
          camera_sources=sources, fps_fallback=ANNOTATED_FPS, frame_ids=set(frame_ids))
    cfg = Config(appearance_weight=.95, spatial_weight=0., temporal_weight=0.,
                 min_similarity=.80, max_cost=.125, fps=ANNOTATED_FPS,
                 use_geometry=False, use_time=False)
    _, _, _, predictions, _ = link(out, cfg, memory=True, memory_similarity=.95)
    metrics = evaluate(predictions, args.root, cameras, frame_ids=set(frame_ids))
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    main()
