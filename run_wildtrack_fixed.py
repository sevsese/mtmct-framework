#!/usr/bin/env python3
"""One-command WILDTRACK experiment with corrected ReID-aware local tracking.

Run from the root of the existing MTMCT project:
    python run_wildtrack_fixed.py

It reads local WILDTRACK images, runs YOLO + OSNet, makes local/global IDs,
exports a 4-camera mosaic and compares GT metrics with the old baseline.
No WILDTRACK labels are used during tracking or association.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path


def args_parser():
    p = argparse.ArgumentParser(description="One-command WILDTRACK v3: YOLO + ReID-aware tracking + GT")
    p.add_argument("--root", type=Path, default=Path("data/WILDTRACK"),
                   help="Dataset root containing Image_subsets and annotations_positions")
    p.add_argument("--out", type=Path, default=Path("runs/wildtrack_4cam_40f_reid_v3"))
    p.add_argument("--baseline", type=Path, default=Path("runs/wildtrack_4cam_40f_v2/metrics.json"))
    p.add_argument("--cameras", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--max-frames", type=int, default=40, help="Annotated moments, not raw suffix")
    p.add_argument("--model", default="yolo26l.pt")
    p.add_argument("--reid-model", default="osnet_ibn_x1_0")
    p.add_argument("--reid-weights", type=str, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imgsz", type=int, default=1536)
    p.add_argument("--conf", type=float, default=0.10, help="Raw YOLO confidence")
    p.add_argument("--new-track-conf", type=float, default=0.12)
    p.add_argument("--max-missed", type=int, default=2, help="Annotated frames lost, not raw suffix")
    p.add_argument("--match-cos", type=float, default=0.72)
    p.add_argument("--lost-cos", type=float, default=0.82)
    p.add_argument("--max-motion", type=float, default=1.8, help="BBox heights per annotated frame")
    p.add_argument("--global-sim", type=float, default=0.84)
    p.add_argument("--global-max-cost", type=float, default=0.08)
    p.add_argument("--reid-batch", type=int, default=32)
    p.add_argument("--scale", type=float, default=0.40)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--check-only", action="store_true", help="Verify dataset without GPU inference")
    p.add_argument("--overwrite", action="store_true", help="Replace this run's known output files")
    return p


def _load_json(path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _metrics_to_percent(report):
    if not report:
        return None
    det = report.get("detection", {})
    tp, fp, fn = [int(det.get(key,0)) for key in ("TP", "FP", "FN")]
    return {"global_idf1":round(float(report["global"]["IDF1"])*100,2),
            "detection_precision":round(100*tp/max(1,tp+fp),2),
            "detection_recall":round(100*tp/max(1,tp+fn),2),
            "detection_tp":tp,"detection_fp":fp,"detection_fn":fn,
            "predictions":int(report["global"].get("pred_count",0))}


def _raw_as_predictions(raw_file: Path, pred_file: Path):
    """Use unique dummy IDs for raw boxes; score only the detection fields."""
    with raw_file.open(newline="", encoding="utf-8") as inp, \
         pred_file.open("w",newline="",encoding="utf-8") as out:
        reader = csv.DictReader(inp)
        fieldnames = ["camera_id","frame_id","global_id","x1","y1","x2","y2"]
        writer = csv.DictWriter(out, fieldnames=fieldnames); writer.writeheader()
        for idx, r in enumerate(reader, start=1):
            writer.writerow({k:r[k] for k in ("camera_id","frame_id","x1","y1","x2","y2")} |
                            {"global_id":idx})


def _print_comparison(baseline, current, raw):
    print("\n" + "="*70)
    print("QUALITY COMPARISON (custom WILDTRACK bbox IoU=0.50; NOT official WILDTRACK metric)")
    print("Metric                     old v2         new v3         delta")
    print("-"*70)
    for key, label in (("global_idf1","Global IDF1"),
                       ("detection_precision","Detection precision"),
                       ("detection_recall","Detection recall")):
        old = baseline.get(key) if baseline else None
        new = current.get(key)
        old_s = f"{old:.2f}%" if old is not None else "N/A"
        diff = f"{new-old:+.2f} pp" if old is not None else "N/A"
        print(f"{label:26s} {old_s:>10s} {new:>12.2f}% {diff:>13s}")
    if raw:
        print(f"Raw YOLO detector: precision={raw['detection_precision']:.2f}%, "
              f"recall={raw['detection_recall']:.2f}% (without tracking)")
    print("="*70)


def main(argv=None):
    p = args_parser()
    args = p.parse_args(argv)
    from mtmct.wildtrack import frame_sources, select_frame_ids
    if args.max_frames < 0 or args.imgsz <= 0 or args.reid_batch <= 0:
        p.error("--max-frames must be >= 0, --imgsz/--reid-batch must be > 0")
    if args.max_missed < 0 or not (0 <= args.conf <= 1) or not (0 <= args.new_track_conf <= 1):
        p.error("invalid tracking thresholds")
    if not (0 <= args.match_cos <= 1 and 0 <= args.lost_cos <= 1 and
            0 <= args.global_sim <= 1 and 0 <= args.global_max_cost <= 1):
        p.error("cosine and cost thresholds must be in [0,1]")
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        p.error(f"WILDTRACK not found: {root}. Supply --root PATH.")
    cameras = sorted(set(args.cameras))
    sources = frame_sources(root, cameras)
    selected = select_frame_ids(root, cameras, args.max_frames)
    print(f"WILDTRACK: {root}")
    print(f"Cameras: {cameras}; annotated images: {len(selected)}; "
          f"raw image IDs: {selected[:3]} ... {selected[-3:]}")
    if args.check_only:
        return {"selected_frame_ids":selected,"cameras":cameras,
                "sources":{str(k):str(v) for k,v in sources.items()}}

    out = args.out
    output_names = ["raw_detections.csv","observations.csv","local_tracklets.json",
                    "global_tracks_2d.csv","global_ids.json","association_edges.json",
                    "mosaic.mp4","metrics.json","raw_detection_metrics.json",
                    "performance.json","observation_embeddings.npz","comparison.json",
                    "raw_eval_predictions.csv","config.yaml","manifest.json"]
    if not args.overwrite and any((out/name).exists() for name in output_names):
        p.error(f"Output directory {out} already has a run. Use --overwrite or choose --out.")
    out.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in output_names:
            (out / name).unlink(missing_ok=True)
    from mtmct.cache import write_config
    from mtmct.tracking_reid import MatchConfig, run_wildtrack_local
    from mtmct.association import Config
    from mtmct.link import run as global_associate
    from mtmct.wildtrack import evaluate
    from mtmct.visualize import render

    match_cfg = MatchConfig(max_missed=args.max_missed,
                            max_motion_heights=args.max_motion,
                            match_cos=args.match_cos,
                            lost_cos=args.lost_cos,
                            min_new_conf=args.new_track_conf)
    write_config(out, {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()} |
                 {"root":str(root),"selected_frame_ids":selected,
                  "tracker":"reid_motion_v1", "annotated_fps":2.0,
                  "geometry":False,"temporal_association":False})
    print("\n[1/4] YOLO raw detections + OSNet + ReID-aware local tracking", flush=True)
    local_stats = run_wildtrack_local(root,out,cameras,selected,model_name=args.model,
                     reid_model=args.reid_model,reid_weights=args.reid_weights,
                     device=args.device,imgsz=args.imgsz,conf=args.conf,
                     match_cfg=match_cfg,reid_batch=args.reid_batch)
    print("\n[2/4] Global association: constrained graph", flush=True)
    # WILDTRACK filenames step by 5; t = raw_frame/10 (2 annotated FPS).
    # Geometry and time gating disabled because the camera network is not calibrated.
    cfg = Config(appearance_weight=1.,spatial_weight=0.,temporal_weight=0.,
                 max_cost=args.global_max_cost,min_similarity=args.global_sim,
                 fps=10.,use_geometry=False,use_time=False)
    tracks, mapping, edges, predictions, _ = global_associate(
        out, cfg, matcher="graph", allow_same_camera_fragments=True,
        memory=False,correct_identities=False,repair_occlusions=False)
    print("Global IDs:",len(set(mapping.values())), "local tracklets:",len(tracks),
          "accepted edges:",sum(x.get('decision')=='accepted' for x in edges))
    print("\n[3/4] Ground-truth metrics + raw detector audit", flush=True)
    evaluated_frames = set(selected)
    report = evaluate(predictions,root,cameras,threshold=0.5,frame_ids=evaluated_frames)
    (out/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    _raw_as_predictions(out/"raw_detections.csv",out/"raw_eval_predictions.csv")
    raw_report = evaluate(out/"raw_eval_predictions.csv",root,cameras,threshold=0.5,
                          frame_ids=evaluated_frames)
    (out/"raw_detection_metrics.json").write_text(
        json.dumps({"detection":raw_report["detection"],"note":"Raw YOLO detection-only evaluation; dummy identity labels"},
                   ensure_ascii=False,indent=2),encoding="utf-8")
    baseline_path = args.baseline
    if not baseline_path.is_file():
        bundled = Path(__file__).with_name("baseline_wildtrack_v2_metrics.json")
        if bundled.is_file():
            print(f"Baseline metrics not found at {baseline_path}; using bundled v2 metrics: {bundled}")
            baseline_path = bundled
    previous_report = _load_json(baseline_path)
    if previous_report and (previous_report.get("cameras") != cameras or
                            previous_report.get("frame_ids") != selected):
        print("WARNING: baseline has different cameras/annotated frames; comparison disabled")
        previous_report = None
    before,after,raw = (_metrics_to_percent(x) for x in (previous_report,report,raw_report))
    summary = {"baseline_metrics_path":str(baseline_path) if previous_report else None,
               "old":before,"new":after,"raw_yolo":raw,
               "delta_pp":{k:round(after[k]-before[k],2) for k in
                           ["global_idf1","detection_precision","detection_recall"]} if before else None,
               "local":local_stats,"camera_ids":cameras,"frame_ids":selected,
               "protocol":"custom WILDTRACK bbox IoU 0.5, not official metric",
               "no_gt_leakage":True}
    (out/"comparison.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    _print_comparison(before,after,raw)
    if not args.no_video:
        print("\n[4/4] Rendering 4-camera video", flush=True)
        render(root,predictions,out/"mosaic.mp4",max_frames=0,stride=1,
               scale=args.scale,cameras=cameras,camera_sources=sources,
               fps_fallback=2.0,frame_ids=evaluated_frames)
    else:
        print("\n[4/4] Video disabled (--no-video)")
    print("\nDONE. Results:",out.resolve())
    for key in ("mosaic.mp4","comparison.json","metrics.json","raw_detection_metrics.json",
                "raw_detections.csv","global_tracks_2d.csv"):
        path = out/key
        if path.exists():
            print(" ",path)
    return summary


if __name__ == "__main__":
    main()
