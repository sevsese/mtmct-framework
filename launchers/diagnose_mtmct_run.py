#!/usr/bin/env python3
"""Diagnose local-ID drift and appearance-based re-link candidates in MTMCT runs.

Usage:
    python diagnose_mtmct_run.py --input runs/my_run --output runs/diagnostics
    python diagnose_mtmct_run.py --input wildtrack_4cam_40f_v2.zip --output diagnostics

Only Python standard library is needed. This is a diagnostic; it does not modify
predictions or claim that ReID similarity alone establishes identity.
"""
from pathlib import Path as _LauncherPath
import sys as _launcher_sys
_launcher_sys.path.insert(0, str(_LauncherPath(__file__).resolve().parents[1]))
_launcher_sys.path.insert(0, str(_LauncherPath(__file__).resolve().parent))
import argparse
import collections
import csv
import io
import json
import math
from pathlib import Path
import zipfile

FILES = ("observations.csv", "local_tracklets.json", "global_ids.json")


def read_run(path):
    path = Path(path)
    if path.is_dir():
        def get(name):
            return (path / name).read_text(encoding="utf-8")
    elif path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            paths = {Path(name).name: name for name in archive.namelist()}
            missing = [name for name in FILES if name not in paths]
            if missing:
                raise ValueError(f"Missing files in ZIP: {missing}")
            return {name: archive.read(paths[name]).decode("utf-8") for name in FILES}
    else:
        raise ValueError(f"Not a run directory or ZIP file: {path}")
    return {name: get(name) for name in FILES}


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(x) * float(x) for x in b))
    return dot / (na * nb) if na > 1e-12 and nb > 1e-12 else None


def write_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def bbox(row):
    return [float(row[k]) for k in ("x1", "y1", "x2", "y2")]


def movement(box_a, box_b):
    ca_x = (box_a[0] + box_a[2]) / 2
    ca_y = (box_a[1] + box_a[3]) / 2
    cb_x = (box_b[0] + box_b[2]) / 2
    cb_y = (box_b[1] + box_b[3]) / 2
    height = max(1, (box_a[3] - box_a[1] + box_b[3] - box_b[1]) / 2)
    return math.hypot(ca_x - cb_x, ca_y - cb_y) / height


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Run folder or output ZIP")
    p.add_argument("--output", type=Path, required=True, help="Diagnostic output directory")
    p.add_argument("--frame-step", type=int, default=5, help="Raw frame number step (WILDTRACK: 5)")
    p.add_argument("--switch-cos", type=float, default=0.5, help="Within-local-ID mismatch flag")
    p.add_argument("--link-cos", type=float, default=0.85, help="Adjacent-frame re-link proposal")
    p.add_argument("--max-distance", type=float, default=1.5, help="Max center movement in bbox heights")
    args = p.parse_args()
    if args.frame_step < 1:
        p.error("--frame-step must be >=1")
    raw = read_run(args.input)
    observations = list(csv.DictReader(io.StringIO(raw["observations.csv"])))
    tracks = json.loads(raw["local_tracklets.json"])
    mapping = {(int(r["camera_id"]), int(r["local_id"])): int(r["global_id"])
               for r in json.loads(raw["global_ids.json"])}
    feat = {}
    quality = {}
    track_by_key = {}
    for t in tracks:
        key = (int(t["camera_id"]), int(t["local_id"]))
        track_by_key[key] = t
        for f, vector, q in zip(t.get("feature_frames", []), t.get("feature_vectors", []), t.get("feature_quality", [])):
            feat[key[0], key[1], int(f)] = vector
            quality[key[0], key[1], int(f)] = float(q)
    by_cam_frame = collections.defaultdict(list)
    obs_by_id = collections.defaultdict(list)
    for r in observations:
        cam, lid, frame = (int(r[k]) for k in ("camera_id", "local_id", "frame_id"))
        by_cam_frame[(cam, frame)].append(r)
        obs_by_id[(cam, lid)].append(r)

    switches = []
    for (cam, lid), rows in obs_by_id.items():
        rows.sort(key=lambda r: int(r["frame_id"]))
        obs_frames = [int(r["frame_id"]) for r in rows]
        sampled_frames = [f for f in obs_frames if (cam, lid, f) in feat]
        for old, new in zip(sampled_frames, sampled_frames[1:]):
            sim = cosine(feat[cam, lid, old], feat[cam, lid, new])
            if sim is None or sim >= args.switch_cos:
                continue
            switches.append({"camera_id": cam, "local_id": lid,
                             "global_id": mapping.get((cam, lid), ""),
                             "frame_a": old, "frame_b": new,
                             "gap_steps": round((new - old) / args.frame_step, 2),
                             "cosine": round(sim, 4),
                             "min_feature_quality": round(min(quality[cam, lid, old], quality[cam, lid, new]), 4)})

    links = []
    for (cam, frame), rows in by_cam_frame.items():
        next_rows = by_cam_frame.get((cam, frame + args.frame_step), [])
        for ra in rows:
            ida = int(ra["local_id"])
            va = feat.get((cam, ida, frame))
            if va is None:
                continue
            for rb in next_rows:
                idb = int(rb["local_id"])
                if ida == idb:
                    continue
                vb = feat.get((cam, idb, frame + args.frame_step))
                sim = cosine(va, vb) if vb is not None else None
                if sim is None or sim < args.link_cos:
                    continue
                norm_dist = movement(bbox(ra), bbox(rb))
                if norm_dist > args.max_distance:
                    continue
                links.append({"camera_id": cam, "frame_a": frame,
                              "frame_b": frame + args.frame_step,
                              "local_id_a": ida, "local_id_b": idb,
                              "global_id_a": mapping.get((cam, ida), ""),
                              "global_id_b": mapping.get((cam, idb), ""),
                              "cosine": round(sim, 4), "normalized_motion": round(norm_dist, 4)})

    switches.sort(key=lambda r: (r["cosine"], r["camera_id"], r["frame_a"]))
    links.sort(key=lambda r: (-r["cosine"], r["camera_id"], r["frame_a"]))
    camera_summary = []
    for cam in sorted({int(r["camera_id"]) for r in observations}):
        frames = sorted(f for c, f in by_cam_frame if c == cam)
        camera_summary.append({
            "camera_id": cam,
            "num_frames": len(frames),
            "num_predictions": sum(len(by_cam_frame[cam, f]) for f in frames),
            "num_local_tracklets": sum(t["camera_id"] == cam for t in tracks),
            "first_frame_detections": len(by_cam_frame[cam, frames[0]]),
            "second_frame_detections": len(by_cam_frame[cam, frames[1]]) if len(frames) >= 2 else None,
            "one_observation_tracklets": sum(len(rows) == 1 for (c, _), rows in obs_by_id.items() if c == cam),
        })
    result = {
        "observations": len(observations),
        "local_tracklets": len(tracks),
        "global_ids": len(set(mapping.values())),
        "distinct_global_multi_camera": sum(len({c for c, lid in keys}) > 1 for gid, keys in
                                           ((g, [k for k, v in mapping.items() if v == g]) for g in set(mapping.values()))),
        "saved_sample_embeddings": len(feat),
        "num_suspected_within_id_switches": len(switches),
        "num_adjacent_frame_cross_id_candidates": len(links),
        "cameras": camera_summary,
        "important": "Cosine and position are indicators, not confirmed GT; manually validate before merging."
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "suspected_switches.csv", switches,
              ["camera_id", "local_id", "global_id", "frame_a", "frame_b", "gap_steps", "cosine", "min_feature_quality"])
    write_csv(args.output / "adjacent_link_candidates.csv", links,
              ["camera_id", "frame_a", "frame_b", "local_id_a", "local_id_b", "global_id_a", "global_id_b", "cosine", "normalized_motion"])
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Outputs:", args.output.resolve())

if __name__ == "__main__":
    main()
