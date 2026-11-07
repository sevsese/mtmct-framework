"""Output per-camera 2D predictions and a heuristic person-only 3D baseline.

The official Track-1 3D submission needs accurate 3D cuboids for ALL classes.
This intentionally crude 3D conversion is illustrative, not competitive.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median

FIELDS = ["camera_id","frame_id","local_id","x1","y1","x2","y2",
          "confidence","world_x","world_y"]


def export_predictions(observations_path: str | Path,
                       mapping_path: str | Path,
                       out_dir: str | Path,
                       scene_id: int = 0,
                       height: float = 1.7,
                       width: float = 0.6,
                       length: float = 0.6,
                       excluded_keys: set[tuple[int, int]] | None = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, encoding="utf-8") as f:
        mp = {(int(r["camera_id"]),int(r["local_id"])):int(r["global_id"])
              for r in json.load(f)}
    combined = defaultdict(list)
    out_csv = out / "global_tracks_2d.csv"
    with open(observations_path, newline="", encoding="utf-8") as fi, \
         open(out_csv, "w", newline="",encoding="utf-8") as fo:
        reader = csv.DictReader(fi)
        optional = ["timestamp_s"] if "timestamp_s" in (reader.fieldnames or ()) else []
        writer = csv.DictWriter(fo, fieldnames=FIELDS+optional+["global_id"])
        writer.writeheader()
        for r in reader:
            key=(int(r["camera_id"]),int(r["local_id"]))
            if key not in mp or (excluded_keys and key in excluded_keys):
                continue
            g=mp[key]
            writer.writerow({**r,"global_id":g})
            if r.get("world_x", "") != "" and r.get("world_y", "") != "":
                combined[(int(r["frame_id"]),g)].append(
                    (float(r["world_x"]),float(r["world_y"])))
    out_txt = out / "track1_person_approx.txt"
    with open(out_txt,"w",encoding="utf-8") as f:
        for (frame,g),pts in sorted(combined.items()):
            x=median(p[0] for p in pts)
            y=median(p[1] for p in pts)
            # 11 fields. Person class=0; world axis Z is vertical.
            f.write(f"{scene_id} 0 {g} {frame} {x:.2f} {y:.2f} {height/2:.2f} "
                    f"{width:.2f} {length:.2f} {height:.2f} 0.00\n")
    return out_csv, out_txt
