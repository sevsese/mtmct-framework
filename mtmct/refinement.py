"""Optional offline tracklet split/merge with an auditable provenance map."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

from .features import normalize


def diagnose(tracklets: list[dict], split_similarity: float = 0.5,
             merge_similarity: float = 0.9, max_gap_frames: int = 90) -> dict:
    splits = []
    for index, tracklet in enumerate(tracklets):
        samples = sorted(zip(tracklet.get("feature_frames", []),
                             tracklet.get("feature_vectors", [])))
        for (left_frame, left), (right_frame, right) in zip(samples, samples[1:]):
            similarity = float(normalize(left) @ normalize(right))
            if similarity < split_similarity:
                splits.append({"index": index, "camera_id": tracklet["camera_id"],
                               "local_id": tracklet["local_id"], "split_frame": right_frame,
                               "similarity": round(similarity, 6)})
                break
    merges = []
    for left_index, left in enumerate(tracklets):
        for right_index in range(left_index + 1, len(tracklets)):
            right = tracklets[right_index]
            if int(left["camera_id"]) != int(right["camera_id"]):
                continue
            earlier, later = (left, right) if left["end_frame"] < right["start_frame"] else (right, left)
            gap = int(later["start_frame"]) - int(earlier["end_frame"]) - 1
            if gap < 0 or gap > max_gap_frames or not left.get("embedding") or not right.get("embedding"):
                continue
            similarity = float(normalize(left["embedding"]) @ normalize(right["embedding"]))
            if similarity >= merge_similarity:
                merges.append({"left": left_index, "right": right_index,
                               "camera_id": left["camera_id"], "gap_frames": gap,
                               "similarity": round(similarity, 6)})
    return {"splits": splits, "merges": merges}


def refine(tracklets: list[dict], *, auto_split: bool = False, auto_merge: bool = False,
           split_similarity: float = 0.5, merge_similarity: float = 0.9,
           max_gap_frames: int = 90):
    original = copy.deepcopy(tracklets)
    report = diagnose(original, split_similarity, merge_similarity, max_gap_frames)
    if not auto_split and not auto_merge:
        return original, report, []
    next_id = {}
    for row in original:
        camera = int(row["camera_id"])
        next_id[camera] = max(next_id.get(camera, 0), int(row["local_id"]) + 1)
    rows = []
    remap = []
    split_by_index = {item["index"]: item for item in report["splits"]} if auto_split else {}
    for index, row in enumerate(original):
        source = {"camera_id": int(row["camera_id"]), "local_id": int(row["local_id"]),
                  "start_frame": int(row["start_frame"]), "end_frame": int(row["end_frame"])}
        split = split_by_index.get(index)
        if not split:
            row["provenance"] = [source]
            rows.append(row)
            remap.append({**source, "new_local_id": int(row["local_id"])})
            continue
        frame = int(split["split_frame"])
        parts = []
        for start, end, local_id in ((source["start_frame"], frame - 1, source["local_id"]),
                                     (frame, source["end_frame"], next_id[source["camera_id"]])):
            part = copy.deepcopy(row)
            part["local_id"], part["start_frame"], part["end_frame"] = local_id, start, end
            selected = [sample_index for sample_index, sample_frame in enumerate(row.get("feature_frames", []))
                        if start <= sample_frame <= end]
            part["feature_frames"] = [row["feature_frames"][i] for i in selected]
            part["feature_vectors"] = [row["feature_vectors"][i] for i in selected]
            part["feature_quality"] = [row.get("feature_quality", [1] * len(row.get("feature_frames", [])))[i]
                                       for i in selected]
            if part["feature_vectors"]:
                part["embedding"] = normalize(np.mean([normalize(v) for v in part["feature_vectors"]], axis=0)).tolist()
            part["provenance"] = [{**source, "segment_start": start, "segment_end": end}]
            parts.append(part)
            remap.append({**source, "start_frame": start, "end_frame": end,
                          "new_local_id": local_id})
        next_id[source["camera_id"]] += 1
        rows.extend(parts)

    if auto_merge:
        changed = True
        while changed:
            changed = False
            current = diagnose(rows, split_similarity, merge_similarity, max_gap_frames)["merges"]
            if not current:
                break
            merge = current[0]
            left, right = rows[merge["left"]], rows[merge["right"]]
            primary, secondary = (left, right) if int(left["local_id"]) < int(right["local_id"]) else (right, left)
            primary["start_frame"] = min(int(left["start_frame"]), int(right["start_frame"]))
            primary["end_frame"] = max(int(left["end_frame"]), int(right["end_frame"]))
            primary["embedding"] = normalize(np.mean([normalize(left["embedding"]), normalize(right["embedding"])], axis=0)).tolist()
            primary["provenance"] = left.get("provenance", []) + right.get("provenance", [])
            for item in remap:
                if item["camera_id"] == int(secondary["camera_id"]) and item["new_local_id"] == int(secondary["local_id"]):
                    item["new_local_id"] = int(primary["local_id"])
            rows = [row for idx, row in enumerate(rows) if idx not in {merge["left"], merge["right"]}] + [primary]
            changed = True
    rows.sort(key=lambda row: (int(row["camera_id"]), int(row["local_id"]), int(row["start_frame"])))
    return rows, report, remap


def rewrite_observations(source: Path, destination: Path, remap: list[dict]):
    lookup = {}
    for item in remap:
        lookup.setdefault((item["camera_id"], item["local_id"]), []).append(item)
    with source.open(newline="", encoding="utf-8") as input_stream, \
         destination.open("w", newline="", encoding="utf-8") as output_stream:
        reader = csv.DictReader(input_stream)
        writer = csv.DictWriter(output_stream, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            camera, local_id, frame = int(row["camera_id"]), int(row["local_id"]), int(row["frame_id"])
            match = next((item for item in lookup.get((camera, local_id), ())
                          if item["start_frame"] <= frame <= item["end_frame"]), None)
            if match:
                row["local_id"] = match["new_local_id"]
            writer.writerow(row)


def split_sustained(tracklets, cfg):
    """Split only when sustained ReID change also has a same-view motion break."""
    from .scoring import Evidence
    from .time_sync import observation_time

    next_id = {}
    for row in tracklets:
        c, lid = int(row["camera_id"]), int(row["local_id"])
        next_id[c] = max(next_id.get(c, 0), lid + 1)
    output, remap, diagnostics = [], [], []
    for row in tracklets:
        track = Evidence(row, cfg)
        c, lid = track.key
        frames = row.get("temporal_frames") or row.get("feature_frames", [])
        vectors = row.get("temporal_vectors") or row.get("feature_vectors", [])
        qualities = row.get("temporal_quality", []) if row.get("temporal_vectors") else row.get("feature_quality", [])
        if len(qualities) != len(vectors):
            qualities = [None] * len(vectors)
        samples = sorted((int(f), i, normalize(v)) for i, (f, v) in enumerate(zip(frames, vectors))
                         if len(v) and np.isfinite(v).all() and np.linalg.norm(v) > 1e-9
                         and (qualities[i] is None or qualities[i] >= .20))
        boundary = None
        boxes = row.get("observation_boxes", {})
        for index in range(3, len(samples) - 2):
            before = np.asarray([v for _, _, v in samples[index-3:index]])
            after = np.asarray([v for _, _, v in samples[index:index+3]])
            a, b = normalize(before.mean(axis=0)), normalize(after.mean(axis=0))
            similarity = float(a @ b)
            if min(before @ a) < .8 or min(after @ b) < .8 or similarity >= .5:
                continue
            left, right = samples[index-1][0], samples[index][0]
            box_a, box_b = boxes.get(str(left)), boxes.get(str(right))
            if not box_a or not box_b:
                continue
            x1, y1, x2, y2 = box_a
            u1, v1, u2, v2 = box_b
            jump = np.hypot((x1+x2-u1-u2)/2, y2-v2) / max(y2-y1, v2-v1, 1.)
            dt = track.time(right) - track.time(left)
            confirmed = 0 < dt <= 1.0 and jump > .9 + 2 * dt
            diagnostics.append({"camera_id": c, "local_id": lid, "frame": right,
                                "cosine": similarity, "box_jump_heights": float(jump), "split": confirmed})
            if confirmed:
                boundary = right
                break
        if boundary is None:
            output.append(dict(row, suspect_switch=True) if diagnostics and
                          diagnostics[-1]["camera_id"] == c and diagnostics[-1]["local_id"] == lid else row)
            continue
        provenance = {"camera_id": c, "local_id": lid,
                      "start_frame": int(row["start_frame"]), "end_frame": int(row["end_frame"]),
                      "source_uid": track.uid}
        for start, end, new_id in ((provenance["start_frame"], boundary-1, lid),
                                   (boundary, provenance["end_frame"], next_id[c])):
            observed = sorted(f for f in track.frames if start <= f <= end)
            if observed:
                start, end = observed[0], observed[-1]
            part = dict(row, local_id=new_id, start_frame=start, end_frame=end,
                        start_s=observation_time(row, start, track.fps),
                        end_s=observation_time(row, end, track.fps), observation_frames=observed,
                        tracklet_uid=track.uid if new_id == lid else f"{track.uid}:from:{boundary}",
                        provenance=[provenance], body_parts=[])
            part["positions"] = [p for p in row.get("positions", []) if start <= int(p[0]) <= end]
            for key in ("observation_boxes", "frame_timestamps", "position_covariances"):
                part[key] = {k: v for k, v in row.get(key, {}).items() if start <= int(k) <= end}
            for prefix in ("feature", "temporal"):
                indexes = [i for i, f in enumerate(row.get(prefix+"_frames", [])) if start <= int(f) <= end]
                suffixes = ("frames", "vectors", "quality", "quality_details") if prefix == "feature" else ("frames", "vectors", "quality")
                for suffix in suffixes:
                    values = row.get(prefix+"_"+suffix, [])
                    part[prefix+"_"+suffix] = [values[i] for i in indexes if i < len(values)]
                if prefix == "feature" and "foreground_ratio" in row:
                    part["foreground_ratio"] = [row["foreground_ratio"][i] for i in indexes]
            values = part["temporal_vectors"] or part["feature_vectors"]
            part["embedding"] = normalize(np.mean([normalize(v) for v in values], axis=0)).tolist() if values else []
            part["samples"] = len(values)
            output.append(part)
            remap.append({**provenance, "start_frame": start, "end_frame": end, "new_local_id": new_id})
        next_id[c] += 1
    return output, remap, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--auto-split", action="store_true")
    parser.add_argument("--auto-merge", action="store_true")
    parser.add_argument("--split-similarity", type=float, default=0.5)
    parser.add_argument("--merge-similarity", type=float, default=0.9)
    parser.add_argument("--max-gap-frames", type=int, default=90)
    args = parser.parse_args()
    tracklets = json.loads((args.run_dir / "local_tracklets.json").read_text(encoding="utf-8"))
    rows, report, remap = refine(tracklets, auto_split=args.auto_split, auto_merge=args.auto_merge,
                                 split_similarity=args.split_similarity,
                                 merge_similarity=args.merge_similarity,
                                 max_gap_frames=args.max_gap_frames)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "refinement_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.auto_split or args.auto_merge:
        (args.out / "local_tracklets.json").write_text(json.dumps(rows), encoding="utf-8")
        rewrite_observations(args.run_dir / "observations.csv", args.out / "observations.csv", remap)
        (args.out / "refinement_provenance.json").write_text(json.dumps(remap, indent=2), encoding="utf-8")
    print(json.dumps({"tracklets_before": len(tracklets), "tracklets_after": len(rows), **report}, indent=2))


if __name__ == "__main__":
    main()
