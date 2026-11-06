"""Bounded feature samples and offline aggregation policies."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

MAX_SAMPLES = 20


def reliable_gallery(row: dict, count: int = 3) -> tuple[list, float]:
    """Diverse high-quality prototypes; weak samples never enter the gallery.

    Keep the all-observation mean as an anchor, instead of replacing it with
    three individual crops (which regressed on the baseline experiment).
    """
    vectors = row.get("feature_vectors", [])
    qualities = row.get("feature_quality", [])
    if len(vectors) != len(qualities):
        raise ValueError("feature vectors and qualities must align")
    anchor = normalize(row.get("embedding", []))
    if anchor.size and not np.isfinite(anchor).all():
        return [], 0.0
    valid = [(normalize(v), float(q)) for v, q in zip(vectors, qualities)
             if len(v) and np.isfinite(v).all() and np.isfinite(q) and q > 0]
    if not valid:
        saved = row.get('prototypes') or ([anchor.tolist()] if anchor.size else [])
        return saved[:count], float(row.get('gallery_reliability', 0.0))
    best = max(q for _, q in valid)
    pool = [(v, q) for v, q in valid if q >= max(0.15, best * 0.6)
            and (not anchor.size or (v.shape == anchor.shape and v @ anchor >= 0.5))]
    selected = [anchor] if anchor.size else []
    while pool and len(selected) < count:
        index = max(range(len(pool)), key=lambda i: pool[i][1] * (
            min(1 - float(pool[i][0] @ p) for p in selected) if selected else 1))
        vector, _ = pool.pop(index)
        if not selected or max(float(vector @ p) for p in selected) < 0.995:
            selected.append(vector)
    reliability = min(1., best / 0.5) * min(1., len(valid) / 3)
    return [v.tolist() for v in selected], float(reliability)


def gallery_similarity(a: dict, b: dict) -> tuple[float | None, float]:
    va, ra = reliable_gallery(a)
    vb, rb = reliable_gallery(b)
    if not va or not vb or len(va[0]) != len(vb[0]):
        return None, 0.0
    scores = np.asarray(va) @ np.asarray(vb).T
    anchor = float(scores[0, 0])
    # A single accidental high pair cannot dominate the mean anchor.
    support = float((scores.max(axis=0).mean() + scores.max(axis=1).mean()) / 2)
    reliability = min(ra, rb)
    return float(np.clip(anchor + 0.25 * reliability * (support - anchor), -1, 1)), reliability


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def crop_quality(crop: np.ndarray, confidence: float, clipped: bool = False) -> float:
    if crop.size == 0:
        return 0.0
    area = min(crop.shape[0] * crop.shape[1] / (256 * 128), 1.0)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0, 1.0)
    border = 0.5 if clipped else 1.0
    return float(max(0.0, min(confidence, 1.0)) * border * (0.5 * area + 0.5 * sharpness))


def aggregate(vectors, qualities=None, policy="mean", top_k=5, prototype_count=3):
    samples = [normalize(vector) for vector in vectors if len(vector)]
    if not samples:
        return np.asarray([], dtype=np.float32), []
    quality = list(qualities or [1.0] * len(samples))
    if len(quality) != len(samples):
        raise ValueError("feature vectors and quality values must have equal length")
    if policy == "first":
        selected = [0]
    elif policy == "mean":
        selected = list(range(len(samples)))
    elif policy == "topk":
        selected = sorted(range(len(samples)), key=lambda index: (-quality[index], index))[:top_k]
    elif policy == "prototypes":
        selected = [max(range(len(samples)), key=lambda index: (quality[index], -index))]
        while len(selected) < min(prototype_count, len(samples)):
            remaining = [index for index in range(len(samples)) if index not in selected]
            selected.append(max(
                remaining,
                key=lambda index: min(1.0 - float(samples[index] @ samples[chosen]) for chosen in selected),
            ))
    else:
        raise ValueError("policy must be first, mean, topk or prototypes")
    chosen = [samples[index] for index in selected]
    return normalize(np.mean(chosen, axis=0)), [sample.tolist() for sample in chosen]


def apply_policy(tracklets: list[dict], policy: str, top_k: int = 5,
                 prototype_count: int = 3) -> list[dict]:
    output = []
    for original in tracklets:
        row = dict(original)
        if policy == "mean" and row.get("embedding"):
            row["prototypes"] = []
            row["reid_policy"] = policy
            output.append(row)
            continue
        vectors = row.get("feature_vectors") or ([row["embedding"]] if row.get("embedding") else [])
        qualities = row.get("feature_quality") or [1.0] * len(vectors)
        embedding, prototypes = aggregate(vectors, qualities, policy, top_k, prototype_count)
        row["embedding"] = embedding.tolist()
        row["prototypes"] = prototypes if policy == "prototypes" else []
        row["reid_policy"] = policy
        output.append(row)
    return output


def analyze(tracklets_path: Path, labels_path: Path, output: Path):
    tracklets = json.loads(tracklets_path.read_text(encoding="utf-8"))
    labels = {(int(row["camera_id"]), int(row["local_id"])): int(row["gt_object_id"])
              for row in json.loads(labels_path.read_text(encoding="utf-8"))}
    pairs = []
    for index, left in enumerate(tracklets):
        left_key = (int(left["camera_id"]), int(left["local_id"]))
        if left_key not in labels or not left.get("embedding"):
            continue
        for right in tracklets[index + 1:]:
            right_key = (int(right["camera_id"]), int(right["local_id"]))
            if left_key[0] == right_key[0] or right_key not in labels or not right.get("embedding"):
                continue
            pairs.append({
                "camera_a": left_key[0], "local_a": left_key[1],
                "camera_b": right_key[0], "local_b": right_key[1],
                "positive": int(labels[left_key] == labels[right_key]),
                "cosine": float(normalize(left["embedding"]) @ normalize(right["embedding"])),
            })
    ranked = sorted(pairs, key=lambda row: -row["cosine"])
    positives = sum(row["positive"] for row in ranked)
    seen_positive = 0
    precision_sum = 0.0
    precision_at_recall_90 = 0.0
    for rank, row in enumerate(ranked, 1):
        if row["positive"]:
            seen_positive += 1
            precision_sum += seen_positive / rank
        if positives and seen_positive / positives >= 0.9 and precision_at_recall_90 == 0:
            precision_at_recall_90 = seen_positive / rank
    queries = 0
    rank1 = 0
    for tracklet in tracklets:
        key = (int(tracklet["camera_id"]), int(tracklet["local_id"]))
        candidates = [row for row in pairs
                      if (row["camera_a"], row["local_a"]) == key
                      or (row["camera_b"], row["local_b"]) == key]
        if key in labels and candidates and any(row["positive"] for row in candidates):
            queries += 1
            rank1 += max(candidates, key=lambda row: row["cosine"])["positive"]
    report = {
        "tracklets": len(tracklets), "cross_camera_pairs": len(pairs),
        "positive_pairs": positives, "negative_pairs": len(pairs) - positives,
        "rank1": round(rank1 / queries, 6) if queries else 0.0,
        "average_precision": round(precision_sum / positives, 6) if positives else 0.0,
        "precision_at_recall_0.90": round(precision_at_recall_90, 6),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "reid_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["camera_a", "local_a", "camera_b", "local_b", "positive", "cosine"])
        writer.writeheader(); writer.writerows(ranked)
    (output / "reid_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    canvas = np.full((300, 640, 3), 255, dtype=np.uint8)
    for positive, color in ((0, (220, 80, 80)), (1, (60, 160, 60))):
        scores = [row["cosine"] for row in pairs if row["positive"] == positive]
        hist, _ = np.histogram(scores, bins=40, range=(-1, 1))
        scale = 240 / max(int(hist.max()), 1)
        for index, count in enumerate(hist):
            x1 = index * 16
            cv2.rectangle(canvas, (x1, 280), (x1 + 14, 280 - int(count * scale)), color, -1)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Could not encode cosine distribution plot")
    encoded.tofile(output / "cosine_distributions.png")
    return report


def enrich_temporal(tracklets, observations_path: Path, scene: Path, *,
                    device='cuda:0', sample_every=5):
    """Recover chronological samples from immutable cached boxes; no detector/GT.

    Legacy caches kept only their 20 best crops, which can hide an ID switch.
    Keep original mean/gallery intact so this is a controlled split ablation.
    """
    import copy
    from .frame_sources import FrameSource
    from .reid import ReID
    from .weights import download_weights
    if sample_every < 1:
        raise ValueError('sample_every must be positive')
    rows = copy.deepcopy(tracklets)
    by_key = {(int(t['camera_id']), int(t['local_id'])): t for t in rows}
    selected = {}
    seen = set()
    with observations_path.open(newline='', encoding='utf-8') as stream:
        for obs in csv.DictReader(stream):
            cam, lid, fid = (int(obs[k]) for k in ('camera_id', 'local_id', 'frame_id'))
            if (cam, lid) not in by_key:
                continue
            if fid % sample_every == 0 or (cam, lid) not in seen:
                selected.setdefault(cam, {}).setdefault(fid, []).append(obs)
                seen.add((cam, lid))
    model = ReID('osnet', device=device, weights=str(download_weights('osnet_x1_0')))
    for t in rows:
        t['temporal_frames'], t['temporal_vectors'] = [], []
    for cam, by_frame in sorted(selected.items()):
        with FrameSource(scene/'videos'/f'Camera_{cam:04d}.mp4') as source:
            for fid, observations in sorted(by_frame.items()):
                frame = source.get(fid)
                if frame is None:
                    raise RuntimeError(f'Missing camera {cam} frame {fid}')
                crops, keys = [], []
                for obs in observations:
                    x1, y1, x2, y2 = [int(float(obs[k])) for k in ('x1', 'y1', 'x2', 'y2')]
                    crop = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
                    if crop.size:
                        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        keys.append((cam, int(obs['local_id'])))
                for key, vector in zip(keys, model.encode(crops)):
                    t = by_key[key]
                    t['temporal_frames'].append(fid)
                    t['temporal_vectors'].append(vector.tolist())
                    if len(t['temporal_frames']) > 256:
                        t['temporal_frames'] = t['temporal_frames'][::2]
                        t['temporal_vectors'] = t['temporal_vectors'][::2]
        print(f'Temporal features camera {cam} complete', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=["first", "mean", "topk", "prototypes"], default="mean")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prototypes", type=int, default=3)
    parser.add_argument("--labels", type=Path, help="Analysis-only GT labels")
    parser.add_argument("--analysis-out", type=Path)
    parser.add_argument('--temporal-observations', type=Path)
    parser.add_argument('--scene-dir', type=Path)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if args.temporal_observations:
        if not args.scene_dir:
            parser.error('--temporal-observations requires --scene-dir')
        output = enrich_temporal(rows, args.temporal_observations, args.scene_dir)
    else:
        output = apply_policy(rows, args.policy, args.top_k, args.prototypes)
    args.output.write_text(json.dumps(output), encoding="utf-8")
    if args.labels and args.analysis_out:
        print(json.dumps(analyze(args.output, args.labels, args.analysis_out), indent=2))


if __name__ == "__main__":
    main()
