"""Versioned Global 2D IDF1 protocol; this is not official NVIDIA 3D HOTA."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .nvidia import NvidiaDatasetAdapter

PROTOCOL_VERSION = "global-2d-idf1-v1"


def iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    union += max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union else 0.0


def _match_boxes(gt, predictions, threshold):
    if not gt or not predictions:
        return []
    overlaps = np.asarray([[iou(g[1], p[1]) for p in predictions] for g in gt])
    benefit = np.where(overlaps >= threshold, 1.0 + overlaps, 0.0)
    rows, columns = linear_sum_assignment(-benefit)
    return [(gt[row][0], predictions[column][0])
            for row, column in zip(rows, columns) if benefit[row, column] > 0]


def _safe_ratio(numerator, denominator, empty_value=0.0):
    return numerator / denominator if denominator else empty_value


def _identity_metrics(matches: Counter, gt_total: int, pred_total: int):
    gt_ids = sorted({gt_id for gt_id, _ in matches})
    pred_ids = sorted({pred_id for _, pred_id in matches})
    if gt_ids and pred_ids:
        matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
        gt_index = {identity: index for index, identity in enumerate(gt_ids)}
        pred_index = {identity: index for index, identity in enumerate(pred_ids)}
        for (gt_id, pred_id), count in matches.items():
            matrix[gt_index[gt_id], pred_index[pred_id]] = count
        rows, columns = linear_sum_assignment(-matrix)
        idtp = int(matrix[rows, columns].sum())
    else:
        idtp = 0
    idfp = pred_total - idtp
    idfn = gt_total - idtp
    empty = 1.0 if gt_total == pred_total == 0 else 0.0
    idp = _safe_ratio(idtp, pred_total, empty)
    idr = _safe_ratio(idtp, gt_total, empty)
    idf1 = _safe_ratio(2 * idtp, 2 * idtp + idfp + idfn, empty)
    return {
        "IDTP": idtp,
        "IDFP": idfp,
        "IDFN": idfn,
        "IDP": round(idp, 6),
        "IDR": round(idr, 6),
        "IDF1": round(idf1, 6),
        "gt_count": gt_total,
        "pred_count": pred_total,
    }


def _pairwise_metrics(matched_by_camera: dict[int, Counter]):
    true_positive = 0
    predicted_positive = 0
    actual_positive = 0
    for camera_a, camera_b in combinations(sorted(matched_by_camera), 2):
        left, right = matched_by_camera[camera_a], matched_by_camera[camera_b]
        for (gt_a, pred_a), count_a in left.items():
            for (gt_b, pred_b), count_b in right.items():
                pairs = count_a * count_b
                if pred_a == pred_b:
                    predicted_positive += pairs
                if gt_a == gt_b:
                    actual_positive += pairs
                if gt_a == gt_b and pred_a == pred_b:
                    true_positive += pairs
    precision = _safe_ratio(true_positive, predicted_positive, 1.0 if actual_positive == 0 else 0.0)
    recall = _safe_ratio(true_positive, actual_positive, 1.0 if predicted_positive == 0 else 0.0)
    f1 = _safe_ratio(2 * precision * recall, precision + recall, 1.0 if not actual_positive and not predicted_positive else 0.0)
    return {
        "true_positive_pairs": true_positive,
        "predicted_positive_pairs": predicted_positive,
        "actual_positive_pairs": actual_positive,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "F1": round(f1, 6),
    }


def eval_2d(pred_path: Path, gt_path: Path, cameras: list[int],
            max_frames: int, stride: int = 1, threshold: float = 0.5, *, start_frame: int = 0):
    if not 0 <= start_frame < max_frames or stride <= 0:
        raise ValueError("Use finite positive --max-frames and --stride")
    if not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must be between 0 and 1")
    cameras = sorted(set(cameras))
    selected_frames = set(range(start_frame, max_frames, stride))
    selected_keys = {(frame, camera) for frame in selected_frames for camera in cameras}

    predictions = defaultdict(list)
    pred_by_camera = Counter()
    with Path(pred_path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["frame_id"]), int(row["camera_id"]))
            if key not in selected_keys:
                continue
            predictions[key].append((int(row["global_id"]), [
                float(row[name]) for name in ("x1", "y1", "x2", "y2")
            ]))
            pred_by_camera[key[1]] += 1

    ground_truth = defaultdict(list)
    gt_by_camera = Counter()
    for observation in NvidiaDatasetAdapter(gt_path).observations(strict=False, max_frames=max_frames):
        key = (observation.frame_index, observation.camera_id)
        if key not in selected_keys:
            continue
        ground_truth[key].append((observation.gt_object_id, observation.bbox_xyxy))
        gt_by_camera[observation.camera_id] += 1

    matches = Counter()
    matches_by_camera: dict[int, Counter] = {camera: Counter() for camera in cameras}
    previous = {}
    switches = 0
    for key in sorted(selected_keys):
        for gt_id, pred_id in _match_boxes(ground_truth[key], predictions[key], threshold):
            matches[(gt_id, pred_id)] += 1
            matches_by_camera[key[1]][(gt_id, pred_id)] += 1
            track = key[1], gt_id
            switches += int(track in previous and previous[track] != pred_id)
            previous[track] = pred_id

    gt_total, pred_total = sum(gt_by_camera.values()), sum(pred_by_camera.values())
    global_metrics = _identity_metrics(matches, gt_total, pred_total)
    by_prediction, by_truth = defaultdict(set), defaultdict(set)
    for gt_id, pred_id in matches:
        by_prediction[pred_id].add(gt_id)
        by_truth[gt_id].add(pred_id)
    detection_tp = sum(matches.values())
    detection = {
        "TP": detection_tp,
        "FP": pred_total - detection_tp,
        "FN": gt_total - detection_tp,
        "precision": round(_safe_ratio(detection_tp, pred_total, 1.0 if gt_total == 0 else 0.0), 6),
        "recall": round(_safe_ratio(detection_tp, gt_total, 1.0 if pred_total == 0 else 0.0), 6),
        "F1": round(_safe_ratio(2 * detection_tp, gt_total + pred_total, 1.0 if gt_total == pred_total == 0 else 0.0), 6),
    }
    per_camera = {
        str(camera): {
            **_identity_metrics(matches_by_camera[camera], gt_by_camera[camera], pred_by_camera[camera]),
            "HOTA": None,
            "HOTA_note": "Not computed; TrackEval is not installed or invoked by this protocol",
        }
        for camera in cameras
    }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": "bbox IoU matching within (scene,camera,frame), then global one-to-one identity assignment across selected cameras",
        "official_nvidia_3d_hota": False,
        "iou_threshold": threshold,
        "max_frames": max_frames,
        "start_frame": start_frame,
        "stride": stride,
        "cameras": cameras,
        "global": global_metrics,
        "detection": detection,
        "per_camera": per_camera,
        "pairwise_association": _pairwise_metrics(matches_by_camera),
        "identity_diagnostics": {
            "IDSW": switches,
            "N_global": len({gid for frame in predictions.values() for gid, _ in frame}),
            "N_false_merge": sum(len(ids) > 1 for ids in by_prediction.values()),
            "N_fragment": sum(len(ids) > 1 for ids in by_truth.values()),
            "protocol": "At least one IoU match per identity pair; IDSW on consecutive matched observations per camera and GT ID.",
        },
        # Compatibility with the previous CLI/report consumers.
        "diagnostic_IDF1": round(global_metrics["IDF1"], 4),
        "IDTP": global_metrics["IDTP"],
        "IDFP": global_metrics["IDFP"],
        "IDFN": global_metrics["IDFN"],
        "detection_F1": round(detection["F1"], 4),
        "detection_precision": round(detection["precision"], 4),
        "detection_recall": round(detection["recall"], 4),
        "gt_count": global_metrics["gt_count"],
        "pred_count": global_metrics["pred_count"],
        "note": "Custom Global 2D IDF1; NOT official NVIDIA 3D HOTA",
    }
    return result


def save_report(result: dict, output: Path) -> None:
    """Store exactly the same JSON and error totals for CLI and pipeline runs."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output.parent / "errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["error_type", "count"])
        writer.writerows([
            ["identity_false_positive", result["global"]["IDFP"]],
            ["identity_false_negative", result["global"]["IDFN"]],
            ["detection_false_positive", result["detection"]["FP"]],
            ["detection_false_negative", result["detection"]["FN"]],
        ])


def inferred_max_frames(pred_path: Path) -> int:
    """For --max-frames 0, evaluate up through the last predicted frame."""
    last = -1
    with Path(pred_path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            last = max(last, int(row["frame_id"]))
    if last < 0:
        raise ValueError("Cannot infer evaluation window from empty predictions; use --max-frames")
    return last + 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = eval_2d(args.pred, args.gt, args.cameras, args.max_frames, args.stride, args.iou)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        save_report(result, args.out)


if __name__ == "__main__":
    main()
