"""Separate world-position identity protocol for CITIC-style annotations."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .evaluate import _identity_metrics

PROTOCOL_VERSION = "citic-world-position-idf1-v1"


def evaluate_world_positions(pred_path: Path, gt_path: Path, distance_threshold_m: float = 1.0):
    def load(path, id_field):
        rows = defaultdict(list)
        with Path(path).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows[float(row["timestamp_s"])].append(
                    (int(row[id_field]), (float(row["x"]), float(row["y"]))))
        return rows
    predictions, ground_truth = load(pred_path, "global_id"), load(gt_path, "person_id")
    matches = Counter()
    for timestamp in sorted(set(predictions) | set(ground_truth)):
        pred, gt = predictions[timestamp], ground_truth[timestamp]
        if not pred or not gt:
            continue
        distances = np.asarray([[np.linalg.norm(np.subtract(g[1], p[1])) for p in pred] for g in gt])
        benefit = np.where(distances <= distance_threshold_m, 1.0 + (distance_threshold_m - distances), 0.0)
        rows, columns = linear_sum_assignment(-benefit)
        for row, column in zip(rows, columns):
            if benefit[row, column] > 0:
                matches[(gt[row][0], pred[column][0])] += 1
    gt_total = sum(len(rows) for rows in ground_truth.values())
    pred_total = sum(len(rows) for rows in predictions.values())
    return {"protocol_version": PROTOCOL_VERSION,
            "official_bbox_idf1": False,
            "distance_threshold_m": distance_threshold_m,
            "global": _identity_metrics(matches, gt_total, pred_total)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate_world_positions(args.pred, args.gt, args.distance)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
