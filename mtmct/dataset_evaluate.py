"""Dataset-neutral 2D Global-IDF1 (custom, not official benchmark metric)."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import csv
from .datasets import MTMCTDataset
from .evaluate import _identity_metrics, _match_boxes, _pairwise_metrics, _safe_ratio

PROTOCOL = 'multidataset-global-2d-idf1-v1'


def evaluate_bbox(pred_path: str | Path, dataset: MTMCTDataset, scene: str,
                  cameras: list[int], max_frames: int, iou_threshold: float = 0.5) -> dict:
    if not 0 <= iou_threshold <= 1:
        raise ValueError('IoU threshold must be in [0, 1]')
    if not dataset.supports_bbox_evaluation(scene, cameras):
        raise ValueError(f'{dataset.kind}/{scene}: no trusted camera bbox ground truth for {cameras}. '
                         'Do not score world-only GT or SCOUT pseudo-labels as manually verified boxes.')
    selected = set(cameras)
    ground_truth, predictions = defaultdict(list), defaultdict(list)
    gt_count, pred_count = Counter(), Counter()
    for obs in dataset.observations(scene, max_frames, cameras):
        if obs.camera_id not in selected or obs.bbox_xyxy is None:
            continue
        key = (obs.camera_id, obs.frame_id)
        ground_truth[key].append((obs.identity, obs.bbox_xyxy))
        gt_count[obs.camera_id] += 1
    with Path(pred_path).open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            cam, fid = int(row['camera_id']), int(row['frame_id'])
            if cam not in selected or (max_frames > 0 and fid >= max_frames):
                continue
            predictions[(cam, fid)].append((int(row['global_id']),
                                           tuple(float(row[k]) for k in ('x1','y1','x2','y2'))))
            pred_count[cam] += 1
    matches, camera_matches = Counter(), {c: Counter() for c in selected}
    for key in set(predictions) | set(ground_truth):
        for gt_id, predicted_id in _match_boxes(ground_truth[key], predictions[key], iou_threshold):
            matches[(gt_id, predicted_id)] += 1
            camera_matches[key[0]][(gt_id, predicted_id)] += 1
    total_gt, total_pred = sum(gt_count.values()), sum(pred_count.values())
    tp = sum(matches.values())
    precision = _safe_ratio(tp, total_pred, 1.0 if total_gt == 0 else 0.0)
    recall = _safe_ratio(tp, total_gt, 1.0 if total_pred == 0 else 0.0)
    return {
        'protocol_version': PROTOCOL,
        'dataset': dataset.kind,
        'scene': scene,
        'official_metric': False,
        'iou_threshold': iou_threshold,
        'cameras': sorted(selected),
        'max_frames': max_frames,
        'global': _identity_metrics(matches, total_gt, total_pred),
        'detection': {
            'TP': tp, 'FP': total_pred-tp, 'FN': total_gt-tp,
            'precision': round(precision,6), 'recall': round(recall,6),
            'F1': round(_safe_ratio(2*tp, total_gt+total_pred,
                                    1.0 if total_gt == total_pred == 0 else 0.0),6),
        },
        'per_camera': {str(c): _identity_metrics(camera_matches[c], gt_count[c], pred_count[c])
                       for c in sorted(selected)},
        'pairwise_association': _pairwise_metrics(camera_matches),
        'warning': 'Custom 2D bbox-based global IDF1, NOT official NVIDIA 3D HOTA or SCOUT TrackEval HOTA.',
        'gt_annotation_quality': ('mixed/manual+automated' if dataset.kind == 'scout' and
                                  getattr(dataset,'allow_pseudo_labels',False) else
                                  'dataset-provided'),
    }
