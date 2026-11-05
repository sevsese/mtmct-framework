import numpy as np
import json

from mtmct.features import aggregate, analyze, apply_policy, crop_quality


def test_feature_policies_normalize_and_select_expected_samples():
    vectors = [[1, 0], [0.8, 0.2], [0, 1]]
    quality = [0.2, 0.9, 0.1]
    first, _ = aggregate(vectors, quality, "first")
    mean, _ = aggregate(vectors, quality, "mean")
    topk, selected = aggregate(vectors, quality, "topk", top_k=1)
    prototypes, prototype_vectors = aggregate(vectors, quality, "prototypes", prototype_count=2)
    assert np.allclose(first, [1, 0])
    assert np.allclose(topk, np.asarray(vectors[1]) / np.linalg.norm(vectors[1]))
    assert len(selected) == 1 and len(prototype_vectors) == 2
    assert np.isclose(np.linalg.norm(mean), 1) and np.isclose(np.linalg.norm(prototypes), 1)


def test_missing_features_remain_empty_and_quality_is_bounded():
    rows = apply_policy([{"embedding": [], "camera_id": 0, "local_id": 1}], "topk")
    assert rows[0]["embedding"] == []
    crop = np.zeros((64, 32, 3), dtype=np.uint8)
    assert 0 <= crop_quality(crop, 0.8) <= 1


def test_reid_analysis_reports_perfect_separated_pairs(tmp_path):
    tracklets = [
        {"camera_id": 0, "local_id": 1, "embedding": [1, 0]},
        {"camera_id": 1, "local_id": 2, "embedding": [1, 0]},
        {"camera_id": 1, "local_id": 3, "embedding": [0, 1]},
    ]
    labels = [
        {"camera_id": 0, "local_id": 1, "gt_object_id": 10},
        {"camera_id": 1, "local_id": 2, "gt_object_id": 10},
        {"camera_id": 1, "local_id": 3, "gt_object_id": 20},
    ]
    tracklets_path, labels_path = tmp_path / "tracklets.json", tmp_path / "labels.json"
    tracklets_path.write_text(json.dumps(tracklets), encoding="utf-8")
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    report = analyze(tracklets_path, labels_path, tmp_path / "report")
    assert report["rank1"] == 1 and report["average_precision"] == 1
    assert (tmp_path / "report" / "cosine_distributions.png").exists()
