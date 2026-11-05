import csv
import json

from mtmct.evaluate import PROTOCOL_VERSION, eval_2d


FIELDS = ["camera_id", "frame_id", "global_id", "x1", "y1", "x2", "y2"]


def write_gt(path, rows):
    frames = {}
    for frame, camera, identity, box in rows:
        objects = frames.setdefault(str(frame), [])
        existing = next((obj for obj in objects if obj["object id"] == identity), None)
        if existing is None:
            existing = {"object type": "Person", "object id": identity, "2d bounding box visible": {}}
            objects.append(existing)
        existing["2d bounding box visible"][f"Camera_{camera:04d}"] = box
    path.write_text(json.dumps(frames), encoding="utf-8")


def write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for frame, camera, identity, box in rows:
            writer.writerow(dict(zip(FIELDS, [camera, frame, identity, *box])))


def evaluate(tmp_path, gt_rows, prediction_rows, cameras=(0, 1), max_frames=3):
    gt, predictions = tmp_path / "gt.json", tmp_path / "pred.csv"
    write_gt(gt, gt_rows)
    write_predictions(predictions, prediction_rows)
    return eval_2d(predictions, gt, list(cameras), max_frames=max_frames)


def perfect_rows():
    return [
        (frame, camera, identity, box)
        for frame in (0, 1)
        for camera in (0, 1)
        for identity, box in ((10, [0, 0, 10, 10]), (20, [20, 0, 30, 10]))
    ]


def test_perfect_prediction_is_one(tmp_path):
    rows = perfect_rows()
    result = evaluate(tmp_path, rows, rows)
    assert result["protocol_version"] == PROTOCOL_VERSION
    assert result["global"]["IDF1"] == 1.0
    assert result["global"]["IDP"] == 1.0
    assert result["global"]["IDR"] == 1.0
    assert result["pairwise_association"]["F1"] == 1.0


def test_lost_detection_lowers_recall_and_false_detection_lowers_precision(tmp_path):
    rows = perfect_rows()
    lost = evaluate(tmp_path, rows, rows[:-1])
    extra = evaluate(tmp_path, rows, rows + [(2, 0, 99, [40, 0, 50, 10])])
    assert lost["global"]["IDR"] < 1 and lost["global"]["IDP"] == 1
    assert extra["global"]["IDP"] < 1 and extra["global"]["IDR"] == 1


def test_false_merge_false_split_and_temporal_swap_lower_idf1(tmp_path):
    rows = perfect_rows()
    false_merge = [(f, c, 1, box) for f, c, _identity, box in rows]
    false_split = [(f, c, index + 1, box) for index, (f, c, _identity, box) in enumerate(rows)]
    temporal_swap = [
        (frame, camera, identity if frame == 0 else 30 - identity, box)
        for frame, camera, identity, box in rows
    ]
    assert evaluate(tmp_path, rows, false_merge)["global"]["IDF1"] < 1
    assert evaluate(tmp_path, rows, false_split)["global"]["IDF1"] < 1
    assert evaluate(tmp_path, rows, temporal_swap)["global"]["IDF1"] < 1


def test_same_frame_boxes_from_different_cameras_never_match(tmp_path):
    gt = [(0, 0, 10, [0, 0, 10, 10])]
    prediction = [(0, 1, 10, [0, 0, 10, 10])]
    result = evaluate(tmp_path, gt, prediction)
    assert result["global"]["IDTP"] == 0
    assert result["global"]["IDFP"] == 1
    assert result["global"]["IDFN"] == 1


def test_empty_gt_and_predictions_are_deterministic(tmp_path):
    result = evaluate(tmp_path, [], [])
    assert result["global"] == {
        "IDTP": 0, "IDFP": 0, "IDFN": 0, "IDP": 1.0, "IDR": 1.0,
        "IDF1": 1.0, "gt_count": 0, "pred_count": 0,
    }
