import csv
import json

import numpy as np

from mtmct.floor_map import build_floor_tracks, homographies, load_calibration, project_foot


def test_identity_floor_projection(tmp_path):
    config = tmp_path / "map.json"
    config.write_text(json.dumps({
        "status": "verified", "camera_fps": {"0": 10}, "camera_offsets_s": {"0": 2},
        "cameras": {"0": {"image_size": [100, 100],
            "image_points_norm": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "map_points_norm": [[0, 0], [1, 0], [1, 1], [0, 1]]}}
    }), encoding="utf-8")
    matrix = homographies(load_calibration(config))[0]
    assert np.allclose(project_foot(matrix, (20, 10, 40, 60), (100, 100)), (.3, .6))

    predictions = tmp_path / "tracks.csv"
    with predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "camera_id", "frame_id", "local_id", "global_id", "x1", "y1", "x2", "y2"])
        writer.writeheader(); writer.writerow({"camera_id": 0, "frame_id": 20, "local_id": 1,
            "global_id": 7, "x1": 20, "y1": 10, "x2": 40, "y2": 60})
    rows = build_floor_tracks(predictions, config, tmp_path / "floor.csv")
    assert rows[0]["timestamp_s"] == 4.0
    assert rows[0]["global_id"] == 7
