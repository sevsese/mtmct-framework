import json

import cv2
import numpy as np

from mtmct.oracle import generate


def test_oracle_tracklets_hide_gt_identity_from_matcher_artifacts(tmp_path):
    scene = tmp_path / "scene"
    videos = scene / "videos"
    videos.mkdir(parents=True)
    gt = {
        str(frame): [{
            "object type": "Person", "object id": 101,
            "2d bounding box visible": {
                "Camera_0000": [5, 5, 20, 30], "Camera_0001": [5, 5, 20, 30]
            },
        }]
        for frame in range(2)
    }
    (scene / "ground_truth.json").write_text(json.dumps(gt), encoding="utf-8")
    for camera in (0, 1):
        writer = cv2.VideoWriter(str(videos / f"Camera_{camera:04d}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 40))
        assert writer.isOpened()
        for _ in range(2):
            writer.write(np.full((40, 32, 3), (30, 100, 200), dtype=np.uint8))
        writer.release()

    output = tmp_path / "run"
    rows = generate(scene, output, [0, 1], 2, reid_method="color", sample_every=1)

    assert {(row["camera_id"], row["local_id"]) for row in rows} == {(0, 1), (1, 1)}
    assert all(row["local_id"] != 101 for row in rows)
    assert "101" not in (output / "local_tracklets.json").read_text(encoding="utf-8")
    assert "object_id" not in (output / "observations.csv").read_text(encoding="utf-8")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tracking_config"]["gt_identity_exposed_to_matcher"] is False
