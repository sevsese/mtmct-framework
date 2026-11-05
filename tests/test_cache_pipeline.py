import csv
import json
import sys

import pytest

from mtmct.association import Config
from mtmct.cache import validate_cache, write_manifest
from mtmct.link import run as link_run


def make_cache(run_dir, input_path, detector="model.pt"):
    run_dir.mkdir()
    tracklets = [
        {"camera_id": 0, "local_id": 1, "start_frame": 0, "end_frame": 2,
         "embedding": [1.0, 0.0], "positions": [], "samples": 1},
        {"camera_id": 1, "local_id": 2, "start_frame": 0, "end_frame": 2,
         "embedding": [], "positions": [], "samples": 0},
    ]
    (run_dir / "local_tracklets.json").write_text(json.dumps(tracklets), encoding="utf-8")
    with (run_dir / "observations.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["camera_id", "frame_id", "local_id", "x1", "y1", "x2", "y2",
                  "confidence", "world_x", "world_y"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for camera, local_id in ((0, 1), (1, 2)):
            writer.writerow({"camera_id": camera, "frame_id": 0, "local_id": local_id,
                             "x1": 0, "y1": 0, "x2": 10, "y2": 20,
                             "confidence": 1, "world_x": "", "world_y": ""})
    write_manifest(run_dir, scene="Synthetic", cameras=[0, 1], inputs=[input_path],
                   detector=detector, reid="test", feature_dimension=2, config={"conf": 0.1})


def test_link_only_needs_no_video_gpu_or_scene(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("input", encoding="utf-8")
    run_dir = tmp_path / "run"
    make_cache(run_dir, source)
    source.unlink()  # Offline reuse: original input may be unavailable.

    tracklets, mapping, _edges, csv_path, _txt_path = link_run(
        run_dir, Config(use_geometry=False, spatial_weight=0), expected_detector="model.pt")

    assert len(tracklets) == 2
    assert len(mapping) == 2
    assert mapping[(0, 1)] != mapping[(1, 2)]  # no-feature tracklet remains a singleton
    assert csv_path.exists()


def test_cache_rejects_model_feature_and_input_changes(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("input", encoding="utf-8")
    run_dir = tmp_path / "run"
    make_cache(run_dir, source)
    with pytest.raises(ValueError, match="detector mismatch"):
        validate_cache(run_dir, "other.pt")
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        validate_cache(run_dir, "model.pt")
    source.write_text("input", encoding="utf-8")
    rows = json.loads((run_dir / "local_tracklets.json").read_text(encoding="utf-8"))
    rows[0]["embedding"] = [1, 0, 0]
    (run_dir / "local_tracklets.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="feature dimension mismatch"):
        validate_cache(run_dir, "model.pt")


def test_main_link_only_does_not_check_data_or_render(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("input", encoding="utf-8")
    run_dir = tmp_path / "run"
    make_cache(run_dir, source)
    source.unlink()
    monkeypatch.setattr(sys, "argv", ["mtmct", "--link-only", "--out", str(run_dir),
                                      "--model", "model.pt", "--data-root", str(tmp_path / "missing")])
    from mtmct.__main__ import main
    main()
    assert (run_dir / "global_ids.json").exists()
    assert not (run_dir / "mosaic.mp4").exists()
