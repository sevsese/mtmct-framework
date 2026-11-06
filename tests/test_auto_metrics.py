import csv
import importlib
import json
import sys

import pytest
from mtmct.evaluate import save_report, inferred_max_frames


def test_save_report_writes_json_and_csv(tmp_path):
    report = {"global": {"IDFP": 4, "IDFN": 5, "IDF1": .5},
              "detection": {"FP": 2, "FN": 3, "recall": .7}}
    p = tmp_path / "metrics.json"
    save_report(report, p)
    assert json.loads(p.read_text()) == report
    rows = list(csv.DictReader((tmp_path / "errors.csv").open()))
    assert len(rows) == 4
    assert rows[0] == {"error_type": "identity_false_positive", "count": "4"}


def test_evaluation_auto_called_when_link_only(tmp_path, monkeypatch):
    main = importlib.import_module("mtmct.__main__")
    evaluate = importlib.import_module("mtmct.evaluate")
    gt = tmp_path / "ground_truth.json"
    gt.write_text("{}", encoding="utf-8")
    out = tmp_path / "case"
    out.mkdir()
    pred = out / "global_tracks_2d.csv"
    pred.write_text("camera_id,frame_id,local_id,global_id,x1,y1,x2,y2\n")
    called = []
    monkeypatch.setattr(main, "link_run", lambda *a, **kw: ([], {}, [], pred, None))
    monkeypatch.setattr(evaluate, "eval_2d", lambda *a, **kw: (
        called.append((a, kw)) or
        {"global": {"IDFP": 0, "IDFN": 0, "IDF1": .8},
         "detection": {"FP": 0, "FN": 0, "recall": .6}}))
    monkeypatch.setattr(sys, "argv", ["mtmct", "--link-only", "--evaluate",
                                       "--no-download", "--out", str(out),
                                       "--gt", str(gt), "--max-frames", "12"])
    main.main()
    assert called and called[0][0][3] == 12
    assert (out / "metrics.json").is_file()
    assert (out / "errors.csv").is_file()


def test_missing_gt_fails_before_gpu(tmp_path, monkeypatch):
    main = importlib.import_module("mtmct.__main__")
    monkeypatch.setattr(sys, "argv", ["mtmct", "--no-download", "--evaluate",
                                       "--gt", str(tmp_path / "missing_gt.json"),
                                       "--out", str(tmp_path / "out")])
    with pytest.raises(FileNotFoundError, match="Cannot compute metrics"):
        main.main()


def test_frame_limit_inferred_from_predictions(tmp_path):
    csv_file = tmp_path / "pred.csv"
    csv_file.write_text("camera_id,frame_id,global_id\n3,2,4\n3,19,4\n")
    assert inferred_max_frames(csv_file) == 20
