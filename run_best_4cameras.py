"""Показать победивший MTMCT с ночного поиска: один запуск кнопкой ▶.

Берёт train-сцену с готовыми bbox/embedding из runs/overnight_v1,
пересчитывает association с лучшими найденными параметрами и рисует видео.
Не запускает YOLO заново и не меняет исходные кэши.
"""
from __future__ import annotations
from pathlib import Path
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENT_NAME = "overnight_v1"
SCENE = ("train", "Warehouse_000")
RENDER_VIDEO = True
EVALUATE = True
MAX_FRAMES = 900
SCALE = 0.35


def main():
    from mtmct.association import Config
    from mtmct.link import run as link_run
    from mtmct.occlusion_repair import RepairConfig
    from mtmct.evaluate import eval_2d, save_report
    from mtmct.visualize import render

    base = ROOT / "runs" / EXPERIMENT_NAME
    best_path = base / "best_train_config.json"
    if not best_path.is_file():
        raise FileNotFoundError("Сначала запусти run_nightly_optimize.py: " + str(best_path))
    best = json.loads(best_path.read_text(encoding="utf-8"))
    scene_tag = f"{SCENE[0]}_{SCENE[1]}"
    camsets = json.loads((base / "selected_cameras.json").read_text(encoding="utf-8"))
    cameras = camsets[scene_tag]
    model = best["model"]
    dtrial = best["detector_trial"]
    import run_nightly_optimize as night
    detector_id = night.detector_id(dtrial)
    common = base / "detector" / detector_id / scene_tag
    source = base / "features" / detector_id / model / scene_tag
    out = base / "best_video" / scene_tag
    out.mkdir(parents=True, exist_ok=True)
    for origin, name in ((common, "observations.csv"), (source, "local_tracklets.json"),
                         (source, "manifest.json")):
        dst = out / name
        if dst.exists():
            dst.unlink()
        try:
            os.link(origin / name, dst)
        except OSError:
            shutil.copy2(origin / name, dst)

    p = best["params"]
    cfg = Config(appearance_weight=p["appearance_weight"], spatial_weight=p["spatial_weight"],
                 temporal_weight=p["temporal_weight"], min_similarity=p["min_similarity"],
                 max_cost=p["max_cost"], max_gap_seconds=p["max_gap_seconds"],
                 max_speed_mps=p["max_speed_mps"], spatial_slack_m=p["spatial_slack_m"],
                 overlap_distance_m=p["overlap_distance_m"], use_geometry=p["use_geometry"],
                 use_time=p["use_time"], fps=30.0)
    repair = RepairConfig(duplicate_min_iou=p["duplicate_min_iou"],
                          duplicate_min_similarity=p["duplicate_min_similarity"],
                          max_gap_frames=p["repair_max_gap_frames"],
                          min_witnessed_similarity=p["min_witnessed_similarity"],
                          min_unwitnessed_similarity=p["min_unwitnessed_similarity"],
                          strong_witness_min_similarity=p["strong_witness_min_similarity"],
                          strong_witness_max_distance_m=p["strong_witness_max_distance_m"])
    print("Модель:", model, "Камеры:", cameras, "\nПараметры:", p)
    _, mapping, edges, pred, _ = link_run(
        out, cfg, expected_detector=night.YOLO_MODEL,
        matcher=p["matcher"], allow_same_camera_fragments=p["allow_same_camera_fragments"],
        memory=p["memory"], memory_similarity=p["memory_similarity"],
        correct_identities=p["correct_identities"],
        repair_occlusions=p["repair_occlusions"], repair_config=repair)
    location = ROOT / "data" / "MTMC_Tracking_2026" / SCENE[0] / SCENE[1]
    if EVALUATE:
        report = eval_2d(pred, location / "ground_truth.json", cameras, MAX_FRAMES)
        save_report(report, out / "metrics.json")
        print("IDF1:", report["global"]["IDF1"])
    if RENDER_VIDEO:
        render(location, pred, out / "mosaic.mp4", MAX_FRAMES, 1, SCALE, cameras)
    (out / "best_configuration.json").write_text(json.dumps(best, indent=2, ensure_ascii=False),encoding="utf-8")
    print("Готово:", out)


if __name__ == "__main__":
    main()
