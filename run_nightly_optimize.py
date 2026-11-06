"""Ночной подбор MTMCT по РАЗНЫМ сценам NVIDIA: запускай как обычный Python-файл в VS Code.

Зависимости: существующий пакет mtmct (repair v2 и OSNet variants patch),
и run_compare_all_osnet.py рядом с этим файлом (для честного ReID пересчёта).

Стадии: (1) YOLO/ByteTrack на train-сценах, (2) OSNet на неизменных bbox,
(3) random + local search по association/memory/occlusion-repair,
(4) единственная заключительная проверка победителя на val-сценах.

Не подбирает по val и не подменяет официальный NVIDIA 3D HOTA.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ========================== ИЗМЕНЯЕМЫЕ НАСТРОЙКИ ===========================
DATA_ROOT = ROOT / "data" / "MTMC_Tracking_2026"
TRAIN_SCENES = (("train", "Warehouse_000"), ("train", "Warehouse_002"))
VALIDATION_SCENES = (("val", "Warehouse_020"), ("val", "Warehouse_022"))
CAMERAS_PER_SCENE = 4
# Для фиксированного набора камер одной сцены можно указать, например, (0, 1, 3, 15).
# Для разных сцен рекомендуется None — автоматический выбор четырёх камер из GT.
FIXED_CAMERAS = None
MAX_FRAMES = 900
DEVICE = "cuda:0"
YOLO_MODEL = "yolo26l.pt"
REID_MODELS = ("osnet_x1_0", "osnet_ibn_x1_0", "osnet_ain_x1_0")
SAMPLE_EVERY = 5
EVAL_IOU = 0.50
SEED = 42
EXPERIMENT_NAME = "overnight_v1"  # изменить, если поменял настройки
RESUME = True
AUTO_DOWNLOAD = True  # скачать отсутствующие GT, calibration и выбранные MP4 с Hugging Face
KEEP_TRIAL_OUTPUTS = False  # True может занять много гигабайт
# Необязательные контрольные варианты. Ограничь список, если ночь короткая.
DETECTOR_TRIALS = (
    # imgsz, conf, track_high_thresh, track_low_thresh, new_track_thresh, track_buffer, match_thresh
    (1536, 0.10, 0.15, 0.03, 0.10, 45, 0.80),
    (1920, 0.10, 0.15, 0.03, 0.10, 45, 0.80),
    (1920, 0.07, 0.12, 0.02, 0.08, 60, 0.82),
    (1536, 0.07, 0.12, 0.02, 0.08, 60, 0.82),
    (1280, 0.10, 0.20, 0.05, 0.15, 45, 0.80),
    (1920, 0.15, 0.20, 0.05, 0.15, 60, 0.85),
)
TOP_DETECTORS = 1  # сколько лучших YOLO/ByteTrack конфигураций переводить в ReID/association
RANDOM_ASSOCIATION_TRIALS = 80  # испытаний на КАЖДУЮ ReID-модель
LOCAL_REFINEMENT_TRIALS = 25    # уточнений возле лучших параметров на КАЖДУЮ модель
# Неизвестную геометрию нельзя подставлять. Для режима с геометрией необходим calibration.json.
TRY_GEOMETRY = True
# ===========================================================================

RUN_ROOT = ROOT / "runs" / EXPERIMENT_NAME
RUN_ROOT.mkdir(parents=True, exist_ok=True)


def dump(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def write_jsonl(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def existing_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if "id" in row:
                    rows[row["id"]] = row
            except json.JSONDecodeError:
                continue  # последняя строка могла не успеть завершиться
    return rows


def scene_dir(scene: tuple[str, str]) -> Path:
    return DATA_ROOT / scene[0] / scene[1]


def scene_key(scene: tuple[str, str]) -> str:
    return f"{scene[0]}_{scene[1]}"


def camera_selection(scene: tuple[str, str]) -> list[int]:
    """Выбрать камеры с достаточной активностью и межкамерными ID, без зависимости от номера."""
    from mtmct.nvidia import NvidiaDatasetAdapter
    from mtmct.geometry import camera_number
    location = scene_dir(scene)
    videos = {camera_number(p.stem) for p in (location / "videos").glob("Camera_*.mp4")}
    if FIXED_CAMERAS is not None:
        cams = list(FIXED_CAMERAS)
        if len(cams) != CAMERAS_PER_SCENE or (not AUTO_DOWNLOAD and not set(cams).issubset(videos)):
            raise FileNotFoundError(f"{scene}: не все фиксированные камеры {cams} есть среди {sorted(videos)}")
        return sorted(cams)
    gt = location / "ground_truth.json"
    if not gt.is_file():
        raise FileNotFoundError(f"Нужна разметка для выбора камер и IDF1: {gt}")
    ids = defaultdict(set)
    counts = Counter()
    # При AUTO_DOWNLOAD=True MP4 могут отсутствовать на этапе выбора камер.
    for ob in NvidiaDatasetAdapter(gt).observations(max_frames=MAX_FRAMES):
        if not AUTO_DOWNLOAD and ob.camera_id not in videos:
            continue
        counts[ob.camera_id] += 1
        ids[ob.camera_id].add(ob.gt_object_id)
    eligible = {cam for cam in counts if counts[cam] > 0}
    if len(eligible) < CAMERAS_PER_SCENE:
        raise RuntimeError(
            f"{scene}: найдено лишь {len(eligible)} камер с разметкой в первых "
            f"{MAX_FRAMES} кадрах (локальных MP4: {len(videos)})"
        )
    picked = [max(eligible, key=lambda c: (counts[c], -c))]
    while len(picked) < CAMERAS_PER_SCENE:
        rest = eligible - set(picked)
        best = max(rest, key=lambda c: (
            sum(len(ids[c] & ids[p]) for p in picked),
            len(ids[c]), counts[c], -c))
        picked.append(best)
    return sorted(picked)


def prepare_scenes(scenes):
    cameras = {}
    print("\nПроверка файлов и камер:", flush=True)
    from mtmct.download import download
    hf_data_root = DATA_ROOT.parent  # папка data/
    for scene in scenes:
        loc = scene_dir(scene)
        gt = loc / "ground_truth.json"
        calib = loc / "calibration.json"
        if not (gt.exists() and calib.exists()) and AUTO_DOWNLOAD:
            print("  Загружаю метаданные:", scene_key(scene), flush=True)
            download(scene[0], scene[1], [], hf_data_root, True, videos=False)
        if not gt.is_file() or not calib.is_file():
            raise FileNotFoundError(f"Не найдены GT или calibration для {scene}: {loc}")
        selection = camera_selection(scene)
        missing = [c for c in selection if not (loc / "videos" / f"Camera_{c:04d}.mp4").exists()]
        if missing and AUTO_DOWNLOAD:
            print("  Загружаю видео камер:", missing, scene_key(scene), flush=True)
            download(scene[0], scene[1], missing, hf_data_root, False, videos=True)
        for c in selection:
            p = loc / "videos" / f"Camera_{c:04d}.mp4"
            if not p.is_file():
                raise FileNotFoundError(f"Не найдено видео {p}")
        cameras[scene_key(scene)] = selection
        print(f"  {scene_key(scene)}: камеры {selection}", flush=True)
    return cameras


def initial_settings():
    return {
        "train": [list(x) for x in TRAIN_SCENES],
        "validation": [list(x) for x in VALIDATION_SCENES],
        "max_frames": MAX_FRAMES, "camera_count": CAMERAS_PER_SCENE,
        "fixed_cameras": list(FIXED_CAMERAS) if FIXED_CAMERAS is not None else None, "device": DEVICE,
        "yolo": YOLO_MODEL, "reid_models": list(REID_MODELS),
        "sample_every": SAMPLE_EVERY, "detector_trials": [list(x) for x in DETECTOR_TRIALS],
        "seed": SEED, "eval_iou": EVAL_IOU,
        "top_detectors": TOP_DETECTORS,
        "association_trials": RANDOM_ASSOCIATION_TRIALS,
        "local_trials": LOCAL_REFINEMENT_TRIALS,
        "try_geometry": TRY_GEOMETRY, "auto_download": AUTO_DOWNLOAD,
    }


def preflight():
    if not (ROOT / "mtmct" / "tracking.py").is_file():
        raise RuntimeError("Положи скрипт рядом с папкой mtmct/")
    if not (ROOT / "run_compare_all_osnet.py").is_file():
        raise RuntimeError("Положи run_compare_all_osnet.py рядом — он нужен для одинаковых crop ReID")
    import ijson
    import torch
    import cv2
    import run_compare_all_osnet as compare
    from mtmct.weights import MODELS, download_weights
    for name in REID_MODELS:
        if name not in MODELS:
            raise RuntimeError(f"В mtmct/weights.py не зарегистрирована модель {name}; установи OSNet patch")
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Нет CUDA GPU. Проверь Python interpreter в VS Code")
    print("Python:", sys.executable, "| torch:", torch.__version__, "| cv2:", cv2.__version__)
    paths = {}
    for model in REID_MODELS:
        paths[model] = Path(download_weights(model))
        print("ReID:", model, "->", paths[model])
    return compare, paths


def detector_id(trial) -> str:
    return "det_" + digest({"trial": trial, "yolo": YOLO_MODEL, "sample_every": SAMPLE_EVERY, "frames": MAX_FRAMES})


def make_tracker_yaml(trial) -> Path:
    img, conf, high, low, new, buffer, match = trial
    if not (0 <= low <= high <= 1 and 0 <= new <= 1 and 0 < conf <= 1):
        raise ValueError(f"Недопустимые пороги ByteTrack/YOLO: {trial}")
    if buffer < 1 or not (0 < match <= 1):
        raise ValueError(f"Недопустимая конфигурация ByteTrack: {trial}")
    tracker_file = RUN_ROOT / "trackers" / f"{detector_id(trial)}.yaml"
    tracker_file.parent.mkdir(parents=True, exist_ok=True)
    tracker_file.write_text(
        "tracker_type: bytetrack\n"
        f"track_high_thresh: {high}\ntrack_low_thresh: {low}\n"
        f"new_track_thresh: {new}\ntrack_buffer: {buffer}\n"
        f"match_thresh: {match}\nfuse_score: true\n", encoding="utf-8")
    return tracker_file


def shared_dir(trial, scene):
    return RUN_ROOT / "detector" / detector_id(trial) / scene_key(scene)


def set_compare(compare, scene, cams, shared, trial):
    # Применяем параметры к ранее протестированной реализации извлечения одинаковых кропов.
    compare.CAMERAS = tuple(cams)
    compare.SCENE_DIR = scene_dir(scene)
    compare.SHARED_DIR = shared
    compare.MAX_FRAMES = MAX_FRAMES
    compare.SAMPLE_EVERY = SAMPLE_EVERY
    compare.IMGSZ = int(trial[0])
    compare.CONF = float(trial[1])
    compare.YOLO_MODEL = YOLO_MODEL
    compare.TRACKER = str(make_tracker_yaml(trial).resolve())
    compare.DEVICE = DEVICE
    compare.RESUME = RESUME
    compare.SEED = SEED


def detection_predictions(shared: Path) -> Path:
    src = shared / "observations.csv"
    dst = shared / "detection_only_predictions.csv"
    if dst.exists():
        return dst
    with src.open(encoding="utf-8", newline="") as r, dst.open("w", encoding="utf-8", newline="") as w:
        reader = csv.DictReader(r)
        writer = csv.DictWriter(w, fieldnames=list(reader.fieldnames or []) + ["global_id"])
        writer.writeheader()
        for row in reader:
            row["global_id"] = int(row["camera_id"]) * 1000000 + int(row["local_id"])
            writer.writerow(row)
    return dst


def run_detection(compare, trial, scene, cams) -> dict:
    from mtmct.evaluate import eval_2d
    shared = shared_dir(trial, scene)
    set_compare(compare, scene, cams, shared, trial)
    # Если компьютер выключился на середине GPU-этапа, не доверяем частичному кэшу.
    needed = [shared / x for x in ("observations.csv", "local_tracklets.json", "manifest.json", "performance.json")]
    if shared.exists() and any(shared.iterdir()) and not all(x.is_file() for x in needed):
        print("  Очищаю незавершённый кэш:", shared, flush=True)
        shutil.rmtree(shared)
    shared.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    compare.create_shared_tracking()
    predictions = detection_predictions(shared)
    report = eval_2d(predictions, scene_dir(scene) / "ground_truth.json", cams,
                     MAX_FRAMES, threshold=EVAL_IOU)
    score = report["detection"]["F1"]
    result = {
        "id": f"{detector_id(trial)}_{scene_key(scene)}", "phase": "detector",
        "scene": scene_key(scene), "trial": list(trial),
        "detection_f1": score,
        "detection_recall": report["detection"]["recall"],
        "detection_precision": report["detection"]["precision"],
        "seconds": round(time.perf_counter() - start, 1),
        "shared": str(shared), "status": "ok",
    }
    dump(shared / "detection_metrics.json", report)
    return result


def detector_search(compare, cameras: dict) -> list[tuple]:
    log = RUN_ROOT / "detector_search.jsonl"
    completed = existing_jsonl(log)
    output = []
    for trial in DETECTOR_TRIALS:
        scores = []
        for scene in TRAIN_SCENES:
            k = f"{detector_id(trial)}_{scene_key(scene)}"
            try:
                if RESUME and completed.get(k, {}).get("status") == "ok":
                    row = completed[k]
                else:
                    print("\n[ДЕТЕКЦИЯ]", trial, scene_key(scene), flush=True)
                    row = run_detection(compare, trial, scene, cameras[scene_key(scene)])
                    write_jsonl(log, row)
                scores.append(row)
            except Exception as e:
                err = {"id": k, "status": "error", "error": str(e), "traceback": traceback.format_exc()}
                write_jsonl(log, err)
                print("Ошибка детекции:", e, flush=True)
                break
        if len(scores) == len(TRAIN_SCENES):
            score = sum(x["detection_f1"] for x in scores) / len(scores)
            recall = sum(x["detection_recall"] for x in scores) / len(scores)
            output.append((score, recall, trial))
            print(f"Детектор {detector_id(trial)}: mean F1={score:.5f} recall={recall:.5f}", flush=True)
    if not output:
        raise RuntimeError("Ни одна конфигурация YOLO/ByteTrack не прошла")
    output.sort(key=lambda x: (x[0], x[1]), reverse=True)
    dump(RUN_ROOT / "detector_ranking.json", [
        {"mean_detection_f1": a, "mean_detection_recall": b,
         "trial": list(trial), "id": detector_id(trial)} for a,b,trial in output])
    print("\nПобедитель детекции:", output[0])
    return [x[2] for x in output[:TOP_DETECTORS]]


def reid_cache(compare, weights, trial, model, scene, cams):
    from mtmct.cache import validate_cache
    shared = shared_dir(trial, scene)
    target = RUN_ROOT / "features" / detector_id(trial) / model / scene_key(scene)
    features = target / "local_tracklets.json"
    manifest = target / "manifest.json"
    if RESUME and features.is_file() and manifest.is_file() and (target / "reid_performance.json").is_file():
        return target
    print(f"\n[ReID] {model} / {scene_key(scene)}", flush=True)
    target.mkdir(parents=True, exist_ok=True)
    set_compare(compare, scene, cams, shared, trial)
    # Хотя изображения совпадают, каждую модель запускаем независимо на тех же bbox.
    obs = compare.observations_by_cam()
    local_tracks = load(shared / "local_tracklets.json")
    m = load(shared / "manifest.json")
    processed, timing = compare.extract_features(model, weights[model], obs, local_tracks)
    dump(features, processed)
    dump(manifest, compare.make_manifest(model, weights[model], m, timing["feature_dim"]))
    dump(target / "reid_performance.json", timing)
    return target


def sample_param(rng: random.Random, model: str, use_geometry: bool):
    """Одна конфигурация; пространственные параметры влияют только при geometry=True."""
    use_geom = use_geometry and TRY_GEOMETRY and rng.choice((True, False))
    ap = rng.uniform(0.60, 1.0)
    sp = rng.uniform(0.02, 0.30) if use_geom else 0.0
    tp = rng.uniform(0.0, 0.16)
    total = ap + sp + tp
    matcher = rng.choices(("baseline", "graph"), weights=(8, 2))[0]
    return {
        "matcher": matcher,
        "appearance_weight": round(ap / total, 4),
        "spatial_weight": round(sp / total, 4),
        "temporal_weight": round(tp / total, 4),
        "min_similarity": round(rng.uniform(0.69, 0.93), 3),
        "max_cost": round(rng.uniform(0.045, 0.27), 3),
        "max_gap_seconds": rng.choice((5., 12., 20., 40., 90.)),
        "max_speed_mps": round(rng.uniform(1.0, 5.0), 2),
        "spatial_slack_m": round(rng.uniform(0.5, 2.5), 2),
        "overlap_distance_m": round(rng.uniform(0.7, 4.0), 2),
        "use_geometry": use_geom,
        "use_time": rng.random() < 0.90,
        "allow_same_camera_fragments": rng.choice((True, False)),
        "correct_identities": matcher == "graph" and rng.random() < 0.50,
        "memory": rng.choice((True, False)),
        "memory_similarity": round(rng.uniform(0.87, 0.985), 3),
        "repair_occlusions": rng.choice((True, False)),
        "duplicate_min_iou": round(rng.uniform(0.42, 0.72), 2),
        "duplicate_min_similarity": round(rng.uniform(0.60, 0.88), 2),
        "repair_max_gap_frames": rng.choice((15, 30, 45, 60)),
        "min_witnessed_similarity": round(rng.uniform(0.64, 0.85), 2),
        "min_unwitnessed_similarity": round(rng.uniform(0.88, 0.98), 2),
        "strong_witness_min_similarity": round(rng.uniform(0.60, 0.76), 2),
        "strong_witness_max_distance_m": round(rng.uniform(0.45, 1.0), 2),
    }


def baseline_params():
    return {
        "matcher":"baseline", "appearance_weight":0.95,"spatial_weight":0.0,
        "temporal_weight":0.05,"min_similarity":0.80,"max_cost":0.125,
        "max_gap_seconds":12.0,"max_speed_mps":3.5,"spatial_slack_m":1.5,
        "overlap_distance_m":2.0,"use_geometry":False,"use_time":True,
        "allow_same_camera_fragments":True,"correct_identities":False,
        "memory":True,"memory_similarity":0.95,"repair_occlusions":True,
        "duplicate_min_iou":0.45,"duplicate_min_similarity":0.72,
        "repair_max_gap_frames":30,"min_witnessed_similarity":0.72,
        "min_unwitnessed_similarity":0.92,"strong_witness_min_similarity":0.65,
        "strong_witness_max_distance_m":0.65,
    }


def mutate(rng: random.Random, parent: dict):
    q = dict(parent)
    floating = {
        "min_similarity": (0.60, 0.96, .025),
        "max_cost": (.025, .35, .02),
        "memory_similarity": (.85, .99, .016),
        "max_gap_seconds": (2., 150., 8.),
        "appearance_weight": (.4, 1., .08),
        "spatial_weight": (0., .5, .05),
        "temporal_weight": (0., .3, .04),
        "max_speed_mps": (.7, 6., .45),
        "overlap_distance_m": (.3, 5., .3),
        "min_witnessed_similarity": (.55, .9, .04),
        "min_unwitnessed_similarity": (.8, .99, .025),
        "strong_witness_min_similarity": (.50, .85, .03),
    }
    keys = rng.sample(list(floating), k=rng.choice((1, 2, 3)))
    for k in keys:
        low, high, sigma = floating[k]
        q[k] = round(min(high, max(low, float(q[k]) + rng.gauss(0, sigma))), 4)
    if not q["use_geometry"]:
        q["spatial_weight"] = 0.0
    if sum(q[k] for k in ("appearance_weight","spatial_weight","temporal_weight")) == 0:
        q["appearance_weight"] = 0.95
    return q


def hardlink_or_copy(src, dst):
    try:
        os.link(src, dst)  # На NTFS безопасно: локальные треки только читаются.
    except OSError:
        shutil.copy2(src, dst)


def evaluate_association(trial, model, scene, cams, params: dict) -> dict:
    from mtmct.association import Config
    from mtmct.link import run as link_run
    from mtmct.occlusion_repair import RepairConfig
    from mtmct.evaluate import eval_2d
    uid = "assoc_" + digest({"model": model, "detector": detector_id(trial), "scene": scene_key(scene), "p": params})
    target = RUN_ROOT / "scratch" / uid
    source = RUN_ROOT / "features" / detector_id(trial) / model / scene_key(scene)
    shared = shared_dir(trial, scene)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for base, name in ((shared, "observations.csv"), (source, "local_tracklets.json"),
                       (source, "manifest.json")):
        hardlink_or_copy(base / name, target / name)
    cfg = Config(
        appearance_weight=params["appearance_weight"],
        spatial_weight=params["spatial_weight"],
        temporal_weight=params["temporal_weight"],
        min_similarity=params["min_similarity"],max_cost=params["max_cost"],
        max_gap_seconds=params["max_gap_seconds"],max_speed_mps=params["max_speed_mps"],
        spatial_slack_m=params["spatial_slack_m"],overlap_distance_m=params["overlap_distance_m"],
        use_geometry=params["use_geometry"],use_time=params["use_time"],
        fps=30.0)
    rep_cfg = RepairConfig(
        duplicate_min_iou=params["duplicate_min_iou"],
        duplicate_min_similarity=params["duplicate_min_similarity"],
        max_gap_frames=params["repair_max_gap_frames"],
        min_witnessed_similarity=params["min_witnessed_similarity"],
        min_unwitnessed_similarity=params["min_unwitnessed_similarity"],
        strong_witness_min_similarity=params["strong_witness_min_similarity"],
        strong_witness_max_distance_m=params["strong_witness_max_distance_m"])
    try:
        start = time.perf_counter()
        _, mapping, edges, pred_path, _ = link_run(
            target, cfg, expected_detector=YOLO_MODEL,
            matcher=params["matcher"],
            allow_same_camera_fragments=params["allow_same_camera_fragments"],
            memory=params["memory"],memory_similarity=params["memory_similarity"],
            correct_identities=params["correct_identities"],
            repair_occlusions=params["repair_occlusions"],repair_config=rep_cfg)
        report = eval_2d(pred_path, scene_dir(scene) / "ground_truth.json", cams,
                         MAX_FRAMES, threshold=EVAL_IOU)
        result = {
            "global_idf1": report["global"]["IDF1"],
            "pairwise_f1": report["pairwise_association"]["F1"],
            "pairwise_precision": report["pairwise_association"]["precision"],
            "detection_recall": report["detection"]["recall"],
            "elapsed_s": round(time.perf_counter()-start,2),
            "global_ids": len(set(mapping.values())),
            "scene": scene_key(scene),
            "status": "ok",
        }
        if KEEP_TRIAL_OUTPUTS:
            dump(target / "metrics.json", report)
        return result
    finally:
        if not KEEP_TRIAL_OUTPUTS:
            shutil.rmtree(target, ignore_errors=True)


def aggregate(scene_rows):
    # Среднее по сценам (а не по числу боксов) — чтобы большая сцена не доминировала.
    return sum(float(x["global_idf1"]) for x in scene_rows) / len(scene_rows)


def search_association(trial, model, scenes, cameras, rng, random_trials, local_trials):
    path = RUN_ROOT / f"search_{detector_id(trial)}_{model}.jsonl"
    done = existing_jsonl(path)
    successes = [r for r in done.values() if r.get("status") == "ok"]
    def try_params(params, label):
        pid = "p_" + digest({"model":model, "detector":detector_id(trial),"params":params})
        if pid in done:
            return done[pid]
        rows = []
        try:
            for sc in scenes:
                rows.append(evaluate_association(trial, model, sc, cameras[scene_key(sc)], params))
            out = {"id": pid, "status": "ok", "detector": detector_id(trial),
                   "model": model, "label":label, "params":params,
                   "train_idf1": round(aggregate(rows), 6),
                   "train_pairwise_f1": round(sum(x["pairwise_f1"] for x in rows)/len(rows),6),
                   "scenes": rows}
            successes.append(out)
        except Exception as e:
            out = {"id": pid, "status": "error", "model":model, "params":params,
                   "error": str(e), "traceback": traceback.format_exc()}
            print("  Ошибка association", model, str(e), flush=True)
        write_jsonl(path, out)
        done[pid] = out
        return out

    print(f"\n[ASSOCIATION] {model}, detector {detector_id(trial)}", flush=True)
    try_params(baseline_params(), "reference")
    for i in range(random_trials):
        p = sample_param(rng, model, True)
        result = try_params(p, "random")
        if i % 10 == 0:
            print("  random", i+1, "/", random_trials,
                  "best", max((r["train_idf1"] for r in successes), default=None), flush=True)
    if not successes:
        raise RuntimeError(f"Нет успешных вариантов association для {model}")
    # Local search вокруг 3 лучших случайных конфигураций.
    elite = sorted(successes, key=lambda r:r["train_idf1"], reverse=True)[:3]
    for i in range(local_trials):
        base = elite[i % len(elite)]
        result = try_params(mutate(rng, base["params"]), "refinement")
        if result.get("status") == "ok" and result["train_idf1"] > min(x["train_idf1"] for x in elite):
            elite = sorted(elite + [result], key=lambda r:r["train_idf1"], reverse=True)[:3]
    return sorted(successes, key=lambda r:(r["train_idf1"],r["train_pairwise_f1"]),reverse=True)


def validation_of_winner(compare, weights, best, cameras):
    trial = tuple(best["detector_trial"])
    model = best["model"]
    params = best["params"]
    result_path = RUN_ROOT / "validation_results.json"
    if RESUME and result_path.is_file():
        return load(result_path)
    rows = []
    for scene in VALIDATION_SCENES:
        print(f"\n[НЕЗАВИСИМАЯ ПРОВЕРКА] {scene_key(scene)}: {model}", flush=True)
        run_detection(compare, trial, scene, cameras[scene_key(scene)])
        reid_cache(compare, weights, trial, model, scene, cameras[scene_key(scene)])
        rows.append(evaluate_association(trial, model, scene, cameras[scene_key(scene)], params))
    result = {"winner_selected_on_train_only": best, "validation_scenes": rows,
              "validation_mean_idf1": round(aggregate(rows),6),
              "WARNING": "Validation used once, for reporting only. Do not retune on these scenes."}
    dump(result_path, result)
    return result


def summarize(best, validation):
    lines = ["НОЧНАЯ ОПТИМИЗАЦИЯ MTMCT", "="*68,
             f"Training scenes: {TRAIN_SCENES}", f"Holdout scenes: {VALIDATION_SCENES}",
             f"Frames per camera: {MAX_FRAMES}; cameras per scene: {CAMERAS_PER_SCENE}",
             f"Best model: {best['model']}",
             f"Best detector: {best['detector_trial']}",
             f"Train mean Global IDF1: {best['train_idf1']:.6f}",
             f"Holdout mean Global IDF1: {validation['validation_mean_idf1']:.6f}",
             "", "Full best association configuration:",
             json.dumps(best["params"],indent=2,ensure_ascii=False),
             "", "Holdout per-scene results:"]
    for r in validation["validation_scenes"]:
        lines.append(f"  {r['scene']}: IDF1 {r['global_idf1']:.6f} pairwise F1 {r['pairwise_f1']:.6f}")
    lines += ["", "This is custom global-2d-idf1-v1, NOT official 3D HOTA.",
              "Do not tune against the validation scenes after opening the results.",
              "More datasets (CITIC, SCOUT) need separate GT format adapters."]
    (RUN_ROOT / "summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    print("Ночной MTMCT оптимизатор: train по независимым сценам + val holdout", flush=True)
    settings_file = RUN_ROOT / "optimizer_settings.json"
    settings = initial_settings()
    if settings_file.exists():
        if load(settings_file) != settings:
            raise RuntimeError("Для изменения параметров задай другое EXPERIMENT_NAME, иначе смешаются кэши")
    else:
        dump(settings_file, settings)
    rng = random.Random(SEED)
    compare, weights = preflight()
    cameras = prepare_scenes(tuple(dict.fromkeys(TRAIN_SCENES + VALIDATION_SCENES)))
    dump(RUN_ROOT / "selected_cameras.json", cameras)
    train_detectors = detector_search(compare, cameras)
    all_rankings = []
    for detector in train_detectors:
        for model in REID_MODELS:
            for sc in TRAIN_SCENES:
                reid_cache(compare, weights, detector, model, sc, cameras[scene_key(sc)])
            ranked = search_association(detector, model, TRAIN_SCENES, cameras,
                                        rng, RANDOM_ASSOCIATION_TRIALS,
                                        LOCAL_REFINEMENT_TRIALS)
            for row in ranked:
                row["detector_trial"] = list(detector)
            all_rankings.extend(ranked)
            dump(RUN_ROOT / "train_leaderboard.json", sorted(all_rankings,
                 key=lambda r:r["train_idf1"],reverse=True)[:50])
    if not all_rankings:
        raise RuntimeError("Нет успешного набора параметров")
    best_by_model = {}
    for model in REID_MODELS:
        candidates = [row for row in all_rankings if row["model"] == model]
        if candidates:
            best_by_model[model] = max(candidates, key=lambda r:(r["train_idf1"],r["train_pairwise_f1"]))
    dump(RUN_ROOT / "best_by_model.json", best_by_model)
    best = max(all_rankings, key=lambda r:(r["train_idf1"],r["train_pairwise_f1"]))
    dump(RUN_ROOT / "best_train_config.json", best)
    print("\nВыбран лучший вариант по TRAIN. Validation НЕ используется для подбора.",flush=True)
    validation = validation_of_winner(compare, weights, best, cameras)
    summarize(best, validation)
    print(f"\nГОТОВО. Отчёты: {RUN_ROOT / 'summary.txt'}",flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        file = RUN_ROOT / "FAILED_TRACEBACK.txt"
        file.write_text(traceback.format_exc(), encoding="utf-8")
        print("\nОшибка! См. файл:", file, "\n",traceback.format_exc(),flush=True)
        raise
