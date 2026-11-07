"""Честное сравнение трёх OSNet в MTMCT. Запускай ▶ Run Python File из VS Code.

Сначала YOLO+ByteTrack выполняются ОДИН РАЗ для всех моделей (с лёгким color
extractor, его признаки затем ПОЛНОСТЬЮ заменяются). Затем все три OSNet
получают ОДИНАКОВЫЕ RGB-кропы одних и тех же bbox/кадров. Для каждой модели
считаются (1) чистый association, (2) association+memory+occlusion repair.
Метрики сравниваются с одним и тем же GT. Файлы разных моделей не смешиваются.

Требует твой mtmct с поддержкой osnet_x1_0 / osnet_ibn_x1_0 / osnet_ain_x1_0
в mtmct/weights.py, torchreid, opencv, ultralytics и PyTorch.
"""
from __future__ import annotations

from pathlib import Path as _LauncherPath
import sys as _launcher_sys
_launcher_sys.path.insert(0, str(_LauncherPath(__file__).resolve().parents[1]))
_launcher_sys.path.insert(0, str(_LauncherPath(__file__).resolve().parent))
from collections import defaultdict
from pathlib import Path
import csv
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# =========================== НАСТРОЙКИ ===========================
CAMERAS = (0, 1, 3, 15)      # Для быстрого A/B можно заменить на (3, 15)
MAX_FRAMES = 900             # Первые 900 кадров с КАЖДОЙ камеры
IMGSZ = 1920
CONF = 0.10
SAMPLE_EVERY = 5
YOLO_MODEL = "yolo26l.pt"
TRACKER = "bytetrack.yaml"
DEVICE = "cuda:0"           # Обычно RTX 3070 Ti
MODELS = ("osnet_x1_0", "osnet_ibn_x1_0", "osnet_ain_x1_0")

# Одинаковые для всех моделей настройки глобального сопоставления
MIN_SIMILARITY = 0.80
MAX_COST = 0.125
MEMORY_SIMILARITY = 0.95
EVAL_IOU = 0.50

SPLIT = "train"
SCENE = "Warehouse_000"
DATA_ROOT = ROOT / "data"
SCENE_DIR = DATA_ROOT / "MTMC_Tracking_2026" / SPLIT / SCENE
GT_FILE = SCENE_DIR / "ground_truth.json"

RUN_LABEL = "v1"             # Если захочешь заново экспериментировать: v2, v3 ...
RENDER_WINNER = True         # Создать mosaic.mp4 для лучшей full-модели
RENDER_ALL = False          # True — видео для каждой модели и обоих режимов
RENDER_SCALE = 0.35         # 4 камеры рисуются в одну строку
RESUME = True               # Использовать готовые этапы после прерывания
SEED = 42
# ================================================================

CAM_TAG = "_".join(map(str, sorted(CAMERAS)))
RUN_DIR = ROOT / "runs" / f"reid_fair_cam{CAM_TAG}_{MAX_FRAMES}_{RUN_LABEL}"
SHARED_DIR = RUN_DIR / "shared_tracking"


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def filesha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def settings_dict():
    return {"cameras": list(CAMERAS), "max_frames": MAX_FRAMES,
            "imgsz": IMGSZ, "conf": CONF, "sample_every": SAMPLE_EVERY,
            "yolo_model": YOLO_MODEL, "tracker": TRACKER,
            "models": list(MODELS), "min_similarity": MIN_SIMILARITY,
            "max_cost": MAX_COST, "memory_similarity": MEMORY_SIMILARITY,
            "eval_iou": EVAL_IOU, "split": SPLIT, "scene": SCENE,
            "scene_dir": str(SCENE_DIR.resolve()), "gt": str(GT_FILE.resolve()),
            "seed": SEED, "pipeline": "same-tracks-compare-v1"}


def preflight():
    if len(CAMERAS) < 2 or len(CAMERAS) != len(set(CAMERAS)):
        raise ValueError("Нужно хотя бы 2 разных номера камер")
    if MAX_FRAMES <= 0 or SAMPLE_EVERY <= 0:
        raise ValueError("MAX_FRAMES и SAMPLE_EVERY должны быть положительными")
    for c in CAMERAS:
        p = SCENE_DIR / "videos" / f"Camera_{c:04d}.mp4"
        if not p.is_file():
            raise FileNotFoundError(f"Отсутствует видео: {p}")
    if not (SCENE_DIR / "calibration.json").is_file():
        raise FileNotFoundError("Текущий tracking.py ожидает calibration.json в папке сцены")
    if not GT_FILE.is_file():
        raise FileNotFoundError(f"Для честного сравнения требуется Ground Truth: {GT_FILE}")
    if not (ROOT / "mtmct" / "tracking.py").is_file():
        raise FileNotFoundError("Файл нужно положить в корень проекта, рядом с mtmct/")
    import cv2
    import torch
    from mtmct.weights import MODELS as KNOWN_WEIGHTS, download_weights
    for model in MODELS:
        if model not in KNOWN_WEIGHTS:
            raise RuntimeError(
                f"В mtmct/weights.py нет модели {model}. Установи патч поддержки OSNet-вариантов")
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна в этом окружении. Проверь выбранный Python в VS Code")
    print("Окружение:", "OpenCV", cv2.__version__, "PyTorch", torch.__version__)
    print("Устройство:", torch.cuda.get_device_name(0) if DEVICE.startswith("cuda") else DEVICE)
    # Скачать/найти ВСЕ веса ДО дорогого YOLO-этапа, чтобы не прерываться в середине.
    weights = {}
    for model in MODELS:
        path = Path(download_weights(model)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        weights[model] = path
        print("Веса", model, "->", path.name)
    return weights


def prepare_root():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    p = RUN_DIR / "benchmark_settings.json"
    new = settings_dict()
    if p.exists():
        old = load_json(p)
        if old != new:
            raise RuntimeError(
                f"Папка уже содержит ДРУГОЙ эксперимент: {RUN_DIR}\n"
                "Измени RUN_LABEL наверху (например, 'v2'). Не смешивай конфигурации!")
        if not RESUME:
            raise RuntimeError("Этот запуск уже существует. Измени RUN_LABEL.")
    else:
        save_json(p, new)
    return p


def seed_everything():
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def create_shared_tracking():
    """Один-единственный запуск YOLO + ByteTrack; его color-признаки НЕ сравниваются."""
    obs = SHARED_DIR / "observations.csv"
    tracks = SHARED_DIR / "local_tracklets.json"
    manifest = SHARED_DIR / "manifest.json"
    if RESUME and all(x.is_file() for x in (obs, tracks, manifest, SHARED_DIR / "performance.json")):
        print("\n[1/3] Использую сохранённые YOLO/ByteTrack detections:", SHARED_DIR)
        return
    if SHARED_DIR.exists() and any(SHARED_DIR.iterdir()):
        raise RuntimeError(
            "Обнаружена неполная shared_tracking; перемести/удали её или измени RUN_LABEL")
    from mtmct.tracking import run as tracking_run
    print("\n[1/3] YOLO + ByteTrack один раз для всех моделей ...", flush=True)
    start = time.perf_counter()
    tracking_run(SCENE_DIR, SHARED_DIR, YOLO_MODEL, "color", None, DEVICE,
                 MAX_FRAMES, 1, SAMPLE_EVERY, IMGSZ, CONF,
                 reid_model="osnet_x1_0", cameras=list(CAMERAS), tracker=TRACKER)
    print(f"Общие bbox и local ID готовы за {time.perf_counter()-start:.1f} с")
    save_json(SHARED_DIR / "tracking_time.json", {"wall_seconds": round(time.perf_counter()-start, 2)})


def observations_by_cam():
    data = {int(c): defaultdict(list) for c in CAMERAS}
    with (SHARED_DIR / "observations.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cam = int(row["camera_id"])
            if cam not in data:
                raise ValueError(f"Неожиданная камера {cam} в observations.csv")
            fid = int(row["frame_id"])
            data[cam][fid].append(row)
    return data


def extract_features(model: str, weights: Path, observations, shared_tracklets) -> tuple[list[dict], dict]:
    """Заново вычислить ВСЕ embedding из одних и тех же округлённых bbox.

    Пересчёт делается ДЛЯ ВСЕХ ТРЁХ, включая x1_0, поэтому изображения-кропы
    идентичны. Выбор кадров повторяет sampling в mtmct/tracking.py.
    """
    import cv2
    import numpy as np
    import torch
    from mtmct.reid import ReID, normalized
    from mtmct.features import MAX_SAMPLES, crop_quality

    print(f"\n[2/3] ReID модель {model}; веса: {weights.name}", flush=True)
    seed_everything()
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    extractor = ReID("osnet", device=DEVICE, weights=str(weights), model_name=model)
    stats = defaultdict(lambda: {"n": 0, "sum": None, "best": []})
    num_crops = 0
    encode_seconds = 0.0
    wall_start = time.perf_counter()

    for cam in sorted(CAMERAS):
        video = SCENE_DIR / "videos" / f"Camera_{cam:04d}.mp4"
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Не могу открыть {video}")
        print(f"  Камера {cam}", flush=True)
        try:
            for fid in range(MAX_FRAMES):
                ok, frame = cap.read()
                if not ok:
                    break
                if fid not in observations[cam]:
                    continue
                h, w = frame.shape[:2]
                crops, current = [], []
                for r in observations[cam][fid]:
                    lid = int(r["local_id"])
                    key = (cam, lid)
                    state = stats[key]
                    if state["n"] > 0 and fid % SAMPLE_EVERY != 0:
                        continue
                    # В tracking.py bbox в CSV записан с точностью до 0.01 пикселя.
                    x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
                    x1 = max(0., min(x1, w-1.)); y1 = max(0., min(y1, h-1.))
                    x2 = max(0., min(x2, w*1.)); y2 = max(0., min(y2, h*1.))
                    if x2-x1 < 2 or y2-y1 < 2:
                        continue
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if not crop.size:
                        continue
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    clipped = x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h
                    qual = float(crop_quality(rgb, float(r["confidence"]), clipped))
                    crops.append(rgb)
                    current.append((key, qual, fid))
                if not crops:
                    continue
                t0 = time.perf_counter()
                embeddings = extractor.encode(crops)
                encode_seconds += time.perf_counter() - t0
                if len(embeddings) != len(current):
                    raise RuntimeError(f"{model}: ожидалось {len(current)} embedding, получено {len(embeddings)}")
                for (key, qual, frame_id), embedding in zip(current, embeddings):
                    s = stats[key]
                    if s["sum"] is None:
                        s["sum"] = np.zeros_like(embedding, dtype=np.float64)
                    s["sum"] += embedding
                    s["n"] += 1
                    s["best"].append((qual, frame_id, embedding.tolist()))
                    s["best"] = sorted(s["best"], key=lambda x: -x[0])[:MAX_SAMPLES]
                    num_crops += 1
        finally:
            cap.release()

    output = []
    for old in shared_tracklets:
        t = dict(old)
        key = (int(t["camera_id"]), int(t["local_id"]))
        s = stats.get(key)
        if s and s["n"]:
            t["embedding"] = normalized(s["sum"]).tolist()
            t["feature_quality"] = [x[0] for x in s["best"]]
            t["feature_frames"] = [x[1] for x in s["best"]]
            t["feature_vectors"] = [x[2] for x in s["best"]]
            t["samples"] = int(s["n"])
        else:
            t.update(embedding=[], feature_quality=[], feature_frames=[], feature_vectors=[], samples=0)
        output.append(t)
    wall = time.perf_counter()-wall_start
    peak = float(torch.cuda.max_memory_allocated()) / (1024**3) if DEVICE.startswith("cuda") else None
    result = {"model": model, "num_crops": num_crops,
              "reid_seconds": round(encode_seconds, 3),
              "decode_plus_reid_seconds": round(wall, 3),
              "images_per_second_reid": round(num_crops/max(encode_seconds, 1e-9), 2),
              "peak_cuda_gib": round(peak, 3) if peak is not None else None,
              "feature_dim": len(next((t["embedding"] for t in output if t["embedding"]), []))}
    print(f"  {num_crops} кропов; ReID encode {encode_seconds:.1f} с; "
          f"{result['images_per_second_reid']} кропов/с", flush=True)
    del extractor
    gc.collect()
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()
    return output, result


def make_manifest(model: str, weights: Path, shared_manifest: dict, feature_dim: int):
    # Из общего manifest сохраняем точную provenance видеозаписей и детектора.
    m = json.loads(json.dumps(shared_manifest))
    m["models"]["reid"] = {"name": str(weights), "path": str(weights),
                          "size": weights.stat().st_size, "sha256": filesha(weights)}
    m["tracking_config"]["reid_method"] = "osnet"
    m["tracking_config"]["reid_model"] = model
    m["tracking_config"]["reid_weights"] = str(weights)
    m["tracking_config"]["sample_every"] = SAMPLE_EVERY
    m["feature_dimension"] = feature_dim
    return m


def prepare_mode(model: str, mode: str, features: Path, manifest: Path) -> Path:
    out = RUN_DIR / model / mode
    out.mkdir(parents=True, exist_ok=True)
    for src, name in ((SHARED_DIR / "observations.csv", "observations.csv"),
                      (features, "local_tracklets.json"), (manifest, "manifest.json"),
                      (SHARED_DIR / "performance.json", "performance.json")):
        shutil.copy2(src, out / name)
    save_json(out / "config.yaml", {**settings_dict(), "model": model, "mode": mode,
                                  "out": str(out)})
    return out


def evaluate_mode(run_dir: Path, mode: str):
    from mtmct.association import Config
    from mtmct.link import run as link_run
    from mtmct.evaluate import eval_2d, save_report
    cfg = Config(appearance_weight=0.95, spatial_weight=0.0,
                 temporal_weight=0.05, min_similarity=MIN_SIMILARITY,
                 max_cost=MAX_COST, use_geometry=False, use_time=True)
    start = time.perf_counter()
    _, mapping, edges, pred, _ = link_run(
        run_dir, cfg, expected_detector=YOLO_MODEL, matcher="baseline",
        memory=(mode == "full"), memory_similarity=MEMORY_SIMILARITY,
        correct_identities=False, repair_occlusions=(mode == "full"))
    link_seconds = time.perf_counter()-start
    report = eval_2d(pred, GT_FILE, list(CAMERAS), MAX_FRAMES, threshold=EVAL_IOU)
    save_report(report, run_dir / "metrics.json")
    identities = len(set(mapping.values()))
    per_camera = {str(cam): report.get("per_camera", {}).get(str(cam), {}).get("IDF1")
                  for cam in sorted(CAMERAS)}
    accepted_relinks = None
    if mode == "full" and (run_dir / "occlusion_repair.json").is_file():
        accepted_relinks = load_json(run_dir / "occlusion_repair.json").get("summary",{}).get("accepted_relinks")
    return {"global_idf1": report["global"]["IDF1"],
            "global_idp": report["global"]["IDP"],
            "global_idr": report["global"]["IDR"],
            "pairwise_f1": report["pairwise_association"]["F1"],
            "pairwise_precision": report["pairwise_association"]["precision"],
            "pairwise_recall": report["pairwise_association"]["recall"],
            "detection_f1": report["detection"]["F1"],
            "detection_precision": report["detection"]["precision"],
            "detection_recall": report["detection"]["recall"],
            "global_ids": identities, "local_tracklets": len(load_json(run_dir / "local_tracklets.json")),
            "accepted_relinks": accepted_relinks, "link_seconds": round(link_seconds,2),
            "per_camera_idf1": per_camera, "predictions": str(pred)}


def result_from_existing(run_dir: Path, mode: str) -> dict:
    report = load_json(run_dir / "metrics.json")
    mapping = load_json(run_dir / "global_ids.json")
    accepted_relinks = None
    if mode == "full" and (run_dir / "occlusion_repair.json").exists():
        accepted_relinks = load_json(run_dir / "occlusion_repair.json").get("summary",{}).get("accepted_relinks")
    return {"global_idf1": report["global"]["IDF1"],
            "global_idp": report["global"]["IDP"],
            "global_idr": report["global"]["IDR"],
            "pairwise_f1": report["pairwise_association"]["F1"],
            "pairwise_precision": report["pairwise_association"]["precision"],
            "pairwise_recall": report["pairwise_association"]["recall"],
            "detection_f1": report["detection"]["F1"],
            "detection_precision": report["detection"]["precision"],
            "detection_recall": report["detection"]["recall"],
            "global_ids": len({int(row["global_id"]) for row in mapping}),
            "local_tracklets": len(load_json(run_dir / "local_tracklets.json")),
            "accepted_relinks": accepted_relinks,
            "link_seconds": None,
            "per_camera_idf1": {str(cam): report.get("per_camera",{}).get(str(cam),{}).get("IDF1")
                                for cam in sorted(CAMERAS)},
            "predictions": str(run_dir / "global_tracks_2d.csv")}


def write_comparison(rows: list[dict], failed: list[dict]):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RUN_DIR / "comparison.json", {"settings": settings_dict(), "results": rows, "failed": failed})
    columns = ["model", "mode", "global_idf1", "pairwise_f1", "global_idp", "global_idr",
               "pairwise_precision", "pairwise_recall", "detection_recall", "detection_precision",
               "global_ids", "local_tracklets", "accepted_relinks", "reid_seconds",
               "images_per_second_reid", "peak_cuda_gib", "link_seconds"] + [
                   f"idf1_cam{cam}" for cam in sorted(CAMERAS)]
    with (RUN_DIR / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({**{key: r.get(key) for key in columns},
                            **{f"idf1_cam{cam}": r.get("per_camera_idf1",{}).get(str(cam))
                               for cam in sorted(CAMERAS)}})

    report = []
    report += ["ЧЕСТНОЕ СРАВНЕНИЕ OSNet", "="*75,
               f"Сцена: {SCENE}, камеры: {sorted(CAMERAS)}, кадров: {MAX_FRAMES}",
               f"YOLO={YOLO_MODEL} imgsz={IMGSZ} conf={CONF}; ByteTrack={TRACKER}",
               f"ReID sample_every={SAMPLE_EVERY}; appearance=0.95; temporal=0.05; geometry=OFF",
               f"min_similarity={MIN_SIMILARITY}; max_cost={MAX_COST}; memory_similarity={MEMORY_SIMILARITY}",
               "Все модели использовали один observations.csv и одни и те же кропы.",
               "Режим baseline: без memory/occlusion repair. Режим full: оба включены.", ""]
    if (SHARED_DIR / "performance.json").is_file():
        perf = load_json(SHARED_DIR / "performance.json")
        report.append("Общая детекция (один раз на три модели):")
        for cam_result in perf:
            report.append(f"  камера {cam_result['camera_id']}: "
                          f"{cam_result['processed_frames']} кадров, "
                          f"{cam_result['fps']} FPS")
        report.append("")
    report.append(f"{'MODEL':21} {'MODE':9} {'IDF1':>8} {'PAIR F1':>9} {'IDP':>8} {'IDR':>8} {'REID crop/s':>12}")
    report.append("-"*75)
    for r in sorted(rows, key=lambda x: (x["mode"]!="baseline", -x["global_idf1"])):
        report.append(f"{r['model']:21} {r['mode']:9} {r['global_idf1']:8.4f} {r['pairwise_f1']:9.4f} "
                      f"{r['global_idp']:8.4f} {r['global_idr']:8.4f} {r['images_per_second_reid']:12.1f}")
    report += [""]
    for mode in ("baseline", "full"):
        candidates = [r for r in rows if r["mode"]==mode]
        if candidates:
            best = max(candidates, key=lambda r: r["global_idf1"])
            report.append(f"Лучший {mode} по Global IDF1: {best['model']} ({best['global_idf1']:.6f})")
    if failed:
        report += ["", "НЕУДАВШИЕСЯ ЗАПУСКИ:"]
        for f in failed:
            report.append(f"{f['model']}: {f['error']}")
    report += ["", "ВАЖНО: Global IDF1 — пользовательская 2D-метрика, НЕ NVIDIA 3D HOTA.",
               "Для честной проверки на новых сценах не подбирай пороги на этих тестовых кадрах.",
               "Если модель не была запущена, она не участвует в рейтинге."]
    (RUN_DIR / "comparison.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    print("\n" + "\n".join(report))
    return {mode: max((r for r in rows if r["mode"]==mode),
                      key=lambda r: r["global_idf1"], default=None)
            for mode in ("baseline", "full")}


def main():
    print("Сравнение ТРЁХ ReID моделей на одинаковых локальных треках")
    print("Камеры:", CAMERAS, "| кадры:", MAX_FRAMES)
    prepare_root()
    seed_everything()
    weights = preflight()
    create_shared_tracking()
    observations = observations_by_cam()
    source = load_json(SHARED_DIR / "local_tracklets.json")
    shared_manifest = load_json(SHARED_DIR / "manifest.json")
    rows, failed = [], []
    expected_num_crops = None

    for model in MODELS:
        model_dir = RUN_DIR / model
        model_dir.mkdir(parents=True, exist_ok=True)
        features_path = model_dir / "local_tracklets.json"
        manifest_path = model_dir / "manifest.json"
        timing_path = model_dir / "reid_performance.json"
        try:
            if RESUME and all(p.exists() for p in (features_path, manifest_path, timing_path)):
                print(f"\n[2/3] Использую сохранённые embeddings: {model}")
                timing = load_json(timing_path)
            else:
                new_features, timing = extract_features(model, weights[model], observations, source)
                save_json(features_path, new_features)
                save_json(manifest_path, make_manifest(model, weights[model], shared_manifest, timing["feature_dim"]))
                save_json(timing_path, timing)
            if expected_num_crops is None:
                expected_num_crops = timing["num_crops"]
            elif timing["num_crops"] != expected_num_crops:
                raise RuntimeError(
                    f"Количество кропов для {model} отличается: "
                    f"{timing['num_crops']} вместо {expected_num_crops} — сравнение некорректно")
            for mode in ("baseline", "full"):
                run_dir = model_dir / mode
                cache_files = (run_dir / "metrics.json", run_dir / "global_ids.json",
                               run_dir / "global_tracks_2d.csv")
                if RESUME and all(f.is_file() for f in cache_files):
                    print(f"[3/3] Уже готово: {model}/{mode}")
                    result = result_from_existing(run_dir, mode)
                else:
                    prepare_mode(model, mode, features_path, manifest_path)
                    print(f"[3/3] Оценка {model}/{mode}", flush=True)
                    result = evaluate_mode(run_dir, mode)
                result.update({"model": model, "mode": mode,
                               "reid_seconds": timing["reid_seconds"],
                               "images_per_second_reid": timing["images_per_second_reid"],
                               "peak_cuda_gib": timing["peak_cuda_gib"],
                               "num_crops": timing["num_crops"]})
                rows.append(result)
                write_comparison(rows, failed)  # сохраняем промежуточные результаты
        except Exception as e:
            error_log = model_dir / "error.log"
            error_log.write_text(traceback.format_exc(), encoding="utf-8")
            failed.append({"model": model, "error": f"{type(e).__name__}: {e}",
                           "traceback": str(error_log)})
            print(f"\nОШИБКА в {model}: {e}\nПодробнее: {error_log}")
            write_comparison(rows, failed)
        finally:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    best = write_comparison(rows, failed)
    if RENDER_ALL:
        targets = [(r["model"], r["mode"], Path(r["predictions"])) for r in rows]
    else:
        winner = best.get("full")
        targets = [(winner["model"], "full", Path(winner["predictions"]))] if RENDER_WINNER and winner else []
    if targets:
        from mtmct.visualize import render
        for model, mode, predictions in targets:
            out = predictions.parent / "mosaic.mp4"
            if out.exists() and RESUME:
                continue
            print("Строю видео:", model, mode)
            render(SCENE_DIR, predictions, out, MAX_FRAMES, 1, RENDER_SCALE, list(CAMERAS))
    print("\nГотово! Таблица:", RUN_DIR / "comparison.csv")
    print("Текстовый отчёт:", RUN_DIR / "comparison.txt")
    print("Полные метрики в каждом каталоге <model>/<mode>/metrics.json")
    if failed:
        print("ВНИМАНИЕ: не все модели удалось протестировать; см. comparison.txt")
    else:
        print("ВСЕ ТРИ МОДЕЛИ проверены при одинаковых условиях.")


if __name__ == "__main__":
    main()
