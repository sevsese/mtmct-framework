"""VS Code: измени настройки и нажми Run Python File."""
from pathlib import Path

from mtmct.association import Config
from mtmct.cache import write_config
from mtmct.link import run as link
from mtmct.tracking import run as track
from mtmct.visualize import render


ROOT = Path(__file__).resolve().parent

# =========================== НАСТРОЙКИ =====================================
VIDEO_DIR = ROOT / "data" / "Hackaton Data"
CAMERAS = {
    0: "camera_new_1.mkv", 1: "camera_new_2.mkv", 2: "camera_new_3.mkv",
    3: "camera_new_4.mkv", 4: "camera_new_5.mkv",
}
DURATION_SECONDS = 60       # первые N секунд каждой камеры
FRAME_STRIDE = 2            # 2 = 12.5 FPS у большинства камер, быстрее первого прогона
MODEL = "yolo26l.pt"
REID_MODEL = "osnet_ibn_x1_0"
IMGSZ = 1536
CONF = 0.10
DEVICE = "cuda:0"
RENDER_SCALE = 0.25
RUN_NAME = "hackaton_5cam_60s_v1"
# ============================================================================


def main():
    sources = {camera: VIDEO_DIR / name for camera, name in CAMERAS.items()}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Не найдены видео:\n" + "\n".join(missing))
    out = ROOT / "runs" / RUN_NAME
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Запуск уже существует: {out}. Измени RUN_NAME.")
    write_config(out, {"dataset": "hackaton", "cameras": list(sources),
                       "duration_seconds": DURATION_SECONDS, "frame_stride": FRAME_STRIDE,
                       "geometry": False, "metrics": False})
    track(VIDEO_DIR, out, MODEL, "osnet", None, DEVICE, 0, FRAME_STRIDE, 5,
          IMGSZ, CONF, reid_model=REID_MODEL, cameras=list(sources),
          camera_sources=sources, max_seconds=DURATION_SECONDS)
    # No calibration or GT: associations are appearance-only and exploratory.
    cfg = Config(appearance_weight=.95, spatial_weight=0., temporal_weight=0.,
                 min_similarity=.85, max_cost=.125, use_geometry=False, use_time=False)
    _, mapping, _, predictions, _ = link(out, cfg, memory=True, memory_similarity=.95)
    render(VIDEO_DIR, predictions, out / "mosaic.mp4", stride=FRAME_STRIDE,
           scale=RENDER_SCALE, cameras=list(sources), camera_sources=sources,
           duration_seconds=DURATION_SECONDS)
    print(f"DONE: {out}")
    print(f"Global IDs: {len(set(mapping.values()))}; no GT metrics were calculated.")


if __name__ == "__main__":
    main()
