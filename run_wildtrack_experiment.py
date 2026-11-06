"""Открой в VS Code, измени настройки ниже и нажми Run Python File."""
from pathlib import Path

from run_wildtrack import main as run_wildtrack


ROOT = Path(__file__).resolve().parent

# =========================== НАСТРОЙКИ =====================================
CAMERAS = (0, 1, 2, 3)     # номера камер WILDTRACK: 0..6
MAX_FRAMES = 40            # 0 = все 400 синхронных размеченных моментов
MODEL = "yolo26l.pt"
REID_MODEL = "osnet_ibn_x1_0"
IMGSZ = 1536
CONF = 0.10
DEVICE = "cuda:0"
RENDER_VIDEO = True        # mosaic.mp4 с bbox и GLOBAL ID
CHECK_ONLY = False         # True: только проверить распакованные данные
RUN_NAME = "wildtrack_4cam_40f"
# ============================================================================


def main():
    args = ["--root", str(ROOT / "data" / "WILDTRACK"),
            "--cameras", *map(str, CAMERAS),
            "--max-frames", str(MAX_FRAMES),
            "--model", MODEL, "--reid-model", REID_MODEL,
            "--imgsz", str(IMGSZ), "--conf", str(CONF), "--device", DEVICE,
            "--out", str(ROOT / "runs" / RUN_NAME)]
    if RENDER_VIDEO:
        args.append("--render")
    if CHECK_ONLY:
        args.append("--check-only")
    print("WILDTRACK settings:", " ".join(args))
    return run_wildtrack(args)


if __name__ == "__main__":
    main()
