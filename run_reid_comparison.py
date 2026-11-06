"""Open this script in VS Code and press Run Python File (no terminal needed).

Requires the two MTMCT files from this patch. Reprocesses videos from scratch;
ReID embeddings from different model types must NEVER be mixed.
"""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# Choose ONE of: osnet_x1_0, osnet_ibn_x1_0, osnet_ain_x1_0
REID_MODEL = "osnet_ain_x1_0"

CAMERAS = (3, 15)
MAX_FRAMES = 900
IMGSZ = 1920       # Match prior new_experiment settings
CONF = 0.10
SAMPLE_EVERY = 5
MEMORY_SIMILARITY = 0.95
CALCULATE_METRICS = True

OUT = ROOT / "runs" / f"reid_compare_{REID_MODEL}_{MAX_FRAMES}"

if __name__ == "__main__":
    if REID_MODEL not in {"osnet_x1_0", "osnet_ibn_x1_0", "osnet_ain_x1_0"}:
        raise ValueError("Unknown ReID variant")
    if (OUT / "manifest.json").exists():
        raise FileExistsError(
            f"Results already exist: {OUT}. Change folder name to avoid overwriting."
        )
    args = [
        "mtmct",
        "--cameras", *map(str, CAMERAS),
        "--max-frames", str(MAX_FRAMES),
        "--imgsz", str(IMGSZ),
        "--conf", str(CONF),
        "--sample-every", str(SAMPLE_EVERY),
        "--model", "yolo26l.pt",
        "--reid-model", REID_MODEL,
        "--device", "cuda:0",
        "--out", str(OUT),
        "--memory",
        "--memory-similarity", str(MEMORY_SIMILARITY),
        "--repair-occlusions",
        "--no-download",
    ]
    if CALCULATE_METRICS:
        args.append("--evaluate")
    print("Running:", args)
    sys.argv = args
    runpy.run_module("mtmct", run_name="__main__", alter_sys=True)
