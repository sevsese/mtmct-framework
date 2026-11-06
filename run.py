"""VS Code: open this file in the ROOT of your MTMCT project and press Run Python File.

This reuses cached local tracklets/embeddings; YOLO and OSNet are NOT rerun.
Requires the existing mtmct package (with --memory and --repair-occlusions).
"""
from pathlib import Path
import os
import runpy
import shutil
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# EDIT JUST THESE IF YOUR EXPERIMENT NAMES DIFFER:
SOURCE = ROOT / "runs" / "new_experiment"
TARGET = ROOT / "runs" / "new_experiment_memory095"
CAMERAS = (3, 15,1,0)
MAX_FRAMES = 900
MEMORY_SIMILARITY = 0.95  # conservative first test; 0.90 made an extra unverified merge

# You need the original videos and GT in data/ to rebuild mosaic.mp4 and metrics.json.
RENDER_VIDEO = True
CALCULATE_METRICS = True


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(
            f"Original experiment not found: {SOURCE}. "
            "Make sure its name matches the folder in runs/."
        )
    for file in ("manifest.json", "local_tracklets.json", "observations.csv"):
        if not (SOURCE / file).is_file():
            raise FileNotFoundError(f"Original experiment is missing {file}: {SOURCE}")
    if SOURCE.resolve() == TARGET.resolve():
        raise ValueError("SOURCE and TARGET must be different to protect your original results")
    if not TARGET.exists():
        shutil.copytree(SOURCE, TARGET)
        print(f"Created a separate experiment: {TARGET}")
    else:
        print(f"Reusing target experiment: {TARGET}")

    args = [
        "mtmct",
        "--out", str(TARGET),
        "--cameras", *map(str, CAMERAS),
        "--max-frames", str(MAX_FRAMES),
        "--link-only",            # No YOLO / OSNet
        "--memory",               # Global identity memory
        "--memory-similarity", str(MEMORY_SIMILARITY),
        "--repair-occlusions",    # Keep the previous short-occlusion fix
        "--no-download",
    ]
    if RENDER_VIDEO:
        args.append("--render-link-only")
    if CALCULATE_METRICS:
        args.append("--evaluate")
    print("Run parameters:", args)
    sys.argv = args
    runpy.run_module("mtmct", run_name="__main__", alter_sys=True)

    import json
    mapping = json.loads((TARGET / "global_ids.json").read_text(encoding="utf-8"))
    key_to_global = {(int(t["camera_id"]), int(t["local_id"])): int(t["global_id"]) for t in mapping}
    print("\nCHECK: expected same identity for the black shirt / light pants person:")
    for key in ((15, 1), (15, 112), (3, 1), (3, 133)):
        print(f"  camera {key[0]}, local {key[1]} -> Global {key_to_global.get(key)}")
    print("Memory decisions:", TARGET / "global_memory.json")
    print("Global IDs:", TARGET / "global_ids.json")
    if CALCULATE_METRICS:
        print("Quality metrics:", TARGET / "metrics.json")


if __name__ == "__main__":
    main()
