"""Inspect NVIDIA GT camera coverage before running MTMCT camera_selection.

Copy to the MTMCT project root (next to run_nightly_optimize.py), then run:
python diagnose_warehouse_cameras.py --scene-dir "data/MTMC_Tracking_2026/train/Warehouse_002" --max-frames 900
"""
from __future__ import annotations

from pathlib import Path as _LauncherPath
import sys as _launcher_sys
_launcher_sys.path.insert(0, str(_LauncherPath(__file__).resolve().parents[1]))
_launcher_sys.path.insert(0, str(_LauncherPath(__file__).resolve().parent))
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from mtmct.geometry import camera_number
from mtmct.nvidia import NvidiaDatasetAdapter


def inspect(scene_dir: Path, max_frames: int) -> None:
    gt_file = scene_dir / "ground_truth.json"
    if not gt_file.is_file():
        raise FileNotFoundError(f"Missing GT: {gt_file}")

    counts = Counter()
    frames = defaultdict(set)
    people = defaultdict(set)
    frame_people = defaultdict(lambda: defaultdict(set))
    bounds = {}

    adapter = NvidiaDatasetAdapter(gt_file)
    for ob in adapter.observations(max_frames=max_frames, strict=False):
        cam = ob.camera_id
        counts[cam] += 1
        frames[cam].add(ob.frame_index)
        people[cam].add(ob.gt_object_id)
        frame_people[ob.frame_index][cam].add(ob.gt_object_id)
        if cam not in bounds:
            bounds[cam] = [ob.frame_index, ob.frame_index]
        else:
            bounds[cam][0] = min(bounds[cam][0], ob.frame_index)
            bounds[cam][1] = max(bounds[cam][1], ob.frame_index)

    videos_dir = scene_dir / "videos"
    video_by_cam = defaultdict(list)
    for path in videos_dir.rglob("*.mp4") if videos_dir.exists() else ():
        try:
            cam = camera_number(path.stem)
        except (ValueError, TypeError, OverflowError):
            continue
        video_by_cam[cam].append(str(path.relative_to(scene_dir)))

    stats = adapter.observation_stats
    print(f"GT: {gt_file}")
    print(f"Window: frame_index < {max_frames} (not {max_frames} distinct frames)")
    print(f"BBox accepted: {stats['valid_bboxes']:,}; rejected: {stats['invalid_bboxes']:,}")
    print(f"Cameras with valid GT: {len(counts)}; cameras with local mp4: {len(video_by_cam)}")
    print("\nCAMERA  BBOXES    FRAMES  PEOPLE   FIRST  LAST    LOCAL_VIDEO")
    for cam in sorted(set(counts) | set(video_by_cam)):
        lo, hi = bounds.get(cam, ["-", "-"])
        print(f"{cam:>6}  {counts[cam]:>8}  {len(frames[cam]):>6}  "
              f"{len(people[cam]):>6}  {str(lo):>6}  {str(hi):>4}    "
              f"{'; '.join(video_by_cam[cam]) or '(none yet)'}")

    top_pairs = []
    for a, b in combinations(sorted(counts), 2):
        common_ids = len(people[a] & people[b])
        common_frames = frames[a] & frames[b]
        co_visible = sum(len(frame_people[f][a] & frame_people[f][b]) for f in common_frames)
        if common_ids or co_visible:
            top_pairs.append((co_visible, common_ids, len(common_frames), a, b))
    top_pairs.sort(reverse=True)

    print("\nTOP CAMERA PAIRS: simultaneous same-person observations, shared IDs, shared frames")
    for positive, common_ids, common_frames, a, b in top_pairs[:15]:
        print(f"{a:>3} - {b:<3}: co-visible={positive:<7} shared_ids={common_ids:<5} same_frames={common_frames}")
    if not top_pairs:
        print("No shared-person camera pairs found in this window.")
    print("\nTo identify the actual eligibility rule, inspect camera_selection() in run_nightly_optimize.py.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=900)
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    inspect(args.scene_dir, args.max_frames)


if __name__ == "__main__":
    main()
