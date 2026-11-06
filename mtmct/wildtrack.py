"""WILDTRACK adapter for annotated 2 FPS moments (no interpolation)."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .evaluate import _identity_metrics, _match_boxes, _pairwise_metrics

PROTOCOL_VERSION = "wildtrack-annotated-2fps-idf1-v1"
ANNOTATED_FPS = 2.0


@dataclass(frozen=True)
class WildtrackObservation:
    raw_frame: int
    annotated_index: int
    timestamp_s: float
    camera_id: int
    gt_object_id: int
    bbox_xyxy: tuple[float, float, float, float]


class WildtrackAdapter:
    def __init__(self, root: Path):
        self.root = Path(root)
        official = self.root / "annotations_positions"
        # Early internal experiments used the singular spelling.  Keep it as a
        # fallback, but prefer the directory name in the official archive.
        self.annotation_dir = official if official.is_dir() else self.root / "annotation_positions"

    def annotation_files(self) -> list[Path]:
        if not self.annotation_dir.is_dir():
            raise FileNotFoundError(
                f"WILDTRACK annotations not found: {self.annotation_dir}"
            )
        files = sorted(self.annotation_dir.glob("*.json"), key=lambda path: int(path.stem))
        if not files:
            raise FileNotFoundError(f"No JSON annotations in {self.annotation_dir}")
        return files

    def observations(self):
        for annotated_index, path in enumerate(self.annotation_files()):
            raw_frame = int(path.stem)
            people = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(people, list):
                raise ValueError(f"Expected a list in {path}")
            for person in people:
                person_id = int(person["personID"])
                for view in person.get("views", []):
                    bbox = tuple(float(view[name]) for name in ("xmin", "ymin", "xmax", "ymax"))
                    if min(bbox) < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue
                    yield WildtrackObservation(
                        raw_frame=raw_frame,
                        annotated_index=annotated_index,
                        timestamp_s=annotated_index / ANNOTATED_FPS,
                        camera_id=int(view["viewNum"]),
                        gt_object_id=person_id,
                        bbox_xyxy=bbox,
                    )


def frame_sources(root: Path, cameras: list[int]) -> dict[int, Path]:
    """Return WILDTRACK's C1..C7 image folders for zero-based camera IDs."""
    root = Path(root)
    sources = {camera: root / "Image_subsets" / f"C{camera + 1}" for camera in cameras}
    missing = [str(path) for path in sources.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing WILDTRACK image folders: " + ", ".join(missing))
    return sources


def select_frame_ids(root: Path, cameras: list[int], max_frames: int = 0) -> list[int]:
    """Choose annotated moments present in every requested camera.

    The image filenames (for example ``00000005.png``) are retained as frame
    IDs.  They must not be naively treated as 2 FPS contiguous indices.
    """
    adapter = WildtrackAdapter(root)
    annotated = {int(path.stem) for path in adapter.annotation_files()}
    available = annotated
    for directory in frame_sources(root, cameras).values():
        available &= {int(path.stem) for path in directory.glob("*.png") if path.stem.isdigit()}
    selected = sorted(available)
    if not selected:
        raise FileNotFoundError("No annotated WILDTRACK frames are available in every selected camera")
    return selected[:max_frames] if max_frames > 0 else selected


def inspect_dataset(root: Path) -> dict:
    adapter = WildtrackAdapter(root)
    files = adapter.annotation_files()
    observations = list(adapter.observations())
    per_camera = Counter(item.camera_id for item in observations)
    return {
        "dataset": "WILDTRACK",
        "annotation_schema": "annotations_positions/*.json",
        "annotated_fps": ANNOTATED_FPS,
        "interpolation": False,
        "annotation_count": len(files),
        "raw_frames": [int(path.stem) for path in files],
        "cameras": sorted(per_camera),
        "people": len({item.gt_object_id for item in observations}),
        "observations": len(observations),
        "observations_per_camera": {str(key): per_camera[key] for key in sorted(per_camera)},
        "calibration_required": False,
        "map_required": False,
    }


def evaluate(pred_path: Path, root: Path, cameras: list[int], threshold: float = 0.5,
             frame_ids: set[int] | None = None) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must be between 0 and 1")
    cameras = sorted(set(cameras))
    selected = set(cameras)
    ground_truth = defaultdict(list)
    gt_by_camera = Counter()
    adapter = WildtrackAdapter(root)
    for item in adapter.observations():
        if item.camera_id not in selected or (frame_ids is not None and item.raw_frame not in frame_ids):
            continue
        key = (item.raw_frame, item.camera_id)
        ground_truth[key].append((item.gt_object_id, item.bbox_xyxy))
        gt_by_camera[item.camera_id] += 1

    predictions = defaultdict(list)
    pred_by_camera = Counter()
    with Path(pred_path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            camera = int(row["camera_id"])
            if camera not in selected or (frame_ids is not None and int(row["frame_id"]) not in frame_ids):
                continue
            key = (int(row["frame_id"]), camera)
            predictions[key].append((int(row["global_id"]), tuple(
                float(row[name]) for name in ("x1", "y1", "x2", "y2")
            )))
            pred_by_camera[camera] += 1

    matches = Counter()
    matches_by_camera = {camera: Counter() for camera in cameras}
    for key in set(ground_truth) | set(predictions):
        for gt_id, pred_id in _match_boxes(ground_truth[key], predictions[key], threshold):
            matches[(gt_id, pred_id)] += 1
            matches_by_camera[key[1]][(gt_id, pred_id)] += 1

    gt_total = sum(gt_by_camera.values())
    pred_total = sum(pred_by_camera.values())
    matched = sum(matches.values())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": "bbox IoU at annotated WILDTRACK moments, then global one-to-one identity assignment",
        "official_wildtrack_metric": False,
        "annotated_fps": ANNOTATED_FPS,
        "interpolation": False,
        "geometry_used": False,
        "iou_threshold": threshold,
        "cameras": cameras,
        "frame_ids": sorted(frame_ids) if frame_ids is not None else None,
        "global": _identity_metrics(matches, gt_total, pred_total),
        "detection": {
            "TP": matched,
            "FP": pred_total - matched,
            "FN": gt_total - matched,
        },
        "per_camera": {
            str(camera): _identity_metrics(
                matches_by_camera[camera], gt_by_camera[camera], pred_by_camera[camera]
            )
            for camera in cameras
        },
        "pairwise_association": _pairwise_metrics(matches_by_camera),
    }


def _write(result: dict, out: Path | None):
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect or evaluate WILDTRACK annotations")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--root", type=Path, required=True)
    inspect_parser.add_argument("--out", type=Path)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--root", type=Path, required=True)
    evaluate_parser.add_argument("--pred", type=Path, required=True)
    evaluate_parser.add_argument("--cameras", type=int, nargs="+", required=True)
    evaluate_parser.add_argument("--iou", type=float, default=0.5)
    evaluate_parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = (inspect_dataset(args.root) if args.command == "inspect" else
              evaluate(args.pred, args.root, args.cameras, args.iou))
    _write(result, args.out)


if __name__ == "__main__":
    main()
