"""Approximate image-to-floor projection and route rendering.

The calibration file stores four or more normalized image/map anchor pairs.
For the office demo these are deliberately labelled ``approximate``: doorways
and visible floor boundaries were matched by eye. Replacing the anchors with
surveyed points upgrades the same pipeline without rerunning detection/ReID.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def load_calibration(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status") not in {"approximate", "verified"}:
        raise ValueError("calibration status must be approximate or verified")
    for camera, row in data.get("cameras", {}).items():
        image = np.asarray(row.get("image_points_norm", []), dtype=float)
        floor = np.asarray(row.get("map_points_norm", []), dtype=float)
        if image.shape != floor.shape or image.ndim != 2 or image.shape[0] < 4 or image.shape[1] != 2:
            raise ValueError(f"camera {camera}: need matching Nx2 image/map anchors, N >= 4")
    return data


def homographies(calibration: dict) -> dict[int, np.ndarray]:
    maps = {}
    for camera, row in calibration["cameras"].items():
        source = np.asarray(row["image_points_norm"], dtype=np.float32)
        target = np.asarray(row["map_points_norm"], dtype=np.float32)
        matrix, _ = cv2.findHomography(source, target, method=0)
        if matrix is None or not np.all(np.isfinite(matrix)):
            raise ValueError(f"camera {camera}: degenerate anchors")
        maps[int(camera)] = matrix
    return maps


def project_foot(matrix: np.ndarray, bbox: tuple[float, float, float, float],
                 image_size: tuple[float, float]) -> tuple[float, float] | None:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    point = np.asarray([[[(bbox[0] + bbox[2]) / (2 * width), bbox[3] / height]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, matrix)[0, 0]
    if not np.all(np.isfinite(mapped)) or np.any(mapped < 0) or np.any(mapped > 1):
        return None
    return float(mapped[0]), float(mapped[1])


def build_floor_tracks(predictions: str | Path, calibration_path: str | Path,
                       output_csv: str | Path, max_gap_seconds: float = 10.0) -> list[dict]:
    calibration = load_calibration(calibration_path)
    matrices = homographies(calibration)
    fps = {int(k): float(v) for k, v in calibration.get("camera_fps", {}).items()}
    offsets = {int(k): float(v) for k, v in calibration.get("camera_offsets_s", {}).items()}
    sizes = {int(k): tuple(v["image_size"]) for k, v in calibration["cameras"].items()}
    rows = []
    with Path(predictions).open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            camera = int(source["camera_id"])
            if camera not in matrices or camera not in fps:
                continue
            point = project_foot(
                matrices[camera], tuple(float(source[k]) for k in ("x1", "y1", "x2", "y2")), sizes[camera])
            if point is None:
                continue
            rows.append({
                "global_id": int(source["global_id"]), "camera_id": camera,
                "frame_id": int(source["frame_id"]),
                "timestamp_s": int(source["frame_id"]) / fps[camera] + offsets.get(camera, 0.0),
                "map_x_norm": point[0], "map_y_norm": point[1],
                "calibration_status": calibration["status"],
            })
    rows.sort(key=lambda row: (row["global_id"], row["timestamp_s"], row["camera_id"]))
    previous = {}
    for row in rows:
        old = previous.get(row["global_id"])
        distance = (float("inf") if old is None else float(np.hypot(
            row["map_x_norm"] - old["map_x_norm"], row["map_y_norm"] - old["map_y_norm"])))
        row["starts_segment"] = int(old is None
            or row["timestamp_s"] - old["timestamp_s"] > max_gap_seconds
            or distance > .18)
        previous[row["global_id"]] = row
    fields = list(rows[0]) if rows else [
        "global_id", "camera_id", "frame_id", "timestamp_s", "map_x_norm", "map_y_norm",
        "calibration_status", "starts_segment"]
    with Path(output_csv).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return rows


def _read_image(path: str | Path):
    data = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def visual_match_stats(left: np.ndarray, right: np.ndarray) -> dict:
    """Return conservative SIFT/RANSAC evidence; diagnostics, not calibration."""
    def prepare(image):
        scale = min(1.0, 1200.0 / max(image.shape[:2]))
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    prepared_a, prepared_b = prepare(left), prepare(right)
    sift = cv2.SIFT_create(nfeatures=2500)
    keys_a, desc_a = sift.detectAndCompute(prepared_a, None)
    keys_b, desc_b = sift.detectAndCompute(prepared_b, None)
    result = {"keypoints_left": len(keys_a), "keypoints_right": len(keys_b),
              "ratio_matches": 0, "ransac_inliers": 0, "inlier_ratio": 0.0,
              "left_coverage": 0.0, "right_coverage": 0.0, "credible": False}
    if desc_a is None or desc_b is None:
        return result
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    good = [first for first, second in pairs if first.distance < .7 * second.distance]
    result["ratio_matches"] = len(good)
    if len(good) < 4:
        return result
    source = np.float32([keys_a[match.queryIdx].pt for match in good])
    target = np.float32([keys_b[match.trainIdx].pt for match in good])
    _, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
    selected = mask.ravel().astype(bool) if mask is not None else np.zeros(len(good), dtype=bool)
    inliers = int(selected.sum())
    def coverage(points, shape):
        if len(points) < 2:
            return 0.0
        span = np.ptp(points, axis=0)
        return float((span[0] * span[1]) / (shape[0] * shape[1]))
    left_coverage = coverage(source[selected], prepared_a.shape)
    right_coverage = coverage(target[selected], prepared_b.shape)
    result.update(ransac_inliers=inliers, inlier_ratio=round(inliers / len(good), 4),
                  left_coverage=round(left_coverage, 4), right_coverage=round(right_coverage, 4),
                  credible=bool(inliers >= 20 and min(left_coverage, right_coverage) >= .15))
    return result


def check_background_matches(plan_image: str | Path, camera_sources: dict[int, str | Path],
                             output_json: str | Path, timestamp_s: float = 10.0) -> dict:
    """Test whether automatic photo/plan or inter-camera feature matching is credible."""
    frames = {}
    for camera, source in camera_sources.items():
        capture = cv2.VideoCapture(str(source))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not read camera {camera} at {timestamp_s}s")
            frames[int(camera)] = frame
        finally:
            capture.release()
    plan = _read_image(plan_image)
    report = {
        "timestamp_s": timestamp_s,
        "interpretation": "Use homography only with many spatially distributed RANSAC inliers.",
        "plan_to_camera": {str(camera): visual_match_stats(plan, frame)
                           for camera, frame in frames.items()},
        "camera_pairs": {},
    }
    for left in sorted(frames):
        for right in sorted(frames):
            if left < right:
                report["camera_pairs"][f"{left}-{right}"] = visual_match_stats(frames[left], frames[right])
    Path(output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def render_floor_tracks(plan_image: str | Path, rows: list[dict], output: str | Path) -> None:
    canvas = _read_image(plan_image)
    height, width = canvas.shape[:2]
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["global_id"])].append(row)
    for global_id, points in grouped.items():
        color = (int((53 * global_id) % 180 + 50), int((97 * global_id) % 180 + 50),
                 int((149 * global_id) % 180 + 50))
        previous = None
        history = []
        for row in points:
            history.append((float(row["map_x_norm"]), float(row["map_y_norm"])))
            if row["starts_segment"]:
                history = history[-1:]
            smooth = np.median(np.asarray(history[-7:]), axis=0)
            point = (round(float(smooth[0]) * width), round(float(smooth[1]) * height))
            if previous is not None and not row["starts_segment"]:
                cv2.line(canvas, previous, point, color, 3, cv2.LINE_AA)
            previous = point
        if previous:
            cv2.circle(canvas, previous, 7, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"G{global_id}", previous, cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    cv2.rectangle(canvas, (0, 0), (width, 42), (255, 255, 255), -1)
    cv2.putText(canvas, "APPROXIMATE: door/floor anchors; not surveyed coordinates", (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 180), 2, cv2.LINE_AA)
    suffix = Path(output).suffix or ".png"
    ok, encoded = cv2.imencode(suffix, canvas)
    if not ok:
        raise RuntimeError(f"could not encode {output}")
    encoded.tofile(Path(output))
