"""Project a detection's foot point onto the z=0 world floor.

AICity 2026 ground truth uses (x, y, z), with z vertical.
Hence the floor coordinates are (x, y).
The 3x4 cameraMatrix P maps homogeneous world (X, Y, Z, 1) to pixels.
For Z=0, P[:, [0, 1, 3]] maps (X,Y,1) to image (u,v,1).
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np


def camera_number(name: str | int) -> int:
    match = re.search(r"\d+", str(name))
    if match is None:
        raise ValueError(f"Could not find camera number: {name}")
    return int(match.group())


def pixel_to_floor_matrix(camera: dict) -> np.ndarray | None:
    """Return a 3x3 pixel->(world X,Y) map assuming floor world Z==0.

    Do *not* blindly use a supplied homography: its direction, units, and
    output axes aren't guaranteed by the textual dataset description.
    """
    raw = camera.get("cameraMatrix")
    if raw is None:
        return None
    p = np.asarray(raw, dtype=float)
    if p.shape != (3, 4) or not np.all(np.isfinite(p)):
        raise ValueError("cameraMatrix must be a finite 3x4 array")
    floor_to_pixel = p[:, [0, 1, 3]]
    if abs(np.linalg.det(floor_to_pixel)) < 1e-10:
        return None
    return np.linalg.inv(floor_to_pixel)


def project_pixel(Hinv: np.ndarray | None, u: float, v: float) -> tuple[float,float] | None:
    if Hinv is None:
        return None
    w = Hinv @ np.asarray([u, v, 1.0], dtype=float)
    if abs(w[2]) < 1e-10:
        return None
    out = w[:2] / w[2]
    if not np.all(np.isfinite(out)):
        return None
    return float(out[0]), float(out[1])


def calibration_maps(path: str | Path) -> dict[int, np.ndarray | None]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    maps = {}
    for sensor in data.get("sensors", []):
        if str(sensor.get("type", "camera")).lower() != "camera":
            continue
        cid = camera_number(sensor["id"])
        maps[cid] = pixel_to_floor_matrix(sensor)
    return maps
