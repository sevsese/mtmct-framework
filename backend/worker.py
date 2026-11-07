"""GPU worker using the same detector and association preset as the office runner."""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import ijson

from backend.adapter import office_camera_ids
from mtmct.cache import fingerprint, validate_cache, write_config
from run_hackaton_experiment import (
    CONF, EXCLUDE_ROIS, FRAME_STRIDE, GALLERY_WEIGHT, GLOBAL_MAX_GAP_SECONDS,
    IMGSZ, MIN_OBSERVATIONS, MIN_SIMILARITY, MODEL, PART_WEIGHT, REID_MODEL,
    TOPOLOGY, TRACKER, ZOOM_ROIS,
)

ROOT = Path(__file__).resolve().parents[1]


def frame_limit(options, camera):
    limit = options["max_frames"]
    camera_id = camera.get("camera_id")
    seconds = options.get("camera_max_seconds", {}).get(
        str(camera_id), options.get("max_seconds")
    )
    if seconds is not None:
        timed = math.ceil(seconds * camera["fps"])
        limit = min(limit, timed) if limit else timed
    return limit


def tracking_config(camera_id, office_id=None):
    return {
        "stride": FRAME_STRIDE,
        "sample_every": 5,
        "imgsz": IMGSZ,
        "conf": CONF,
        "tracker": str(TRACKER),
        "reid_model": REID_MODEL,
        "zoom_rois": {camera_id: ZOOM_ROIS[office_id]} if office_id in ZOOM_ROIS else {},
        "exclude_rois": {camera_id: EXCLUDE_ROIS[office_id]} if office_id in EXCLUDE_ROIS else {},
    }


def track_camera(spec, camera):
    import cv2
    import torch
    from mtmct.tracking import run as track

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    output = Path(spec["output"]) / f"camera_{camera['camera_id']}"
    office_id = office_camera_ids(spec["cameras"]).get(camera["camera_id"])
    settings = tracking_config(camera["camera_id"], office_id)
    track(
        Path(spec["scene"]), output, str(ROOT / MODEL), "osnet", None,
        spec["options"]["device"], frame_limit(spec["options"], camera),
        settings["stride"], settings["sample_every"], settings["imgsz"], settings["conf"],
        reid_model=settings["reid_model"], cameras=[camera["camera_id"]],
        tracker=settings["tracker"], camera_sources={camera["camera_id"]: Path(camera["path"])},
        zoom_rois=settings["zoom_rois"], exclude_rois=settings["exclude_rois"],
    )


def camera_cache_matches(part, camera, options, expected):
    files = ("manifest.json", "performance.json", "observations.csv", "local_tracklets.json")
    if not all((part / name).is_file() for name in files):
        return False
    try:
        manifest = validate_cache(part)
        camera_id = str(camera["camera_id"])
        saved = manifest["tracking_config"]
        expected = json.loads(json.dumps(expected))
        model = manifest["models"]["detector"]
        reid = manifest["models"]["reid"]
        limit = frame_limit(options, camera)
        source = str(Path(camera["path"]).resolve())
        performance = json.loads((part/"performance.json").read_text(encoding="utf-8"))
        return (
            all(saved.get(key) == value for key, value in expected.items())
            and saved.get("max_frames") == limit
            and str(Path(saved["camera_sources"][camera_id]).resolve()) == source
            and any(item["path"] == source for item in manifest["inputs"])
            and model["sha256"] == fingerprint(ROOT / MODEL)["sha256"]
            and reid["sha256"] == fingerprint(reid["path"])["sha256"]
            and len(performance) == 1
            and performance[0]["camera_id"] == camera["camera_id"]
            and performance[0]["processed_frames"] > 0
            and (not limit or performance[0]["processed_frames"] >=
                 math.ceil(min(limit, round(camera["duration"] * camera["fps"])) / expected["stride"]))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def merge_camera_outputs(output, parts):
    for name in ("local_tracklets.json", "performance.json"):
        with (output / name).open("w", encoding="utf-8") as destination:
            destination.write("[")
            first = True
            for part in parts:
                with (part / name).open("rb") as stream:
                    for item in ijson.items(stream, "item", use_float=True):
                        if not first:
                            destination.write(",")
                        json.dump(item, destination)
                        first = False
            destination.write("]")
    with (output / "observations.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = None
        for part in parts:
            with (part / "observations.csv").open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if writer is None:
                    writer = csv.DictWriter(destination, reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)


def main():
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = Path(spec["output"])
    cameras = spec["cameras"]
    office_ids = office_camera_ids(cameras)
    options = spec["options"]
    matcher = options.get("matcher", "v2")
    if not (ROOT / MODEL).is_file():
        raise FileNotFoundError(f"Модель отсутствует: {ROOT / MODEL}")
    write_config(output, options | {
        "model": MODEL, "reid_model": REID_MODEL, "matcher": matcher,
        "memory": False, "geometry": False, "frame_stride": FRAME_STRIDE,
        "imgsz": IMGSZ, "conf": CONF,
        "camera_offset": [f"{c['camera_id']}={c.get('offset', 0)}" for c in cameras],
    })
    if len(sys.argv) > 2:
        track_camera(spec, next(c for c in cameras if c["camera_id"] == int(sys.argv[2])))
        return

    print("STAGE tracking", flush=True)
    parts = []
    for camera in cameras:
        part = output / f"camera_{camera['camera_id']}"
        expected = tracking_config(camera["camera_id"], office_ids.get(camera["camera_id"]))
        reusable = options.get("resume") and camera_cache_matches(part, camera, options, expected)
        if reusable:
            print(f"Reusing completed camera {camera['camera_id']}", flush=True)
        else:
            (part/"manifest.json").unlink(missing_ok=True)
            print(f"STAGE tracking: camera {camera['camera_id']} ({len(parts) + 1}/{len(cameras)})", flush=True)
            subprocess.run(
                [sys.executable, "-u", "-m", "backend.worker", str(spec_path), str(camera["camera_id"])],
                check=True,
            )
        parts.append(part)
    merge_camera_outputs(output, parts)

    manifests = [json.loads((part / "manifest.json").read_text()) for part in parts]
    manifest = manifests[0]
    manifest["cameras"] = [c["camera_id"] for c in cameras]
    manifest["inputs"] = list({item["path"]: item for m in manifests for item in m["inputs"]}.values())
    manifest["tracking_config"].update(
        camera_sources={str(c["camera_id"]): c["path"] for c in cameras},
        per_camera_max_frames={str(c["camera_id"]): frame_limit(options, c) for c in cameras},
        camera_offsets_s={str(c["camera_id"]): c.get("offset", 0) for c in cameras},
        web_worker="office-preset-v1",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("STAGE features", flush=True)
    from mtmct.part_features import attach_body_parts
    tracklets_path = output / "local_tracklets.json"
    tracklets = json.loads(tracklets_path.read_text(encoding="utf-8"))
    sources = {c["camera_id"]: Path(c["path"]) for c in cameras}
    attach_body_parts(tracklets, output / "observations.csv", sources)
    tracklets_path.write_text(json.dumps(tracklets), encoding="utf-8")

    print("STAGE association", flush=True)
    from mtmct.association import Config
    from mtmct.identity_v2 import OFFICE_SETTINGS
    from mtmct.link import run as link
    from mtmct.time_sync import CameraTransitionModel

    fps = {c["camera_id"]: c["fps"] for c in cameras}
    offsets = {c["camera_id"]: c.get("offset", 0) for c in cameras}
    topology = None
    if office_ids:
        preset = CameraTransitionModel.from_json(TOPOLOGY)
        actual = {office: camera for camera, office in office_ids.items()}
        topology = CameraTransitionModel(
            {(actual[a], actual[b]): route for (a, b), route in preset.routes.items()},
            {frozenset(actual[c] for c in pair) for pair in preset.overlap_pairs},
            {frozenset(actual[c] for c in pair) for pair in preset.disjoint_pairs},
        )
    link(
        output,
        Config(
            appearance_weight=.92, spatial_weight=0, temporal_weight=.08,
            min_similarity=MIN_SIMILARITY, max_cost=.40,
            max_gap_seconds=GLOBAL_MAX_GAP_SECONDS, camera_fps=fps,
            max_interior_gap_seconds=GLOBAL_MAX_GAP_SECONDS,
            camera_offsets_s=offsets, transition_model=topology,
            gallery_weight=.25 if matcher == "v2" else GALLERY_WEIGHT, part_weight=PART_WEIGHT,
            use_geometry=False, use_time=True,
        ),
        matcher=matcher, allow_same_camera_fragments=True, memory=False,
        v2_settings=OFFICE_SETTINGS if office_ids else None,
        repair_occlusions=matcher == "v2",
        min_observations=MIN_OBSERVATIONS,
    )
    print("STAGE completed", flush=True)


if __name__ == "__main__":
    main()
