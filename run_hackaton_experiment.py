"""VS Code: измени настройки и нажми Run Python File."""
import json
import shutil
from pathlib import Path

from mtmct.association import Config
from mtmct.cache import write_config
from mtmct.floor_map import build_floor_tracks, render_floor_tracks
from mtmct.link import run as link
from mtmct.part_features import attach_body_parts
from mtmct.time_sync import CameraTransitionModel
from mtmct.track_report import write_global_track_report
from mtmct.tracking import run as track
from mtmct.visualize import render
from mtmct.association import save_mapping
from mtmct.export import export_predictions
from mtmct.office_pass import apply_pass, apply_frame_review, fit_projection, tracks, remove_review_duplicates, draw_pass
from mtmct.office_routes import map_office_routes
from mtmct.office_walls import render as render_walls
from mtmct.floor_map import _read_image


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
TRACKER = "bytetrack.yaml"
DEVICE = "cuda:0"
RENDER_SCALE = 0.25
MIN_SIMILARITY = 0.45      # combined OSNet gallery + upper/lower colour gate
GLOBAL_MAX_GAP_SECONDS = 10.0
CAMERA_FPS = {0: 25.0, 1: 25.0, 2: 25.0, 3: 20.0, 4: 25.0}
CAMERA_OFFSETS_SECONDS = {0: 0.0, 1: 1.0, 2: 1.0, 3: 4.0, 4: 3.0}
TOPOLOGY = ROOT / "configs" / "hackaton_office_topology.json"
FLOOR_PLAN = Path.home() / "Downloads" / "office_camera.png"
FLOOR_CALIBRATION = ROOT / "configs" / "hackaton_office_map_approx.json"
GALLERY_WEIGHT = 0.65      # best quality view + whole-track mean
PART_WEIGHT = 0.30         # view-tolerant upper/lower clothing evidence
MIN_OBSERVATIONS = 5       # reject 1-4 frame ByteTrack noise
ZOOM_ROIS = {4:(.36,0,.64,.34)}  # camera 5, normalized xyxy
EXCLUDE_ROIS = {
    2: [
        (.20, 0, .47, .40),          # far corridor
        (.659, .204, .935, .372),    # people shown on the wall display
    ]
}
LINK_ONLY = False          # True reuses detections/embeddings from SOURCE_RUN
SOURCE_RUN = "hackaton_5cam_60s_v7"
RUN_NAME = "hackaton_5cam_60s_corridor_v13"
REVIEW_REFERENCE = ROOT / "runs" / "hackaton_5cam_60s_v10"
# ============================================================================


def main():
    sources = {camera: VIDEO_DIR / name for camera, name in CAMERAS.items()}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Не найдены видео:\n" + "\n".join(missing))
    out = ROOT / "runs" / RUN_NAME
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Запуск уже существует: {out}. Измени RUN_NAME.")
    if LINK_ONLY:
        source = ROOT / "runs" / SOURCE_RUN
        for name in ("manifest.json", "local_tracklets.json", "observations.csv", "performance.json"):
            if not (source / name).is_file():
                raise FileNotFoundError(f"Нет кэша для LINK_ONLY: {source / name}")
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, out / name)
    else:
        write_config(out, {"dataset": "hackaton", "cameras": list(sources),
                           "duration_seconds": DURATION_SECONDS, "frame_stride": FRAME_STRIDE,
                           "tracker": str(TRACKER), "conf": CONF,
                           "camera_offset": [f"{camera}={seconds}" for camera, seconds in CAMERA_OFFSETS_SECONDS.items()],
                           "geometry": False, "metrics": False})
        track(VIDEO_DIR, out, MODEL, "osnet", None, DEVICE, 0, FRAME_STRIDE, 5,
              IMGSZ, CONF, reid_model=REID_MODEL, cameras=list(sources),
              tracker=TRACKER, camera_sources=sources, max_seconds=DURATION_SECONDS,
              zoom_rois=ZOOM_ROIS, exclude_rois=EXCLUDE_ROIS)
    tracklets_path = out / "local_tracklets.json"
    tracklets = json.loads(tracklets_path.read_text(encoding="utf-8"))
    attach_body_parts(tracklets, out / "observations.csv", sources)
    tracklets_path.write_text(json.dumps(tracklets), encoding="utf-8")
    cfg = Config(appearance_weight=.92, spatial_weight=0., temporal_weight=.08,
                 min_similarity=MIN_SIMILARITY, max_cost=.40,
                 max_gap_seconds=GLOBAL_MAX_GAP_SECONDS,
                 camera_fps=CAMERA_FPS, camera_offsets_s=CAMERA_OFFSETS_SECONDS,
                 transition_model=CameraTransitionModel.from_json(TOPOLOGY),
                 gallery_weight=GALLERY_WEIGHT, part_weight=PART_WEIGHT,
                 use_geometry=False, use_time=True)
    tracklets, mapping, _, predictions, _ = link(
        out, cfg, matcher="graph", allow_same_camera_fragments=True, memory=False,
        min_observations=MIN_OBSERVATIONS)
    finish(out,sources,tracklets,mapping,predictions)


def finish(out,sources,tracklets,mapping,predictions):
    base=REVIEW_REFERENCE/"wall_map_points"/"wall_anchors.json"
    reviewed_map=None
    if base.is_file() and (REVIEW_REFERENCE/"global_tracks_2d.csv").is_file():
        reference=tracks(REVIEW_REFERENCE/"global_tracks_2d.csv")
        reviewed_map,projection=fit_projection(reference,json.loads(base.read_text()),
                                               CAMERA_FPS,CAMERA_OFFSETS_SECONDS)
        if out.resolve()==REVIEW_REFERENCE.resolve():
            raise ValueError("Choose a new run for reviewed outputs")
        save_mapping(mapping,out/"automatic_ids.json")
        try:
            passage=apply_pass(reference,tracks(predictions),mapping)
            apply_frame_review(ROOT/"configs"/"office_pass_review.json",
                               tracks(out/"observations.csv"),mapping,passage)
            all_segments=json.loads((out/"local_tracklets.json").read_text(encoding="utf-8"))
            tracklets=[s for s in all_segments if (s["camera_id"],s["local_id"]) in mapping]
        except ValueError as exc:
            passage={"status":"not_matched","reason":str(exc)}
        (out/"projection_comparison.json").write_text(json.dumps(projection,indent=2))
        (out/"wall_anchors.json").write_text(json.dumps(reviewed_map,indent=2))
        save_mapping(mapping,out/"global_ids.json")
        predictions,_=export_predictions(out/"observations.csv",out/"global_ids.json",out)
        if "reviewed_global_id" in passage:
            passage["duplicate_rows"]=remove_review_duplicates(predictions,passage["reviewed_global_id"],reference)
        grey=passage.get("frame_review",{}).get("grey")
        if grey and len(grey["segments"]) > 1:
            grey["duplicates_removed"]=remove_review_duplicates(predictions,grey["id"],{},grey["segments"][:1])
        (out/"reviewed_pass.json").write_text(json.dumps(passage,indent=2))
    report = write_global_track_report(
        out / "global_tracks_summary.json", tracklets, mapping,
        CAMERA_FPS, CAMERA_OFFSETS_SECONDS)
    if FLOOR_PLAN.is_file() and reviewed_map:
        plan=_read_image(FLOOR_PLAN)
        render_walls(out,plan,reviewed_map,out)
        map_office_routes(out,plan,FLOOR_CALIBRATION)
        if "reviewed_global_id" in passage:
            draw_pass(out,plan,passage["reviewed_global_id"])
    elif FLOOR_PLAN.is_file():
        floor_rows = build_floor_tracks(
            predictions, FLOOR_CALIBRATION, out / "floor_tracks.csv", GLOBAL_MAX_GAP_SECONDS)
        render_floor_tracks(FLOOR_PLAN, floor_rows, out / "floor_tracks.png")
        print("Floor map: approximate door/floor-anchor calibration; refine the JSON for metric use.")
    render(VIDEO_DIR, predictions, out / "mosaic.mp4", stride=FRAME_STRIDE,
           scale=RENDER_SCALE, cameras=list(sources), camera_sources=sources,
           duration_seconds=DURATION_SECONDS,camera_offsets_s=CAMERA_OFFSETS_SECONDS)
    print(f"DONE: {out}")
    print(f"Global tracks: {len(report)} (manual expectation: 5); no GT metrics were calculated.")
    for row in report:
        print(f"  GLOBAL {row['global_id']}: cameras {row['camera_sequence']}, "
              f"{row['start_s']:.1f}-{row['end_s']:.1f}s")


if __name__ == "__main__":
    main()
