"""Run NVIDIA/CITIC/SCOUT with ▶ Run Python File in VS Code. Edit settings here."""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# ==================== НАСТРОЙКИ ====================
DATASET = 'nvidia'  # 'nvidia', 'citic', or 'scout'
SPLIT = 'train'  # NVIDIA only: train / val / test
SCENE = 'Warehouse_000'  # CITIC: capture1; SCOUT: sequence_1
CAMERAS = (0, 1, 3, 15)  # None = auto-select all available cameras
MAX_FRAMES = 900

MODEL = 'yolo26l.pt'
REID_MODEL = 'osnet_ibn_x1_0'
DEVICE = 'cuda:0'
IMGSZ = 1536
CONF = 0.10
SAMPLE_EVERY = 5
TRACKER = 'bytetrack.yaml'
# Optional: apply winner chosen ONLY on NVIDIA training scenes. No retuning on external GT.
BEST_CONFIG_FILE = None  # e.g. ROOT / 'runs' / 'overnight_v1' / 'best_train_config.json'
MATCHER = 'baseline'
MIN_SIMILARITY = 0.80
MAX_COST = 0.125
APPEARANCE_WEIGHT = 0.95
SPATIAL_WEIGHT = 0.20
TEMPORAL_WEIGHT = 0.05
MAX_GAP_SECONDS = 12.0
MAX_SPEED_MPS = 3.5
SPATIAL_SLACK_M = 1.5
OVERLAP_DISTANCE_M = 2.0
CORRECT_IDENTITIES = False
ALLOW_SAME_CAMERA_FRAGMENTS = False
MEMORY = True
MEMORY_SIMILARITY = 0.95
REPAIR_OCCLUSIONS = True
GEOMETRY = False  # Only NVIDIA calibrated, validated floor matrices in this runner
RENDER = True
EVALUATE = True
LINK_ONLY = False
ALLOW_SCOUT_PSEUDO_LABELS = False  # Not recommended for selecting models
RUN_NAME = 'multi_dataset_v1'

# Root folders; place datasets here after downloading from official sites.
DATA_ROOTS = {
    'nvidia': ROOT / 'data' / 'MTMC_Tracking_2026',
    'citic': ROOT / 'data' / 'CITIC-MTMC',
    'scout': ROOT / 'data' / 'SCOUT',
}
# For CITIC, only if automatic camera recognition fails: numeric cam -> relative video path.
# Camera 0 HallWide, 1 HallSeg, 2 PF, 3 PC, 4 Showroom.
CITIC_VIDEO_MAP = {}
# ====================================================


def main():
    from mtmct.datasets import open_dataset
    from mtmct.association import Config
    from mtmct.occlusion_repair import RepairConfig
    from mtmct.tracking import run as track
    from mtmct.link import run as link
    from mtmct.visualize import render
    from mtmct.cache import write_config
    from mtmct.dataset_evaluate import evaluate_bbox
    from mtmct.evaluate import save_report
    from mtmct.time_sync import CameraTransitionModel

    # Start with manually configured settings, optionally override from pinned NVIDIA winner.
    opts = dict(reid_model=REID_MODEL, imgsz=IMGSZ, conf=CONF,
                tracker=TRACKER, matcher=MATCHER, min_similarity=MIN_SIMILARITY,
                max_cost=MAX_COST, appearance_weight=APPEARANCE_WEIGHT,
                spatial_weight=SPATIAL_WEIGHT, temporal_weight=TEMPORAL_WEIGHT,
                max_gap_seconds=MAX_GAP_SECONDS, max_speed_mps=MAX_SPEED_MPS,
                spatial_slack_m=SPATIAL_SLACK_M, overlap_distance_m=OVERLAP_DISTANCE_M,
                geometry=GEOMETRY, memory=MEMORY, memory_similarity=MEMORY_SIMILARITY,
                repair_occlusions=REPAIR_OCCLUSIONS,
                correct_identities=CORRECT_IDENTITIES,
                allow_same_camera_fragments=ALLOW_SAME_CAMERA_FRAGMENTS)
    repair_opts = {}
    if BEST_CONFIG_FILE is not None:
        config_path = Path(BEST_CONFIG_FILE)
        if not config_path.is_file():
            raise FileNotFoundError(f'Pinned train winner not found: {config_path}')
        best = json.loads(config_path.read_text(encoding='utf-8'))
        params = best['params']
        opts['reid_model'] = best['model']
        trial = best['detector_trial']
        if len(trial) != 7:
            raise ValueError('Expected 7 detector trial parameters in best_train_config.json')
        opts['imgsz'], opts['conf'] = int(trial[0]), float(trial[1])
        tracker_file = ROOT / 'runs' / RUN_NAME / 'pinned_tracker.yaml'
        tracker_file.parent.mkdir(parents=True,exist_ok=True)
        tracker_file.write_text('tracker_type: bytetrack\n'
                                f'track_high_thresh: {trial[2]}\n'
                                f'track_low_thresh: {trial[3]}\n'
                                f'new_track_thresh: {trial[4]}\n'
                                f'track_buffer: {trial[5]}\n'
                                f'match_thresh: {trial[6]}\n'
                                'fuse_score: true\n', encoding='utf-8')
        opts['tracker'] = str(tracker_file.resolve())
        for key in ('matcher','min_similarity','max_cost','appearance_weight',
                    'spatial_weight','temporal_weight','max_gap_seconds',
                    'max_speed_mps','spatial_slack_m','overlap_distance_m',
                    'memory','memory_similarity','repair_occlusions','correct_identities',
                    'allow_same_camera_fragments'):
            if key in params:
                opts[key] = params[key]
        opts['geometry'] = bool(params.get('use_geometry', False))
        repair_opts = {
            'duplicate_min_iou': params.get('duplicate_min_iou',0.45),
            'duplicate_min_similarity': params.get('duplicate_min_similarity',0.72),
            'max_gap_frames': params.get('repair_max_gap_frames',30),
            'min_witnessed_similarity': params.get('min_witnessed_similarity',0.72),
            'min_unwitnessed_similarity': params.get('min_unwitnessed_similarity',0.92),
            'strong_witness_min_similarity': params.get('strong_witness_min_similarity',0.65),
            'strong_witness_max_distance_m': params.get('strong_witness_max_distance_m',0.65),
        }
        print('PINNED NVIDIA TRAIN WINNER:', config_path, 'ReID:', opts['reid_model'])
    # Never import unverified NVIDIA geometry into a different coordinate system.
    if DATASET != 'nvidia' and opts['geometry']:
        print('WARNING: external dataset has a different calibration system. '
              'Disabling geometry and spatial weight, without changing other tuned parameters.')
        opts['geometry'], opts['spatial_weight'] = False, 0.0
    # CITIC video recordings are not guaranteed synchronized by frame index.
    use_temporal = DATASET != 'citic'
    if not use_temporal:
        opts['temporal_weight'] = 0.0
    ds = open_dataset(DATASET, DATA_ROOTS[DATASET], split=SPLIT,
                      allow_pseudo_labels=ALLOW_SCOUT_PSEUDO_LABELS,
                      video_map=CITIC_VIDEO_MAP)
    scenes = ds.list_scenes()
    if SCENE not in scenes:
        raise FileNotFoundError(f'Scene {SCENE!r} not found in {DATA_ROOTS[DATASET]}; available: {scenes}')
    available = ds.list_cameras(SCENE)
    cams = list(CAMERAS) if CAMERAS is not None else available
    if not cams or not set(cams).issubset(available):
        raise FileNotFoundError(f'Requested cameras {cams}; available: {available}. '
                                'For CITIC, use 0..4 or supply CITIC_VIDEO_MAP.')
    sources = ds.frame_sources(SCENE, cams)
    out = ROOT / 'runs' / RUN_NAME / DATASET / SCENE
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists() and not LINK_ONLY:
        raise FileExistsError(f'Output already contains an experiment: {out}. '
                              'Set a different RUN_NAME or LINK_ONLY=True.')
    write_config(out, {'dataset': DATASET, 'scene': SCENE, 'cameras': cams,
                       'sources': {str(k): str(v) for k,v in sources.items()},
                       'pipeline_options': opts, 'repair_options': repair_opts,
                       'source_best_config': str(BEST_CONFIG_FILE) if BEST_CONFIG_FILE else None, 'ground_truth': str(ds.get_ground_truth(SCENE)),
                       'fps': ds.fps, 'evaluation_requested': EVALUATE})
    gt = ds.get_ground_truth(SCENE)
    if EVALUATE and DATASET == 'citic':
        print('CITIC: world-only Ground Truth; custom bbox IDF1 is NOT valid. '
              'Processing video without auto-evaluation. Use separate CITIC world protocol after '
              'validating camera floor projections and time synchronization.')
    elif EVALUATE and not ds.supports_bbox_evaluation(SCENE, cams):
        raise ValueError(f'No trusted bbox GT for {DATASET}/{SCENE}/{cams}. '
                         'Set EVALUATE=False to run inference without evaluation.')
    if opts['geometry'] and DATASET != 'nvidia':
        raise ValueError('This runner only supports validated NVIDIA cameraMatrix floor geometry. '
                         'External camera calibration must be converted and verified first.')
    if opts['geometry'] and ds.get_calibration(SCENE) is None:
        raise FileNotFoundError('Geometry requested but NVIDIA calibration.json missing.')
    # Keep raw source path for NVIDIA so tracking can load calibration.json if present.
    # CITIC/SCOUT have no NVIDIA-layout cameraMatrix; geometry stays optional.
    scene_dir = ds._dir(SCENE) if DATASET == 'nvidia' else DATA_ROOTS[DATASET] / SCENE
    if not LINK_ONLY:
        track(scene_dir, out, MODEL, 'osnet', None, DEVICE, MAX_FRAMES, 1,
              SAMPLE_EVERY, opts['imgsz'], opts['conf'], reid_model=opts['reid_model'],
              cameras=cams, tracker=opts['tracker'], camera_sources=sources, fps_fallback=ds.fps)
    cfg = Config(appearance_weight=opts['appearance_weight'],
                 spatial_weight=opts['spatial_weight'],
                 temporal_weight=opts['temporal_weight'],
                 min_similarity=opts['min_similarity'], max_cost=opts['max_cost'],
                 max_gap_seconds=opts['max_gap_seconds'],
                 max_speed_mps=opts['max_speed_mps'],
                 spatial_slack_m=opts['spatial_slack_m'],
                 overlap_distance_m=opts['overlap_distance_m'],
                 fps=ds.fps, use_geometry=opts['geometry'],
                 use_time=use_temporal)
    rep = RepairConfig(**repair_opts)
    _, mapping, edges, predictions, _ = link(out, cfg, matcher=opts['matcher'],
                                              memory=opts['memory'],
                                              memory_similarity=opts['memory_similarity'],
                                              repair_occlusions=opts['repair_occlusions'],
                                              repair_config=rep,
                                              correct_identities=opts['correct_identities'],
                                              allow_same_camera_fragments=opts['allow_same_camera_fragments'])
    if RENDER:
        render(scene_dir, predictions, out / 'mosaic.mp4', MAX_FRAMES, 1, 0.45, cams,
               camera_sources=sources, fps_fallback=ds.fps)
    if EVALUATE and DATASET != 'citic':
        result = evaluate_bbox(predictions, ds, SCENE, cams, MAX_FRAMES)
        save_report(result, out / 'metrics.json')
        print('Global IDF1:', result['global']['IDF1'])
    print('DONE:', out, 'Global IDs:', len(set(mapping.values())))
    if DATASET == 'citic':
        print('Ground truth is world-position-only; see README for the separate protocol.')


if __name__ == '__main__':
    main()
