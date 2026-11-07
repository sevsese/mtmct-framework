"""EPFL Laboratory-6p: download official AVI files / ▶ Run Python File in VS Code.

Drop this script in the MTMCT project root, next to mtmct/.
"""
from __future__ import annotations

from pathlib import Path as _LauncherPath
import sys as _launcher_sys
_launcher_sys.path.insert(0, str(_LauncherPath(__file__).resolve().parents[1]))
_launcher_sys.path.insert(0, str(_LauncherPath(__file__).resolve().parent))
import json
import os
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# ========================== НАСТРОЙКИ ==============================
AUTO_DOWNLOAD = True          # False -> скачать AVI вручную по ссылкам в README
CAMERAS = (0, 1, 2, 3)       # Синхронные камеры
MAX_FRAMES = 1000            # Первые 40 секунд при 25 FPS; 0 = все кадры
VIDEO_DOWNLOAD_LIMIT_MB = 500  # Ограничение на ОДИН файл; превышение -> ошибка
DEVICE = 'cuda:0'
DETECTOR = 'yolo26l.pt'
MODEL = 'osnet_ain_x1_0'      # Можно сменить на osnet_ibn_x1_0
IMGSZ = 1536
CONF = 0.07
SAMPLE_EVERY = 5
MEMORY = True
REPAIR_OCCLUSIONS = True
EVALUATE = True              # Кастомная GT-оценка по сетке пола
RENDER = True
LINK_ONLY = False            # True -> повторно выполнить только association
RUN_NAME = 'epfl_lab6p_best_v1'  # Измени имя для нового опыта
GT_DISTANCE_CELLS = 2.0     # Допуск сопоставления позиций в ячейках сетки
BEST_CONFIG_FILE = ROOT / 'runs' / 'overnight_v1' / 'best_train_config.json'
# ===================================================================

SCENE=ROOT/'data'/'EPFL'/'Laboratory_6p'
OUT=ROOT/'runs'/RUN_NAME
VIDEO_BASE='https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/'
GT_URL='https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/gt_lab_6p.txt'
CALIB_URL='https://www.epfl.ch/labs/cvlab/wp-content/uploads/2018/08/calibration-6p.txt'


def fetch(url: str, dest: Path, cap_mb: int):
    if dest.is_file() and dest.stat().st_size:
        print('Уже скачано:',dest)
        return
    if not AUTO_DOWNLOAD:
        raise FileNotFoundError(f'Скачай файл вручную: {url}\nИ сохрани как: {dest}')
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix(dest.suffix+'.part')
    headers={'User-Agent':'Mozilla/5.0 (MTMCT research download)'}
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as src, temp.open('wb') as dst:
            size=int(src.headers.get('Content-Length') or 0)
            limit=cap_mb*1024*1024
            if size>limit:
                raise RuntimeError(f'{url} requires {size/1024**2:.0f} MB > {cap_mb} MB limit')
            total=0
            while True:
                block=src.read(1024*1024)
                if not block: break
                total+=len(block)
                if total>limit:
                    raise RuntimeError(f'Остановлена загрузка: {url} > {cap_mb} MB')
                dst.write(block)
                if total%(25*1024*1024)<1024*1024:
                    print(f'  скачано {total/1024**2:.0f} MB: {dest.name}',flush=True)
        if not total:
            raise RuntimeError(f'Пустой файл: {url}')
        if dest.suffix.lower()=='.avi':
            with temp.open('rb') as f:
                if f.read(4)!=b'RIFF':
                    raise RuntimeError(f'Server returned non-AVI content for {url}')
        temp.replace(dest)
        print('Готово:',dest)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def prepare_inputs():
    videos={}
    for cam in CAMERAS:
        path=SCENE/'videos'/f'6p-c{cam}.avi'
        fetch(VIDEO_BASE+f'6p-c{cam}.avi',path,VIDEO_DOWNLOAD_LIMIT_MB)
        videos[cam]=path
    fetch(GT_URL,SCENE/'gt_lab_6p.txt',8)
    fetch(CALIB_URL,SCENE/'calibration-6p.txt',8)
    from mtmct.epfl_lab import read_gt,read_ground_homographies
    gt=read_gt(SCENE/'gt_lab_6p.txt')
    hs=read_ground_homographies(SCENE/'calibration-6p.txt')
    print(f'EPFL GT: {gt.people} человек, сетка {gt.grid_width}x{gt.grid_height}, '
          f'аннотации через {gt.step} кадров; гомографии: {sorted(hs)}')
    import cv2
    for cam,path in videos.items():
        v=cv2.VideoCapture(str(path))
        if not v.isOpened():
            raise RuntimeError(f'OpenCV cannot decode video: {path}; consider converting AVI to MP4 with ffmpeg')
        ok,frame=v.read()
        fps=v.get(cv2.CAP_PROP_FPS)
        v.release()
        if not ok: raise RuntimeError(f'Cannot read first frame: {path}')
        print(f'Камера {cam}: {frame.shape[1]}x{frame.shape[0]}, FPS={fps:.2f}')
    return videos


def configuration():
    defaults={
        'reid_model':MODEL,
        'detector_trial':[IMGSZ,CONF,0.12,0.02,0.08,60,0.82],
        'matcher':'baseline','appearance_weight':0.8776,'temporal_weight':0.0753,
        'min_similarity':0.743,'max_cost':0.226,'max_gap_seconds':40.0,
        'memory':MEMORY,'memory_similarity':0.907,'repair_occlusions':REPAIR_OCCLUSIONS,
        'allow_same_camera_fragments':True,
    }
    if BEST_CONFIG_FILE.is_file():
        saved=json.loads(BEST_CONFIG_FILE.read_text(encoding='utf-8'))
        p=saved['params']
        defaults.update(p)
        defaults['reid_model']=saved['model']
        defaults['detector_trial']=saved['detector_trial']
        print('Зафиксированный победитель NVIDIA:',BEST_CONFIG_FILE)
    else:
        print('best_train_config.json отсутствует — используем опубликованные значения ночного поиска')
    # EPFL H maps to non-metric top-view coordinates; NVIDIA speed/distance is NOT portable.
    defaults['use_geometry']=False
    defaults['spatial_weight']=0.0
    print('Геометрия association отключена: система координат EPFL отличается от NVIDIA.')
    return defaults


def tracker_yaml(p):
    t=p['detector_trial']
    if len(t)!=7: raise ValueError('Expected 7 detector trial values')
    tracker=OUT/'epfl_bytetrack.yaml'
    tracker.parent.mkdir(parents=True,exist_ok=True)
    tracker.write_text('tracker_type: bytetrack\n'
                       f'track_high_thresh: {t[2]}\n'
                       f'track_low_thresh: {t[3]}\n'
                       f'new_track_thresh: {t[4]}\n'
                       f'track_buffer: {int(t[5])}\n'
                       f'match_thresh: {t[6]}\n'
                       'fuse_score: true\n',encoding='utf-8')
    return tracker


def main():
    from mtmct.association import Config
    from mtmct.occlusion_repair import RepairConfig
    from mtmct.tracking import run as track
    from mtmct.link import run as link
    from mtmct.visualize import render
    from mtmct.epfl_lab import evaluate_sparse_ground

    sources=prepare_inputs() if not LINK_ONLY else {
        cam:SCENE/'videos'/f'6p-c{cam}.avi' for cam in CAMERAS}
    params=configuration()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists() and not LINK_ONLY:
        raise FileExistsError(f'Эксперимент уже существует: {OUT}. Измени RUN_NAME или LINK_ONLY=True')
    tracker=tracker_yaml(params)
    (OUT/'epfl_config.json').write_text(json.dumps({
        'cameras':CAMERAS,'max_frames':MAX_FRAMES,
        'dataset':'EPFL Laboratory 6p','params':params,
        'note':'NVIDIA geometry disabled, EPFL GT evaluated only at labelled frames'},indent=2),encoding='utf-8')
    if not LINK_ONLY:
        t=params['detector_trial']
        track(SCENE,OUT,DETECTOR,'osnet',None,DEVICE,MAX_FRAMES,1,SAMPLE_EVERY,
              int(t[0]),float(t[1]),reid_model=params['reid_model'],
              cameras=list(CAMERAS),tracker=tracker,camera_sources=sources,fps_fallback=25.0)
    cfg=Config(
        appearance_weight=float(params['appearance_weight']),
        spatial_weight=0.0,temporal_weight=float(params['temporal_weight']),
        min_similarity=float(params['min_similarity']),max_cost=float(params['max_cost']),
        max_gap_seconds=float(params['max_gap_seconds']),fps=25.0,
        use_geometry=False,use_time=bool(params.get('use_time',True)))
    rep=RepairConfig(
        duplicate_min_iou=params.get('duplicate_min_iou',0.45),
        duplicate_min_similarity=params.get('duplicate_min_similarity',0.72),
        max_gap_frames=int(params.get('repair_max_gap_frames',30)),
        min_witnessed_similarity=params.get('min_witnessed_similarity',0.72),
        min_unwitnessed_similarity=params.get('min_unwitnessed_similarity',0.92),
        strong_witness_min_similarity=params.get('strong_witness_min_similarity',0.65),
        strong_witness_max_distance_m=params.get('strong_witness_max_distance_m',0.65))
    _,mapping,edges,predictions,_=link(OUT,cfg,
        matcher=params.get('matcher','baseline'),
        memory=bool(params.get('memory',True)),
        memory_similarity=float(params.get('memory_similarity',0.907)),
        repair_occlusions=bool(params.get('repair_occlusions',True)),
        repair_config=rep,
        allow_same_camera_fragments=bool(params.get('allow_same_camera_fragments',True)))
    if RENDER:
        render(SCENE,predictions,OUT/'mosaic.mp4',MAX_FRAMES,1,.45,list(CAMERAS),
               camera_sources=sources,fps_fallback=25.0)
    if EVALUATE:
        report=evaluate_sparse_ground(
            predictions,SCENE/'gt_lab_6p.txt',SCENE/'calibration-6p.txt',
            OUT/'epfl_sparse_ground_metrics.json',CAMERAS,MAX_FRAMES,GT_DISTANCE_CELLS)
        print('EPFL spatial F1:',report['spatial_f1'])
        print('EPFL sparse-ground IDF1 PROXY:',report['sparse_ground_idf1_proxy'])
        print('Это не стандартный Global IDF1 NVIDIA и не официальная метрика EPFL!')
    print('РЕЗУЛЬТАТЫ:',OUT,'Групп Global ID:',len(set(mapping.values())))


if __name__=='__main__':
    main()
