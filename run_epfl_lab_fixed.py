"""EPFL Laboratory-6p: download official AVI files / ▶ Run Python File in VS Code.

Drop this script in the MTMCT project root, next to mtmct/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# ========================== НАСТРОЙКИ ==============================
AUTO_DOWNLOAD = True          # False -> скачать AVI/GT/calibration вручную
DOWNLOAD_ATTEMPTS = 3       # Автоматические повторы при сетевых сбоях
METADATA_TIMEOUT_SECONDS = 15
VIDEO_TIMEOUT_SECONDS = 90
ALLOW_VIDEO_ONLY_IF_METADATA_UNAVAILABLE = True  # Не отменять YOLO из-за GT
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


def _validate_file(path: Path) -> None:
    """Reject incomplete files and HTML error pages stored with a .txt extension."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Нет файла или он пустой: {path}')
    if path.suffix.lower() == '.avi':
        with path.open('rb') as file:
            if file.read(4) != b'RIFF':
                raise ValueError(f'Это не AVI (ожидался RIFF): {path}')
    elif path.name == 'gt_lab_6p.txt':
        from mtmct.epfl_lab import read_gt
        data = read_gt(path)
        if data.people != 6 or data.grid_width != 56 or data.grid_height != 56:
            raise ValueError(f'Неверная GT Laboratory 6p: {path}')
    elif path.name == 'calibration-6p.txt':
        from mtmct.epfl_lab import read_ground_homographies
        data = read_ground_homographies(path)
        if set(data) != {0, 1, 2, 3}:
            raise ValueError(f'Неполная калибровка 6p: {path}')


def fetch(url: str, dest: Path, cap_mb: int):
    """Download atomically; reuse already valid files; retry transient HTTP timeouts."""
    if dest.exists():
        try:
            _validate_file(dest)
        except (ValueError, UnicodeError, OSError) as exc:
            if not AUTO_DOWNLOAD:
                raise RuntimeError(f'Локальный файл повреждён: {dest}\n{exc}') from exc
            print(f'Некорректный файл {dest.name}, попробуем загрузить заново: {exc}')
        else:
            print('Уже скачано, повторно не загружаем:', dest)
            return
    if not AUTO_DOWNLOAD:
        raise FileNotFoundError(f'Скачай файл вручную: {url}\nИ сохрани как: {dest}')

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + '.part')
    headers = {'User-Agent': 'Mozilla/5.0 (MTMCT research download)', 'Accept-Encoding': 'identity'}
    limit = cap_mb * 1024 * 1024
    timeout = VIDEO_TIMEOUT_SECONDS if dest.suffix.lower() == '.avi' else METADATA_TIMEOUT_SECONDS
    last_exc = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        temp.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as src, temp.open('wb') as dst:
                announced = int(src.headers.get('Content-Length') or 0)
                if announced > limit:
                    raise RuntimeError(f'Слишком большой файл {dest.name}: {announced / 1024**2:.0f} МБ')
                total = 0
                while True:
                    block = src.read(256 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > limit:
                        raise RuntimeError(f'Превышен лимит {cap_mb} МБ для {dest.name}')
                    dst.write(block)
                    if dest.suffix.lower() == '.avi' and total // (25 * 1024 * 1024) > (total - len(block)) // (25 * 1024 * 1024):
                        print(f'  скачано {total / 1024**2:.0f} МБ: {dest.name}', flush=True)
            # Validate basic downloaded content before replacing the target file.
            if dest.suffix.lower() != '.avi':
                contents = temp.read_text(encoding='utf-8-sig')
                if not contents.strip() or '<html' in contents[:2048].lower() or '<!doctype' in contents[:2048].lower():
                    raise ValueError(f'Сервер вернул HTML/пустой текст вместо {dest.name}')
            temp.replace(dest)
            try:
                _validate_file(dest)
            except Exception:
                # A corrupt download must not be used on the next invocation.
                dest.unlink(missing_ok=True)
                raise
            print('Готово:', dest)
            return
        except (RuntimeError, ValueError, UnicodeError) as exc:
            # Corrupt content / invalid response: retries may succeed on a temporary server glitch.
            last_exc = exc
        except OSError as exc:
            last_exc = exc
        finally:
            temp.unlink(missing_ok=True)
        print(f'Загрузка {dest.name}: попытка {attempt}/{DOWNLOAD_ATTEMPTS} не удалась: '
              f'{type(last_exc).__name__}: {last_exc}', flush=True)
        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(
        f'Не удалось загрузить {dest.name} после {DOWNLOAD_ATTEMPTS} попыток.\n'
        f'Скачай вручную: {url}\nСохрани в: {dest}\n'
        f'Последняя ошибка: {last_exc}'
    ) from last_exc


def prepare_inputs():
    videos = {}
    for cam in CAMERAS:
        path = SCENE / 'videos' / f'6p-c{cam}.avi'
        # For LINK_ONLY no network call is needed for video: the cached predictions exist.
        if not LINK_ONLY:
            fetch(VIDEO_BASE + f'6p-c{cam}.avi', path, VIDEO_DOWNLOAD_LIMIT_MB)
        elif RENDER:
            _validate_file(path)
        videos[cam] = path

    evaluation_ready = bool(EVALUATE)
    if EVALUATE:
        try:
            fetch(GT_URL, SCENE / 'gt_lab_6p.txt', 8)
            fetch(CALIB_URL, SCENE / 'calibration-6p.txt', 8)
            from mtmct.epfl_lab import read_gt, read_ground_homographies
            gt = read_gt(SCENE / 'gt_lab_6p.txt')
            hs = read_ground_homographies(SCENE / 'calibration-6p.txt')
            print(f'EPFL GT: {gt.people} человек, сетка {gt.grid_width}x{gt.grid_height}, '
                  f'аннотации через {gt.step} кадров; гомографии: {sorted(hs)}')
        except (OSError, RuntimeError, ValueError, UnicodeError) as exc:
            if not ALLOW_VIDEO_ONLY_IF_METADATA_UNAVAILABLE:
                raise
            evaluation_ready = False
            print('\nВНИМАНИЕ: GT или калибровка недоступны. '
                  'YOLO/OSNet всё равно будут запущены, но метрики не вычисляются.\n'
                  f'Причина: {exc}\n'
                  f'GT: {GT_URL}\nКалибровка: {CALIB_URL}\n', flush=True)

    if not LINK_ONLY:
        import cv2
        for cam, path in videos.items():
            v = cv2.VideoCapture(str(path))
            if not v.isOpened():
                raise RuntimeError(f'OpenCV не может открыть AVI: {path}')
            ok, frame = v.read()
            fps = v.get(cv2.CAP_PROP_FPS)
            v.release()
            if not ok:
                raise RuntimeError(f'OpenCV не может прочитать первый кадр: {path}')
            print(f'Камера {cam}: {frame.shape[1]}x{frame.shape[0]}, FPS={fps:.2f}')
    return videos, evaluation_ready


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
    defaults['use_geometry']=True
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

    sources, evaluation_ready = prepare_inputs()
    params=configuration()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists() and not LINK_ONLY:
        raise FileExistsError(f'Эксперимент уже существует: {OUT}. Измени RUN_NAME или LINK_ONLY=True')
    tracker=tracker_yaml(params)
    (OUT/'epfl_config.json').write_text(json.dumps({
        'cameras':CAMERAS,'max_frames':MAX_FRAMES,
        'dataset':'EPFL Laboratory 6p','params':params,
        'evaluation_requested':bool(EVALUATE),
        'evaluation_enabled':bool(evaluation_ready),
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
    if evaluation_ready:
        report=evaluate_sparse_ground(
            predictions,SCENE/'gt_lab_6p.txt',SCENE/'calibration-6p.txt',
            OUT/'epfl_sparse_ground_metrics.json',CAMERAS,MAX_FRAMES,GT_DISTANCE_CELLS)
        print('EPFL spatial F1:',report['spatial_f1'])
        print('EPFL sparse-ground IDF1 PROXY:',report['sparse_ground_idf1_proxy'])
        print('Это не стандартный Global IDF1 NVIDIA и не официальная метрика EPFL!')
    print('РЕЗУЛЬТАТЫ:',OUT,'Групп Global ID:',len(set(mapping.values())))


if __name__=='__main__':
    main()
