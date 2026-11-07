"""Native MTMCT artifacts -> indexed DTOs. Never infer or assign identities."""
from __future__ import annotations
import csv
import json
import math
import os
import sqlite3
from pathlib import Path
os.environ.setdefault('OPENCV_FFMPEG_THREADS', '1')
import cv2
import ijson

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {'observations.csv', 'global_tracks_2d.csv', 'local_tracklets.json',
             'refined_tracklets.json', 'global_ids.json', 'association_edges.json',
             'performance.json', 'manifest.json', 'config.yaml', 'metrics.json', 'global_memory.json'}


def read_json(folder: Path, name: str, default=None):
    path = folder / name
    if not path.is_file():
        return default
    if path.stat().st_size > 200 * 1024**2:
        raise ValueError(f'{name}: JSON превышает предел 200 МБ')
    try:
        if name in {'local_tracklets.json','refined_tracklets.json'}:
            with path.open('rb') as stream:
                prefix = stream.read(512).decode('utf-8-sig').lstrip()
                if not prefix.startswith('['):
                    raise ValueError(f'{name}: ожидается массив треков')
                stream.seek(0)
                if stream.read(1) == b'\xef':
                    stream.seek(3)
                else:
                    stream.seek(0)
                # ReID vectors and per-frame geometry never need to enter the web DTO.
                result=[]
                for item in ijson.items(stream,'item',use_float=True):
                    if not isinstance(item,dict):
                        raise ValueError(f'{name}: трек должен быть объектом')
                    result.append({k:v for k,v in item.items() if k in {'camera_id','local_id','start_frame','end_frame','fps','start_s','end_s','tracklet_uid'}})
                return result
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (ValueError, UnicodeError, ijson.JSONError) as exc:
        raise ValueError(f'{name}: повреждён JSON: {exc}') from exc


def finite(value):
    if value in (None, ''):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Число должно быть конечным')
    return number


def office_camera_ids(cameras):
    """Map uploaded IDs to the office preset's zero-based camera numbers."""
    mapping = {}
    for camera in cameras:
        name = Path(camera.get('path') or '').stem.lower()
        if name in {f'camera_new_{n}' for n in range(1, 6)}:
            mapping[camera['camera_id']] = int(name[-1]) - 1
    return mapping if len(cameras) == 5 and set(mapping.values()) == set(range(5)) else {}


def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'OpenCV не может открыть видео {path.name}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        if not ok or fps <= 0:
            raise ValueError(f'Не удалось декодировать видео {path.name}')
        return {'fps': fps, 'duration': frames/fps, 'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def build_index(folder: Path, database: Path, supplied_cameras: list[dict]):
    manifest = read_json(folder, 'manifest.json', {})
    config = read_json(folder, 'config.yaml', {})
    if not isinstance(manifest, dict) or not isinstance(config, dict):
        raise ValueError('manifest.json и config.yaml должны быть объектами')
    for key,kind in [('tracking_config',dict),('models',dict),('inputs',list),('cameras',list)]:
        if key in manifest and not isinstance(manifest[key],kind):
            raise ValueError(f'manifest.json: неверный тип поля {key}')
    if any(not isinstance(item,dict) for item in manifest.get('inputs', [])):
        raise ValueError('manifest.json: inputs должен содержать объекты')
    tracking_config = manifest.get('tracking_config', {})
    if not isinstance(tracking_config.get('camera_sources',{}),dict):
        raise ValueError('manifest.json: camera_sources должен быть объектом')
    if not isinstance(tracking_config.get('camera_offsets_s',{}),dict):
        raise ValueError('manifest.json: camera_offsets_s должен быть объектом')
    if not isinstance(config.get('camera_offset',[]),list) or any(not isinstance(v,str) for v in config.get('camera_offset',[])):
        raise ValueError('config.yaml: camera_offset должен быть массивом строк camera=seconds')
    warnings = []
    if not manifest:
        warnings.append('Нет manifest.json: происхождение результата не подтверждено')
    elif manifest.get('schema_version') != 'mtmct-cache-v1':
        warnings.append('Неизвестная версия manifest; отображаются только известные поля')
    from mtmct.cache import fingerprint
    for item in manifest.get('inputs', []):
        path = Path(item.get('path','')).resolve()
        if path.is_relative_to(ROOT) and path.is_file():
            if item.get('sha256') and fingerprint(path)['sha256'] != item['sha256']:
                warnings.append(f'Manifest: входной файл изменён — {path.name}')
        else:
            warnings.append(f'Manifest: исходный файл недоступен для проверки — {path.name}')
    for name in ARTIFACTS - {'observations.csv', 'global_tracks_2d.csv', 'config.yaml', 'manifest.json'}:
        value = read_json(folder, name)
        if value is not None:
            expected = dict if name in {'metrics.json', 'global_memory.json'} else list
            if not isinstance(value, expected):
                raise ValueError(f'{name}: ожидается {expected.__name__}')
            if expected is list and any(not isinstance(item,dict) for item in value):
                raise ValueError(f'{name}: массив должен содержать объекты')
            if name=='performance.json':
                for item in value:
                    for key in ('seconds','processed_frames','fps','peak_cuda_tensor_gib'):
                        number=finite(item.get(key))
                        if number is not None and number<0:
                            raise ValueError(f'{name}: {key} не может быть отрицательным')
            if name=='metrics.json':
                for key in ('global','detection','pairwise_association'):
                    if value.get(key) is not None and not isinstance(value[key],dict):
                        raise ValueError(f'{name}: {key} должен быть объектом')
    cameras = {int(c['camera_id']): dict(c) for c in supplied_cameras}
    offsets = {}
    for camera, seconds in tracking_config.get('camera_offsets_s', {}).items():
        value = finite(seconds)
        if value is None:
            raise ValueError('manifest.json: camera_offsets_s должен содержать числа')
        offsets[int(camera)] = value
    for entry in config.get('camera_offset', []):
        camera, seconds = entry.split('=', 1)
        offsets[int(camera)] = finite(seconds)
    sources = tracking_config.get('camera_sources', {})
    office_names = {int(camera): Path(name).name.lower() for camera, name in sources.items()}
    if not offsets and office_names == {camera: f'camera_new_{camera+1}.mkv' for camera in range(5)}:
        offsets = dict(enumerate((0., 1., 1., 4., 3.)))
        warnings.append('Применена синхронизация времени офисных камер 0/1/1/4/3 с')
    for camera, name in sources.items():
        path = Path(name)
        path = (ROOT/path).resolve() if not path.is_absolute() else path.resolve()
        # Imported manifests cannot expose arbitrary files on this computer.
        if path.is_relative_to(ROOT) and path.is_file() and path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
            cameras.setdefault(int(camera), {'camera_id': int(camera), 'path': str(path)})
    mapping = {}
    for row in read_json(folder, 'global_ids.json', []):
        key = (int(row['camera_id']), int(row['local_id']))
        if key in mapping:
            raise ValueError('Повторный ключ в global_ids.json')
        mapping[key] = int(row['global_id'])
        if min(*key,mapping[key])<0:
            raise ValueError('global_ids.json: отрицательный идентификатор')
        cameras.setdefault(key[0],{'camera_id':key[0]})
    tracks = read_json(folder, 'refined_tracklets.json', None)
    if tracks is None:
        tracks = read_json(folder, 'local_tracklets.json', [])
    for t in tracks:
        camera = int(t['camera_id'])
        c = cameras.setdefault(camera, {'camera_id': camera})
        if t.get('fps'):
            fps=finite(t['fps'])
            if fps<=0:
                raise ValueError('local_tracklets.json: FPS должен быть положительным')
            c.setdefault('fps', fps)
    for cam in manifest.get('cameras', []):
        cameras.setdefault(int(cam), {'camera_id': int(cam)})
    for cam, c in cameras.items():
        c.setdefault('name', f'Camera {cam}')
        c.setdefault('offset', offsets.get(cam, 0.0))
        if c.get('path'):
            c.update(video_info(Path(c['path'])))
        else:
            warnings.append(f'Camera {cam}: видео не подключено')
        if not c.get('fps'):
            # A stored fallback is configuration evidence, not measured video FPS.
            c['fps'] = finite(manifest.get('tracking_config', {}).get('fps_fallback'))
            c['fps_source'] = 'manifest fallback' if c['fps'] else None
        if c.get('fps') is not None and c['fps']<=0:
            raise ValueError(f'Camera {cam}: FPS должен быть положительным')
    connection = sqlite3.connect(database)
    try:
        connection.executescript('DROP TABLE IF EXISTS observations; CREATE TABLE observations('
            'camera_id INTEGER,frame_id INTEGER,local_id INTEGER,global_id INTEGER,timestamp REAL,'
            'x1 REAL,y1 REAL,x2 REAL,y2 REAL,confidence REAL, PRIMARY KEY(camera_id,frame_id,local_id));')
        source = folder/'global_tracks_2d.csv'
        has_export = source.is_file()
        if not has_export:
            source = folder/'observations.csv'
        if source.is_file():
            with source.open(newline='', encoding='utf-8-sig') as stream:
                reader = csv.DictReader(stream)
                required = {'camera_id', 'frame_id', 'local_id', 'x1', 'y1', 'x2', 'y2'}
                if has_export:
                    required.add('global_id')
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f'{source.name}: отсутствуют столбцы {sorted(required - set(reader.fieldnames or []))}')
                for line, row in enumerate(reader, 2):
                    try:
                        cam, frame, local = (int(row[k]) for k in ('camera_id', 'frame_id', 'local_id'))
                        if min(cam, frame, local) < 0:
                            raise ValueError('Отрицательный ID/номер кадра')
                        c = cameras.setdefault(cam, {'camera_id': cam, 'name': f'Camera {cam}', 'offset': 0.})
                        fps = c.get('fps')
                        timestamp = frame/fps + c.get('offset', 0) if fps else None
                        gid = int(row['global_id']) if has_export else mapping.get((cam, local))
                        if gid is not None and gid<0:
                            raise ValueError('Отрицательный Global ID')
                        if has_export and (cam, local) in mapping and mapping[cam, local] != gid:
                            raise ValueError('global_ids.json не соответствует global_tracks_2d.csv')
                        box = [finite(row[k]) for k in ('x1', 'y1', 'x2', 'y2')]
                        if None in box or box[2] <= box[0] or box[3] <= box[1]:
                            raise ValueError('Неверная рамка bbox')
                        connection.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)',
                                           [cam, frame, local, gid, timestamp, *box, finite(row.get('confidence'))])
                    except (ValueError, TypeError, sqlite3.IntegrityError) as exc:
                        raise ValueError(f'{source.name}, строка {line}: {exc}') from exc
        else:
            warnings.append('Нет CSV наблюдений: рамки и последовательность кадров недоступны')
        if manifest.get('cameras') and set(cameras) - set(map(int, manifest['cameras'])):
            warnings.append('Список камер не совпадает с manifest')
        connection.executescript('CREATE INDEX obs_time ON observations(camera_id,timestamp);'
                                'CREATE INDEX obs_gid ON observations(global_id,timestamp);')
        connection.commit()
    finally:
        connection.close()
    known_local={(int(t['camera_id']),int(t['local_id'])) for t in tracks} | set(mapping)
    return {'cameras': list(cameras.values()), 'warnings': warnings, 'manifest': manifest,
            'known_local_count':len(known_local) if known_local else None,
            'config': config, 'files': sorted(p.name for p in folder.iterdir() if p.name in ARTIFACTS)}


def rows(database: Path, sql: str, parameters=()):
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute(sql, parameters)]


def observation_dto(row):
    return {k: row[k] for k in ('camera_id','frame_id','local_id','global_id','timestamp','confidence')} | {
        'bbox': [row[k] for k in ('x1','y1','x2','y2')]}
