from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import uuid
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from backend.adapter import ROOT, ARTIFACTS, build_index, read_json, rows, observation_dto, office_camera_ids, video_info
from backend.floor_map import fit as fit_floor, project as project_floor

STORE = Path(os.environ.get('MTMCT_WEB_STORE', ROOT/'data/web')).resolve()
STORE.mkdir(parents=True, exist_ok=True)
executor = ThreadPoolExecutor(max_workers=1)
lock = threading.RLock()
reviews = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    with lock:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)


def directory(identifier):
    try:
        uuid.UUID(identifier)
    except ValueError:
        raise HTTPException(404, 'Набор не найден')
    path = STORE/identifier
    if not path.is_dir():
        raise HTTPException(404, 'Набор не найден')
    return path


def dataset(identifier):
    path = directory(identifier)
    if not (path/'dataset.json').is_file():
        raise HTTPException(404, 'Набор не найден')
    return read_json(path, 'dataset.json')


@asynccontextmanager
async def lifespan(app):
    for path in STORE.glob('*/job.json'):
        job = read_json(path.parent, path.name)
        if job['status'] in {'queued', 'running'}:
            job.update(status='failed', error='Backend был перезапущен; запустите обработку снова', finished_at=now())
            save(path, job)
    yield


app = FastAPI(title='MTMCT Local API', version='1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:5173', 'http://localhost:5173'],
                   allow_methods=['GET','POST','PUT'], allow_headers=['Content-Type'])


@app.middleware('http')
async def local_origin(request, call_next):
    origin = request.headers.get('origin')
    if request.method not in {'GET','HEAD','OPTIONS'} and origin and origin not in {
        'http://127.0.0.1:5173','http://localhost:5173','http://127.0.0.1:8000','http://localhost:8000'}:
        return JSONResponse({'detail': 'Недопустимый источник запроса'}, status_code=403)
    return await call_next(request)


@app.exception_handler(ValueError)
@app.exception_handler(ValidationError)
@app.exception_handler(KeyError)
@app.exception_handler(TypeError)
async def invalid_data(request, error):
    return JSONResponse({'detail': str(error)}, status_code=422)


@app.exception_handler(OSError)
async def filesystem_error(request, error):
    return JSONResponse({'detail': f'Ошибка файловой системы: {error}'}, status_code=409)


@app.get('/api/health')
def health():
    return {'status': 'ok', 'adapter': 'native-mtmct-v1'}


def public(data):
    return {k: data.get(k) for k in ('id','name','mode','created_at','status','warnings','files')}


@app.get('/api/datasets')
def datasets():
    return [public(read_json(p.parent,p.name)) for p in sorted(STORE.glob('*/dataset.json'), reverse=True)]


@app.get('/api/runs')
def available_runs():
    return sorted({str(p.parent.relative_to(ROOT)) for p in (ROOT/'runs').rglob('global_tracks_2d.csv')})


class ImportRun(BaseModel):
    relative_path: str
    name: str = Field(default='', max_length=120)


def register(folder, name, mode, cameras, owned=None):
    identifier = owned.name if owned else str(uuid.uuid4())
    owned = owned or STORE/identifier
    owned.mkdir(exist_ok=True)
    metadata = build_index(folder, owned/'index.sqlite', cameras)
    data = metadata | {'id': identifier, 'name': name or folder.name, 'mode': mode,
                       'created_at': now(), 'status': 'ready' if mode == 'results' else 'uploaded',
                       'source': str(folder)}
    office_floor(owned, folder, data['cameras'])
    save(owned/'dataset.json', data)
    return public(data)


def office_floor(owned, result, cameras):
    """Load the existing approximate calibration for the five office videos."""
    config_path = ROOT/'configs'/'hackaton_office_map_approx.json'
    plan_path = ROOT/'data'/'Hackaton Data'/'office_camera.png'
    if not plan_path.is_file():
        plan_path = Path.home()/'Downloads'/'office_camera.png'
    ids = office_camera_ids(cameras)
    if not ids or not config_path.is_file() or not plan_path.is_file():
        return
    config = read_json(config_path.parent, config_path.name)
    calibrations = {}
    for camera_id, office_id in ids.items():
        row = config['cameras'][str(office_id)]
        points = [
            {'video': video, 'plan': plan}
            for video, plan in zip(row['image_points_norm'], row['map_points_norm'])
        ]
        calibrations[str(camera_id)] = fit_floor(points)
    shutil.copyfile(plan_path, owned/'floorplan.png')
    save(owned/'floor_calibration.json', calibrations)


@app.post('/api/datasets/import')
def import_run(body: ImportRun):
    folder = (ROOT/body.relative_path).resolve()
    if not folder.is_relative_to(ROOT/'runs') or not folder.is_dir():
        raise HTTPException(422, 'Выберите существующую папку внутри runs')
    if not any((folder/n).is_file() for n in ARTIFACTS):
        raise HTTPException(422, 'В папке нет файлов результатов MTMCT')
    return register(folder, body.name, 'results', [])


class CameraInput(BaseModel):
    camera_id: int = Field(ge=0, le=100000)
    filename: str
    name: str = Field(default='', max_length=120)
    offset: float = Field(default=0, ge=-86400, le=86400, allow_inf_nan=False)
    fps: float | None = Field(default=None, gt=0, le=1000, allow_inf_nan=False)


@app.post('/api/datasets/upload')
async def upload(files: list[UploadFile] = File(...), name: str = Form('Новый набор'),
                 mode: str = Form('results'), cameras: str = Form('[]')):
    if mode not in {'raw','results'}:
        raise HTTPException(422, 'Режим должен быть raw или results')
    camera_models = [CameraInput.model_validate(c) for c in json.loads(cameras)]
    if len({c.camera_id for c in camera_models}) != len(camera_models):
        raise HTTPException(422, 'Camera ID должны быть уникальными')
    if len({c.filename for c in camera_models}) != len(camera_models):
        raise HTTPException(422, 'Один файл нельзя назначить нескольким камерам')
    names = [f.filename for f in files]
    if len(names) != len(set(names)):
        raise HTTPException(422, 'Имена файлов должны быть уникальными')
    camera_files = {c.filename for c in camera_models}
    if mode == 'raw' and not camera_models:
        raise HTTPException(422, 'Добавьте видео и Camera ID')
    if not camera_files.issubset(names):
        raise HTTPException(422, 'Не все видео камер переданы')
    if mode == 'raw' and set(names) != camera_files:
        raise HTTPException(422, 'Режим исходных видео принимает только видео')
    for filename in names:
        if not filename or '/' in filename or '\\' in filename or ':' in filename:
            raise HTTPException(422, 'Недопустимое имя файла')
        if filename in camera_files:
            if Path(filename).suffix.lower() not in {'.mp4','.avi','.mov','.mkv'}:
                raise HTTPException(422, 'Для декодирования разрешены MP4, AVI, MOV, MKV')
        elif filename not in ARTIFACTS:
            raise HTTPException(422, f'Неизвестный файл результатов: {filename}')
    owned = STORE/str(uuid.uuid4())
    owned.mkdir()
    folder = owned/'input'
    folder.mkdir()
    for file in files:
        size = 0
        limit = 4*1024**3 if file.filename in camera_files else 200*1024**2
        with (folder/file.filename).open('wb') as stream:
            while chunk := await file.read(1024**2):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f'{file.filename}: превышен предел {limit//1024**2} МБ')
                stream.write(chunk)
    camera_data = [c.model_dump() | {'path': str(folder/c.filename), 'name': c.name or f'Camera {c.camera_id}'} for c in camera_models]
    return register(folder, name[:120], mode, camera_data, owned)


def track_list(identifier):
    data = dataset(identifier)
    db = directory(identifier)/'index.sqlite'
    grouped = rows(db, 'SELECT global_id,MIN(timestamp) start_time,MAX(timestamp) end_time,COUNT(*) observations_count '
                       'FROM observations WHERE global_id IS NOT NULL GROUP BY global_id')
    relations = rows(db, 'SELECT DISTINCT global_id,camera_id,local_id FROM observations WHERE global_id IS NOT NULL')
    by_id = {r['global_id']: r for r in grouped}
    if not grouped:
        for m in read_json(Path(data['source']), 'global_ids.json', []):
            gid = m['global_id']
            by_id.setdefault(gid, {'global_id': gid, 'start_time': None, 'end_time': None, 'observations_count': None})
            relations.append(m)
    local_by_id = {}
    for r in relations:
        local_by_id.setdefault(r['global_id'], []).append({'camera_id': r['camera_id'], 'local_id': r['local_id']})
    for t in by_id.values():
        t['local_tracklets'] = local_by_id.get(t['global_id'], [])
        t['cameras'] = sorted({r['camera_id'] for r in t['local_tracklets']})
        t['duration'] = t['end_time']-t['start_time'] if t['start_time'] is not None else None
        t['status'] = None
    return list(by_id.values())


@app.get('/api/datasets/{identifier}')
def detail(identifier: str):
    d = dataset(identifier)
    db = directory(identifier)/'index.sqlite'
    counts = rows(db, 'SELECT COUNT(*) observations,COUNT(DISTINCT camera_id || ":" || local_id) local_tracklets FROM observations')[0]
    if not {'observations.csv','global_tracks_2d.csv'}.intersection(d['files']):
        counts['observations'] = None
        counts['local_tracklets'] = d.get('known_local_count')
    counts['global_tracks'] = len(track_list(identifier))
    if counts['observations'] is None and 'global_ids.json' not in d['files']:
        counts['global_tracks'] = None
    counts['cameras'] = len(d['cameras'])
    durations = [c['duration'] + c.get('offset', 0) for c in d['cameras'] if c.get('duration') is not None]
    counts['duration'] = max(durations) if durations else None
    counts.update(rows(db,'SELECT MIN(timestamp) result_start,MAX(timestamp) result_end FROM observations')[0])
    fields = public(d)
    if d['mode'] == 'raw' and d['status'] != 'ready':
        fields['warnings'] = [warning for warning in fields['warnings']
                              if not warning.startswith(('Нет manifest.json', 'Нет CSV наблюдений'))]
    return fields | {'summary': counts, 'job': read_json(directory(identifier), 'job.json')}


@app.get('/api/datasets/{identifier}/cameras')
def cameras_api(identifier: str):
    return [{k: v for k,v in c.items() if k != 'path'} | {
        'video_url': f"/api/datasets/{identifier}/video/{c['camera_id']}" if c.get('path') else None}
        for c in dataset(identifier)['cameras']]


@app.get('/api/datasets/{identifier}/video/{camera_id}')
def video(identifier: str, camera_id: int):
    c = next((c for c in dataset(identifier)['cameras'] if c['camera_id'] == camera_id), None)
    if not c or not c.get('path') or not Path(c['path']).is_file():
        raise HTTPException(404, 'Видео камеры не подключено')
    return FileResponse(c['path'])


def review_target(identifier, global_id):
    from backend.export_review import review_window

    data = dataset(identifier)
    if data['status'] != 'ready':
        raise HTTPException(409, 'Сначала завершите трекинг')
    owned = directory(identifier)
    start, end, _ = review_window(owned, data, global_id)
    stamp = (owned/'index.sqlite').stat().st_mtime_ns
    variant = 'all' if global_id is None else str(global_id)
    return owned/f'review-v2-{stamp}-{variant}.mp4', start, end


def render_review(output, global_id):
    temporary = output.with_suffix('.partial.mp4')
    temporary.unlink(missing_ok=True)
    command = [sys.executable, '-m', 'backend.export_review', str(output.parent),
               str(temporary), '--cameras-only']
    if global_id is not None:
        command += ['--global-id', str(global_id)]
    env = dict(os.environ, PYTHONIOENCODING='utf-8', OPENCV_FFMPEG_THREADS='1')
    try:
        with output.with_suffix('.log').open('w+', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log)
            if result.returncode:
                log.seek(0)
                message = log.read().strip().splitlines()
                raise RuntimeError(message[-1] if message else 'Ошибка кодирования MP4')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


@app.get('/api/datasets/{identifier}/review')
@app.post('/api/datasets/{identifier}/review')
def review(identifier: str, request: Request, global_id: int | None = Query(None, ge=0)):
    output, start, end = review_target(identifier, global_id)
    query = '' if global_id is None else f'?global_id={global_id}'
    with lock:
        if output.is_file():
            return {'status': 'ready', 'start': start, 'end': end,
                    'url': f'/api/datasets/{identifier}/review/video{query}'}
        job = reviews.get(output)
        if request.method == 'POST' and (job is None or job.done()) and not output.is_file():
            job = reviews[output] = executor.submit(render_review, output, global_id)
        if job is None:
            return {'status': 'idle'}
        if job.done() and job.exception():
            return {'status': 'failed', 'error': str(job.exception())}
        return {'status': 'running' if job.running() or job.done() else 'queued'}


@app.get('/api/datasets/{identifier}/review/video')
def review_video(identifier: str, global_id: int | None = Query(None, ge=0)):
    output, _, _ = review_target(identifier, global_id)
    if not output.is_file():
        raise HTTPException(404, 'Мозаика ещё не готова')
    return FileResponse(output, media_type='video/mp4')


@app.get('/api/datasets/{identifier}/tracks')
def tracks(identifier: str, search: str = '', camera_id: int | None = None,
           start: float | None = None, end: float | None = None, sort: str = 'global_id',
           descending: bool = False, offset: int = Query(0,ge=0), limit: int = Query(25,ge=1,le=200)):
    found = [t for t in track_list(identifier) if search in str(t['global_id'])
             and (camera_id is None or camera_id in t['cameras'])
             and (start is None or t['end_time'] is not None and t['end_time'] >= start)
             and (end is None or t['start_time'] is not None and t['start_time'] <= end)]
    if sort not in {'global_id','start_time','end_time','duration','observations_count'}:
        raise HTTPException(422, 'Недопустимая сортировка')
    found.sort(key=lambda t: (t[sort] is None, t[sort] or 0), reverse=descending)
    return {'total': len(found), 'items': found[offset:offset+limit]}


@app.get('/api/datasets/{identifier}/tracks/{global_id}')
def track_detail(identifier: str, global_id: int):
    result = next((t for t in track_list(identifier) if t['global_id'] == global_id), None)
    if not result:
        raise HTTPException(404, 'Global ID не найден')
    folder = Path(dataset(identifier)['source'])
    local = read_json(folder, 'refined_tracklets.json', None)
    if local is None:
        local = read_json(folder, 'local_tracklets.json', [])
    keys = {(r['camera_id'], r['local_id']) for r in result['local_tracklets']}
    indices = {i for i,t in enumerate(local) if (t['camera_id'],t['local_id']) in keys}
    edges = [e for e in read_json(folder, 'association_edges.json', []) if e.get('i') in indices or e.get('j') in indices]
    timestamps = rows(directory(identifier)/'index.sqlite',
                      'SELECT DISTINCT timestamp FROM observations WHERE global_id=? '
                      'AND timestamp IS NOT NULL ORDER BY timestamp', [global_id])
    episodes = []
    for sample in timestamps:
        t = sample['timestamp']
        if not episodes or t - episodes[-1]['end_time'] > 10:
            episodes.append({'global_track_id': f'{global_id}.{len(episodes)+1}',
                             'start_time': t, 'end_time': t})
        else:
            episodes[-1]['end_time'] = t
    identity = read_json(folder, 'identity_state.json', {})
    identity_state = identity.get('states', {}).get(str(global_id))
    identity_events = [e for e in identity.get('events', [])
                       if global_id in [e.get('global_id'), e.get('from'), e.get('to')]
                       or global_id in (e['from'] if isinstance(e.get('from'), list) else [])
                       or global_id in (e['to'] if isinstance(e.get('to'), list) else [])]
    return result | {'episodes': episodes, 'association_edges': edges[:200],
                     'identity_state': identity_state, 'identity_events': identity_events,
                     'association_edges_total': len(edges)}


@app.get('/api/datasets/{identifier}/observations')
def observations(identifier: str, camera_id: int | None = None, global_id: int | None = None,
                 start: float | None = None, end: float | None = None, frame_id: int | None = None,
                 offset: int = Query(0,ge=0), limit: int = Query(1000,ge=1,le=10000)):
    dataset(identifier)
    clauses, args = [], []
    for column, operator, value in [('camera_id','=',camera_id),('global_id','=',global_id),
        ('timestamp','>=',start),('timestamp','<=',end),('frame_id','=',frame_id)]:
        if value is not None:
            clauses.append(f'{column}{operator}?'); args.append(value)
    where = ' WHERE '+' AND '.join(clauses) if clauses else ''
    db = directory(identifier)/'index.sqlite'
    count = rows(db,'SELECT COUNT(*) n FROM observations'+where,args)[0]['n']
    result = rows(db,'SELECT * FROM observations'+where+' ORDER BY frame_id,camera_id,local_id LIMIT ? OFFSET ?',[*args,limit,offset])
    return {'total': count, 'items': [observation_dto(r) for r in result]}


@app.get('/api/datasets/{identifier}/performance')
def performance(identifier: str):
    folder = Path(dataset(identifier)['source'])
    entries = read_json(folder, 'performance.json', [])
    seconds = sum(float(r.get('seconds') or 0) for r in entries)
    frames = sum(int(r.get('processed_frames') or 0) for r in entries)
    peaks = [r['peak_cuda_tensor_gib'] for r in entries if r.get('peak_cuda_tensor_gib') is not None]
    return {'cameras': entries, 'fps': frames/seconds if seconds else None,
            'processing_time': seconds or None, 'peak_cuda_memory': max(peaks) if peaks else None,
            'metrics': read_json(folder,'metrics.json')}


@app.get('/api/datasets/{identifier}/manifest')
def manifest_api(identifier: str):
    d = dataset(identifier)
    return {'manifest': d['manifest'], 'config': d['config'], 'warnings': d['warnings'], 'files': d['files']}


@app.get('/api/datasets/{identifier}/analytics')
def analytics(identifier: str):
    dataset(identifier)
    db = directory(identifier)/'index.sqlite'
    ts = track_list(identifier)
    span=rows(db,'SELECT MAX(timestamp)-MIN(timestamp) span FROM observations')[0]['span'] or 0
    import math
    bin_seconds=max(5,math.ceil(span/60/5)*5)
    bounds=[0,5,15,30,60,float('inf')]
    return {'by_camera': rows(db,'SELECT camera_id,COUNT(*) observations,COUNT(DISTINCT global_id) tracks FROM observations GROUP BY camera_id'),
            'by_time': rows(db,'SELECT CAST(timestamp/? AS INT)*? time,COUNT(*) observations FROM observations WHERE timestamp IS NOT NULL GROUP BY time ORDER BY time',[bin_seconds,bin_seconds]),
            'bin_seconds':bin_seconds,
            'duration_histogram':[{'label':f'{low}–{high if math.isfinite(high) else "∞"} с',
                'value':sum(t['duration'] is not None and low<=t['duration']<high for t in ts)} for low,high in zip(bounds,bounds[1:])],
            'mean_cameras': sum(len(t['cameras']) for t in ts)/len(ts) if ts else None}


class ProcessOptions(BaseModel):
    max_frames: int = Field(default=0, ge=0, le=1000000)
    max_seconds: float | None = Field(default=None,gt=0,le=86400,allow_inf_nan=False)
    device: str = Field(default='cuda:0', pattern=r'^(cpu|cuda:[0-9]+)$')
    matcher: Literal['v2','graph'] = 'v2'


def run_job(identifier, options):
    owned = directory(identifier)
    job = read_json(owned,'job.json')
    d = dataset(identifier)
    out = owned/'result'
    try:
        spec = {'scene': d['source'], 'output': str(out), 'cameras': d['cameras'], 'options': options}
        save(owned/'worker.json',spec)
        job.update(status='running', started_at=now(), phase='tracking')
        save(owned/'job.json',job)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', OPENCV_FFMPEG_THREADS='1', HF_HUB_OFFLINE='1')
        with (owned/'job.log').open('w',encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable,'-u','-m','backend.worker',str(owned/'worker.json')],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                log.write(line); log.flush()
                if line.startswith('STAGE '):
                    job['phase'] = line.strip().split(' ',1)[1]; save(owned/'job.json',job)
            if process.wait() != 0:
                raise RuntimeError('Пайплайн завершился с ошибкой. Подробности в журнале; проверьте модели и доступную память.')
        metadata = build_index(out, owned/'index.sqlite',d['cameras'])
        d.update(metadata, source=str(out), status='ready')
        save(owned/'dataset.json',d)
        job.update(status='completed',phase='completed')
    except Exception as exc:
        job.update(status='failed',error=str(exc))
    job['finished_at'] = now()
    save(owned/'job.json',job)


@app.post('/api/datasets/{identifier}/process')
def process_dataset(identifier: str, options: ProcessOptions):
    with lock:
        d = dataset(identifier)
        job = read_json(directory(identifier),'job.json',{})
        if d['mode'] != 'raw' or d['status'] == 'ready':
            raise HTTPException(409,'Для обработки загрузите новый набор исходных видео')
        if job.get('status') in {'queued','running'}:
            raise HTTPException(409,'Обработка уже запущена')
        job = {'id':identifier,'dataset_id':identifier,'status':'queued','progress':None,'phase':'queued',
               'created_at':now(),'started_at':None,'finished_at':None,'options':options.model_dump()}
        save(directory(identifier)/'job.json',job)
        executor.submit(run_job,identifier,options.model_dump())
        return job


@app.get('/api/jobs/{identifier}')
def get_job(identifier: str):
    path = directory(identifier)
    job = read_json(path,'job.json')
    if not job:
        raise HTTPException(404,'Задача не найдена')
    log = path/'job.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-12000))
            job['log'] = stream.read().decode('utf-8',errors='replace')
    return job


@app.post('/api/datasets/{identifier}/floorplan')
async def floor_upload(identifier: str, file: UploadFile = File(...)):
    dataset(identifier)
    content = await file.read(20*1024**2+1)
    if len(content)>20*1024**2 or file.filename.lower().endswith(('.png','.jpg','.jpeg')) is False:
        raise HTTPException(422,'План: PNG/JPG до 20 МБ')
    image = cv2.imdecode(np.frombuffer(content,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422,'Повреждённое изображение')
    (directory(identifier)/'floorplan.png').write_bytes(cv2.imencode('.png',image)[1].tobytes())
    save(directory(identifier)/'floor_calibration.json',{})
    return {'status':'ok'}


@app.get('/api/datasets/{identifier}/floorplan/image')
def floor_image(identifier: str):
    path = directory(identifier)/'floorplan.png'
    if not path.exists(): raise HTTPException(404,'План не загружен')
    return FileResponse(path)


class Marker(BaseModel):
    camera_id: int
    x: float = Field(ge=0,le=1,allow_inf_nan=False)
    y: float = Field(ge=0,le=1,allow_inf_nan=False)


@app.get('/api/datasets/{identifier}/floorplan')
def floor_info(identifier: str):
    dataset(identifier)
    return {'available': (directory(identifier)/'floorplan.png').exists(),
            'markers': read_json(directory(identifier),'markers.json',[]),
            'calibrations':read_json(directory(identifier),'floor_calibration.json',{}),
            'calibrated':bool(read_json(directory(identifier),'floor_calibration.json',{}))}


@app.put('/api/datasets/{identifier}/floorplan')
def floor_markers(identifier: str, markers: list[Marker]):
    cams = {c['camera_id'] for c in dataset(identifier)['cameras']}
    if any(m.camera_id not in cams for m in markers) or len({m.camera_id for m in markers}) != len(markers):
        raise HTTPException(422,'Укажите уникальные существующие камеры')
    save(directory(identifier)/'markers.json',[m.model_dump() for m in markers])
    return {'status':'ok'}


class FloorPoint(BaseModel):
    video: tuple[float,float]
    plan: tuple[float,float]


@app.put('/api/datasets/{identifier}/floorplan/calibration/{camera_id}')
def floor_calibration(identifier: str, camera_id: int, points: list[FloorPoint]):
    cameras={c['camera_id']:c for c in dataset(identifier)['cameras']}
    if camera_id not in cameras or not cameras[camera_id].get('width'):
        raise HTTPException(422,'Для привязки подключите видео камеры')
    if not (directory(identifier)/'floorplan.png').is_file():
        raise HTTPException(422,'Сначала загрузите план')
    calibration=fit_floor([p.model_dump() for p in points])
    saved=read_json(directory(identifier),'floor_calibration.json',{})
    saved[str(camera_id)]=calibration
    save(directory(identifier)/'floor_calibration.json',saved)
    return calibration


@app.get('/api/datasets/{identifier}/frame/{camera_id}')
def camera_frame(identifier: str,camera_id: int,time: float=Query(0,ge=0,allow_inf_nan=False)):
    camera=next((c for c in dataset(identifier)['cameras'] if c['camera_id']==camera_id),None)
    if not camera or not camera.get('path'): raise HTTPException(404,'Видео не подключено')
    cap=cv2.VideoCapture(camera['path'])
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC,time*1000);ok,frame=cap.read()
        if not ok: raise HTTPException(422,'Кадр за пределами видео или не декодируется')
        return Response(cv2.imencode('.jpg',frame)[1].tobytes(),media_type='image/jpeg')
    finally: cap.release()


@app.get('/api/datasets/{identifier}/floorplan/trajectories')
def floor_trajectories(identifier: str,start: float=Query(0,allow_inf_nan=False),
                       end: float=Query(30,allow_inf_nan=False),global_id: int|None=None):
    if end<start or end-start>120: raise HTTPException(422,'Выберите интервал не больше 120 секунд')
    d=dataset(identifier);cal=read_json(directory(identifier),'floor_calibration.json',{})
    cameras={c['camera_id']:c for c in d['cameras']}
    clauses='timestamp>=? AND timestamp<=?';args=[start,end]
    if global_id is not None: clauses+=' AND global_id=?';args.append(global_id)
    data=rows(directory(identifier)/'index.sqlite',f'SELECT * FROM observations WHERE {clauses} ORDER BY timestamp,camera_id,local_id LIMIT 10001',args)
    result=project_floor(data[:10000],cameras,cal)
    return {'points':result,'truncated':len(data)>10000,'calibrated_cameras':[int(k) for k in cal],
            'coordinate_system':'normalized_plan_pixels','kind':'manual_estimate',
            'warning':'Приблизительная проекция нижней середины bbox на пол; не измеренные координаты. Межкамерные пробелы не соединяются.'}


@app.get('/{path:path}', include_in_schema=False)
def frontend(path: str):
    if path.startswith('api/'):
        raise HTTPException(404,'API route not found')
    distribution=ROOT/'frontend/dist'
    candidate=(distribution/path).resolve()
    if not candidate.is_relative_to(distribution):
        raise HTTPException(404,'Файл не найден')
    if candidate.is_file():
        return FileResponse(candidate)
    if (distribution/'index.html').is_file():
        return FileResponse(distribution/'index.html')
    return JSONResponse({'detail':'Frontend не собран. Выполните npm run build в frontend или откройте dev-сервер на порту 5173.'},status_code=503)
