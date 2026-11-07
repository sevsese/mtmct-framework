"""Isolated GPU worker calling the unchanged MTMCT functions."""
import json
import sys
import csv
import subprocess
import ijson
from pathlib import Path
from mtmct.cache import write_config
from mtmct.weights import MODELS
import math
ROOT=Path(__file__).resolve().parents[1]


def frame_limit(options, camera):
    limit=options['max_frames']
    seconds=options.get('camera_max_seconds',{}).get(str(camera.get('camera_id')),options.get('max_seconds'))
    if seconds is not None:
        timed=math.ceil(seconds*camera['fps'])
        limit=min(limit,timed) if limit else timed
    return limit


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    out = Path(spec['output'])
    cameras = spec['cameras']
    options = spec['options']
    detector = ROOT/'yolo26l.pt'
    weights = ROOT/'weights'/MODELS['osnet_x1_0']
    for path in (detector, weights):
        if not path.is_file():
            raise FileNotFoundError(f'Модель отсутствует: {path}')
    write_config(out, options | {'model': str(detector), 'reid_model': 'osnet_x1_0',
        'matcher': 'baseline', 'memory': False, 'geometry': False,
        'camera_offset': [f"{c['camera_id']}={c.get('offset', 0)}" for c in cameras]})
    if len(sys.argv)>2:
        import cv2
        import torch
        from mtmct.tracking import run as track
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        camera=next(c for c in cameras if c['camera_id']==int(sys.argv[2]))
        track(Path(spec['scene']), out/f"camera_{camera['camera_id']}", str(detector), 'osnet', str(weights), options['device'],
              frame_limit(options,camera), 1, 5, 1920, .10, camera_sources={camera['camera_id']: Path(camera['path'])})
        return
    print('STAGE tracking', flush=True)
    parts=[]
    for camera in cameras:
        part=out/f"camera_{camera['camera_id']}"
        reusable=False
        if options.get('resume') and all((part/n).is_file() for n in ['manifest.json','performance.json','observations.csv','local_tracklets.json']):
            from mtmct.cache import validate_cache
            saved=validate_cache(part)
            cfg=saved['tracking_config']
            expected={'max_frames':frame_limit(options,camera),'stride':1,'sample_every':5,'imgsz':1920,'conf':.10}
            reusable=all(cfg.get(k)==v for k,v in expected.items()) and cfg.get('camera_sources')=={str(camera['camera_id']):camera['path']}
        if reusable:print(f"Reusing completed camera {camera['camera_id']}",flush=True)
        else:subprocess.run([sys.executable,'-u','-m','backend.worker',sys.argv[1],str(camera['camera_id'])],check=True)
        parts.append(part)
    # Merge the existing per-camera artifacts without altering IDs or embeddings.
    for name in ('local_tracklets.json','performance.json'):
        with (out/name).open('w',encoding='utf-8') as destination:
            destination.write('[');first=True
            for part in parts:
                with (part/name).open('rb') as stream:
                    for item in ijson.items(stream,'item',use_float=True):
                        if not first: destination.write(',')
                        json.dump(item,destination);first=False
            destination.write(']')
    with (out/'observations.csv').open('w',newline='',encoding='utf-8') as destination:
        writer=None
        for part in parts:
            with (part/'observations.csv').open(newline='',encoding='utf-8') as stream:
                reader=csv.DictReader(stream)
                if writer is None:
                    writer=csv.DictWriter(destination,reader.fieldnames);writer.writeheader()
                writer.writerows(reader)
    manifests=[json.loads((p/'manifest.json').read_text()) for p in parts]
    manifest=manifests[0]
    manifest['cameras']=[c['camera_id'] for c in cameras]
    manifest['inputs']=list({item['path']:item for m in manifests for item in m['inputs']}.values())
    manifest['tracking_config']['camera_sources']={str(c['camera_id']):c['path'] for c in cameras}
    manifest['tracking_config']['per_camera_max_frames']={str(c['camera_id']):frame_limit(options,c) for c in cameras}
    manifest['web_worker']='isolated-camera-processes-v1'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('STAGE association', flush=True)
    from mtmct.link import run as link
    from mtmct.association import Config
    link(out, Config(appearance_weight=.95, spatial_weight=0, temporal_weight=.05,
                    max_cost=.125, min_similarity=.80, use_geometry=False,
                    camera_offsets_s={c['camera_id']: c.get('offset', 0) for c in cameras}))
    print('STAGE completed', flush=True)


if __name__ == '__main__':
    main()
