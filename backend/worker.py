"""Isolated GPU worker calling the unchanged MTMCT functions."""
import json
import sys
import csv
import subprocess
import ijson
from pathlib import Path
from mtmct.tracking import run as track
from mtmct.link import run as link
from mtmct.association import Config
from mtmct.cache import write_config
from mtmct.weights import MODELS
from backend.adapter import ROOT


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
        camera=next(c for c in cameras if c['camera_id']==int(sys.argv[2]))
        track(Path(spec['scene']), out/f"camera_{camera['camera_id']}", str(detector), 'osnet', str(weights), options['device'],
              options['max_frames'], 1, 5, 1920, .10, camera_sources={camera['camera_id']: Path(camera['path'])})
        return
    print('STAGE tracking', flush=True)
    parts=[]
    for camera in cameras:
        subprocess.run([sys.executable,'-u','-m','backend.worker',sys.argv[1],str(camera['camera_id'])],check=True)
        parts.append(out/f"camera_{camera['camera_id']}")
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
    manifest['web_worker']='isolated-camera-processes-v1'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('STAGE association', flush=True)
    link(out, Config(appearance_weight=.95, spatial_weight=0, temporal_weight=.05,
                    max_cost=.125, min_similarity=.80, use_geometry=False,
                    camera_offsets_s={c['camera_id']: c.get('offset', 0) for c in cameras}))
    print('STAGE completed', flush=True)


if __name__ == '__main__':
    main()
