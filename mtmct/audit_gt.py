"""Optional check: do selected cameras actually see some common persons?"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from .geometry import camera_number
from .gt import value


def overlapping_ids(gt: Path, cameras: list[int], max_frames: int = 300) -> dict:
    import ijson
    cam_people = defaultdict(set)
    with open(gt, 'rb') as f:
        for frame, objects in ijson.kvitems(f, ''):
            if int(frame) >= max_frames:
                break
            for obj in objects:
                if str(value(obj, 'object_type', '')).lower() != 'person':
                    continue
                ident = value(obj, 'object_id')
                for cam in value(obj, '2d_bounding_box_visible', {}):
                    cid = camera_number(cam)
                    if cid in cameras:
                        cam_people[cid].add(ident)
    return {c: cam_people[c] for c in cameras}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gt', type=Path, required=True)
    p.add_argument('--cameras', type=int, nargs='+', default=[0, 1])
    p.add_argument('--max-frames', type=int, default=300)
    args = p.parse_args()
    people = overlapping_ids(args.gt, args.cameras, args.max_frames)
    for cam in args.cameras:
        print(f'Camera {cam}: {len(people[cam])} unique GT people in selected frames')
    for i, a in enumerate(args.cameras):
        for b in args.cameras[i + 1:]:
            common = people[a] & people[b]
            print(f'Cameras {a},{b}: {len(common)} common GT people (includes sequential sightings)')
    print('GT used for diagnostics ONLY. Do not feed the GT IDs to the matcher.')

if __name__ == '__main__':
    main()
