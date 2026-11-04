"""Check assumed cameraMatrix ground-floor convention against TRAIN ground truth.

This diagnostic READS GT but never uses it in test-time predictions.
Bottom-of-2D-box is approximate foot point, so errors may be non-zero.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
from pathlib import Path
import numpy as np
from .geometry import calibration_maps, camera_number, project_pixel
from .gt import value


def check(scene_dir:Path, max_frames:int=100, max_samples:int=1000):
    import ijson
    maps=calibration_maps(scene_dir/"calibration.json")
    errors=defaultdict(list)
    with open(scene_dir/"ground_truth.json","rb") as f:
        for frame,objects in ijson.kvitems(f,""):
            if int(frame)>=max_frames:break
            for ob in objects:
                if str(value(ob, "object_type", "")).lower()!="person":continue
                pos=value(ob, "3d_location")
                if not pos or len(pos)<3:continue
                gt_xy=np.asarray([float(pos[0]),float(pos[1])])
                for cam,box in value(ob, "2d_bounding_box_visible",{}).items():
                    cid=camera_number(cam)
                    if len(errors[cid])>=max_samples:continue
                    if cid not in maps or maps[cid] is None:continue
                    x1,y1,x2,y2=map(float,box)
                    if x2-x1<20 or y2-y1<50:continue
                    est=project_pixel(maps[cid],(x1+x2)/2.,y2)
                    if est is None:continue
                    errors[cid].append(float(np.linalg.norm(gt_xy-np.asarray(est))))
    if not errors:
        print("No comparable GT points found: check camera ids and data format.")
        return
    print("Camera | n | median abs XY error in m | p90 (m)")
    for cam,arr in sorted(errors.items()):
        if not arr:continue
        print(f"{cam:6d} | {len(arr):4d} | {np.median(arr):8.2f} | {np.percentile(arr,90):8.2f}")
    print("Compare to typical human scale; if errors are large, disable spatial weighting until calibration is fixed.")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scene-dir",type=Path,required=True)
    p.add_argument("--max-frames",type=int,default=100)
    a=p.parse_args()
    check(a.scene_dir,a.max_frames)

if __name__=="__main__":main()
