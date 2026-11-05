"""Development-only 2D ID association diagnostic, NOT official 3D HOTA.

We evaluate only explicitly chosen (frame, camera) samples, including samples
where our detector returned no boxes. This avoids inflated recall on
empty-output frames. Official Track-1 evaluation uses 3D HOTA instead.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import camera_number
from .gt import value


def iou(a,b):
    ax1,ay1,ax2,ay2=map(float,a)
    bx1,by1,bx2,by2=map(float,b)
    w=max(0.,min(ax2,bx2)-max(ax1,bx1))
    h=max(0.,min(ay2,by2)-max(ay1,by1))
    intersection=w*h
    area=max(0.,ax2-ax1)*max(0.,ay2-ay1)+max(0.,bx2-bx1)*max(0.,by2-by1)-intersection
    return intersection/area if area>0 else 0.


def eval_2d(pred_path:Path, gt_path:Path, cameras:list[int],
            max_frames:int, stride:int=3, threshold:float=0.5):
    import ijson
    if max_frames<=0 or stride<=0:
        raise ValueError("Use finite positive --max-frames and --stride for the sampled diagnostic")
    frame_set=set(range(0,max_frames,stride))
    key_set={(f,c) for f in frame_set for c in cameras}
    predictions=defaultdict(list)
    pred_total=0
    with open(pred_path,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key=(int(r["frame_id"]),int(r["camera_id"]))
            if key not in key_set:continue
            predictions[key].append((int(r["global_id"]),[
                float(r[k]) for k in ("x1","y1","x2","y2")]))
            pred_total+=1
    gt_total=0
    matches=Counter()
    with open(gt_path,"rb") as f:
        for frame,objects in ijson.kvitems(f,""):
            frame=int(frame)
            if frame>=max_frames:break
            if frame not in frame_set:continue
            cam_to_gt=defaultdict(list)
            for ob in objects:
                if str(value(ob, "object_type", "")).lower() not in ("person", "people"):
                    continue
                gt_id=int(value(ob, "object_id"))
                for cam,bbox in value(ob, "2d_bounding_box_visible",{}).items():
                    key=(frame,camera_number(cam))
                    if key in key_set:
                        cam_to_gt[key].append((gt_id,bbox))
            for camera in cameras:
                key=(frame,camera)
                gt=cam_to_gt[key]
                pr=predictions[key]
                gt_total+=len(gt)
                if not gt or not pr:continue
                m=np.asarray([[iou(g[1],p[1]) for p in pr] for g in gt])
                # Maximize valid match count first, IoU only breaks ties.
                # Matching raw IoU then filtering can discard valid pairs.
                valid = m >= threshold
                score = valid * (min(m.shape) + 1 + m)
                row,col=linear_sum_assignment(-score)
                for a,b in zip(row,col):
                    if m[a,b]>=threshold:
                        matches[(gt[a][0],pr[b][0])]+=1
    gt_ids=sorted({g for g,_ in matches})
    pred_ids=sorted({p for _,p in matches})
    if gt_ids and pred_ids:
        mat=np.zeros((len(gt_ids),len(pred_ids)),dtype=float)
        g_index={g:i for i,g in enumerate(gt_ids)}
        p_index={p:i for i,p in enumerate(pred_ids)}
        for (g,p),count in matches.items():
            mat[g_index[g],p_index[p]]=count
        ri,ci=linear_sum_assignment(-mat)
        idtp=int(mat[ri,ci].sum())
    else:idtp=0
    idfp=max(pred_total-idtp,0)
    idfn=max(gt_total-idtp,0)
    denom=2*idtp+idfp+idfn
    det_tp=sum(matches.values())
    det_fp=max(pred_total-det_tp,0)
    det_fn=max(gt_total-det_tp,0)
    det_denom=2*det_tp+det_fp+det_fn
    return {
        "diagnostic_IDF1":round(2*idtp/denom,4) if denom else 0.,
        "IDTP":idtp,"IDFP":idfp,"IDFN":idfn,
        "detection_F1":round(2*det_tp/det_denom,4) if det_denom else 0.,
        "detection_precision":round(det_tp/pred_total,4) if pred_total else 0.,
        "detection_recall":round(det_tp/gt_total,4) if gt_total else 0.,
        "gt_count":gt_total,"pred_count":pred_total,
        "note":"Approximate 2D sample-level IDF1; NOT official 3D HOTA"
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pred",type=Path,required=True)
    p.add_argument("--gt",type=Path,required=True)
    p.add_argument("--cameras",type=int,nargs="+",default=[0,1])
    p.add_argument("--max-frames",type=int,default=300)
    p.add_argument("--stride",type=int,default=3)
    a=p.parse_args()
    print(eval_2d(a.pred,a.gt,a.cameras,a.max_frames,a.stride))

if __name__=="__main__": main()
