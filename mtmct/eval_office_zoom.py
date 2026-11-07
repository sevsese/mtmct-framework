"""Score the fixed manual far-field review against the two pre-tracker detector outputs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .office_pass import iou


def evaluate(run, labels, rois):
    manifest=json.loads((run/"manifest.json").read_text())["tracking_config"]
    report={}
    for cam,roi in rois.items():
        camera=cam+1
        fps=manifest["source_fps"][str(cam)]
        stride=manifest["stride"]
        by_frame=defaultdict(list)
        x,y,r,b=np.rint(np.array(roi)*[2592,1944,2592,1944])
        with (run/f"zoom_camera_{camera}.csv").open() as f:
            for row in csv.DictReader(f):
                box=np.array([float(row[k]) for k in ("x1","y1","x2","y2")])
                center=(box[:2]+box[2:])/2
                if x<=center[0]<r and y<=center[1]<b:
                    box=np.clip((box-[x,y,x,y])*240/[r-x,b-y,r-x,b-y],0,240)
                    by_frame[int(row["frame"]),row["pass"]].append(box)
        for mode in ("full","fused"):
            tp=fp=fn=0
            details={}
            for sec,truth in labels[str(camera)].items():
                fid=round(int(sec)*fps/stride)*stride
                boxes=by_frame[fid,mode]
                pairs=sorted(((iou(np.array(a),b),i,j) for i,a in enumerate(truth)
                              for j,b in enumerate(boxes)),reverse=True)
                gt=set()
                pred=set()
                for score,i,j in pairs:
                    if score>=.5 and i not in gt and j not in pred:
                        gt.add(i)
                        pred.add(j)
                tp+=len(gt)
                fp+=len(boxes)-len(pred)
                fn+=len(truth)-len(gt)
                details[sec]=sorted(gt)
            report[f"cam{camera}_{mode}"]=dict(tp=tp,fp=fp,fn=fn,frames=len(details),
                                               recall=tp/(tp+fn),matched_indices=details)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    args=p.parse_args()
    config=json.loads((args.run/"manifest.json").read_text())["tracking_config"]
    rois={int(c):r for c,r in config["zoom_rois"].items()}
    labels=json.loads((Path(__file__).resolve().parents[1]/"configs/office_far_review.json").read_text())
    report=evaluate(args.run,labels,rois)
    (args.run/"far_review.json").write_text(json.dumps(report,indent=2))
    print({k:{f:v[f] for f in ("tp","fp","fn","frames","recall")} for k,v in report.items()})
