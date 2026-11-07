"""Review one annotated office passage and test longitudinal projection on paired views."""
import csv
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np

from .office_walls import transform


# Zero-based camera and local ID in the reference v10 run, confirmed by the reviewer.
REFERENCE = ((2,3),(4,2))


def tracks(path):
    result=defaultdict(dict)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key=int(row["camera_id"]), int(row["local_id"])
            result[key][int(row["frame_id"])]=np.array([float(row[k]) for k in ("x1","y1","x2","y2")])
    return result


def iou(a,b):
    area=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return area/max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-area,1e-9)


def apply_pass(reference, current, mapping):
    selected=[]
    for cam,lid in REFERENCE:
        evidence=reference[cam,lid]
        found=[]
        for key,boxes in current.items():
            if key not in mapping or key[0]!=cam: continue
            shared=boxes.keys() & evidence.keys()
            hits=sum(iou(boxes[f],evidence[f])>=.5 for f in shared)
            # Accept an extension across a short detector dropout, but never
            # propagate this review to later passages sharing an automatic ID.
            if (hits>=4 and hits>=.75*len(shared) and
                min(boxes)>=min(evidence)-50 and max(boxes)<=max(evidence)+50):
                found.append(key)
        if not found and cam == 4:
            raise ValueError(f"Reviewed passage missing from camera {cam+1}")
        selected.extend(found)
    gid=max(mapping.values())+1
    for key in selected:
        mapping[key]=gid
    return {"reference_v10":REFERENCE,"current_local_keys":selected,"reviewed_global_id":gid}


def apply_frame_review(path, observations, mapping, passage):
    """Replay human labels only for segments matching all three annotated boxes."""
    applied = {}
    for label, segments in json.loads(path.read_text()).items():
        keys = []
        for segment in segments:
            for key, boxes in observations.items():
                if key[0] != segment["camera"] or (min(boxes), max(boxes)) != (segment["first"], segment["last"]):
                    continue
                if all(int(f) in boxes and iou(boxes[int(f)], np.array(box)) > .85
                       for f, box in segment["samples"].items()):
                    keys.append(key)
        if keys:
            gid = passage["reviewed_global_id"] if label == "pass" else max(mapping.values()) + 1
            mapping.update(dict.fromkeys(keys, gid))
            applied[label] = {"id": gid, "segments": keys}
    passage["frame_review"] = applied


def fit_projection(reference, base, fps, offsets):
    """Shift only uncertain far-wall stations. Near colour landmarks stay fixed."""
    pts=[]
    for cam,lid in REFERENCE:
        row=base["cameras"][str(cam+1)]
        hull=cv2.convexHull(np.float32(row["image_points"]))
        visible={}
        for fid,b in reference[cam,lid].items():
            p=(float((b[0]+b[2])/2),float(b[3]))
            if cv2.pointPolygonTest(hull,p,False)>=0:
                visible[round(1000*(fid/fps[cam]+offsets[cam]))]=p
        pts.append(visible)
    times=sorted(pts[0].keys() & pts[1].keys())
    if len(times)<12: raise ValueError("Too few synchronized foot observations")
    # Hold out the middle third of the passage, including its neighbouring frames.
    hold=np.zeros(len(times),bool)
    hold[len(times)//3:2*len(times)//3]=True
    def project(side, shift):
        cam=REFERENCE[side][0]+1
        row=base["cameras"][str(cam)]
        targets=np.float32(row["plan_points"])
        targets[[1,2] if cam==3 else [2],1]+=shift
        H=cv2.getPerspectiveTransform(np.float32(row["image_points"]),targets)
        return transform([pts[side][t] for t in times],H),targets
    variants=[{s:project(k,s) for s in range(-120,121,10)} for k in range(2)]
    def residual(a,b):
        return np.linalg.norm(variants[0][a][0]-variants[1][b][0],axis=1)
    a,b=min(((a,b) for a in variants[0] for b in variants[1]),
            key=lambda ab:np.median(residual(*ab)[~hold])+.05*sum(map(abs,ab)))
    old=residual(0,0)
    new=residual(a,b)
    accepted=bool(np.median(new[hold])<np.median(old[hold]))
    config=deepcopy(base)
    if accepted:
        for i,(cam,s) in enumerate(((3,a),(5,b))):
            config["cameras"][str(cam)]["plan_points"]=variants[i][s][1].tolist()
    report={"times_ms":times,"holdout_ms":[t for t,h in zip(times,hold) if h],
            "shift_cam3_cam5_px":[a,b],"accepted":accepted,
            "median_holdout_before_after_px":[float(np.median(d[hold])) for d in (old,new)],
            "median_fit_before_after_px":[float(np.median(d[~hold])) for d in (old,new)]}
    return config,report


def remove_review_duplicates(path, gid, reference, preferred=()):
    """Keep one box for a reviewed person seen twice on the same camera/frame."""
    with path.open(newline="") as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames
        rows=list(reader)
    best={}
    rejected=set()
    for index,row in enumerate(rows):
        if int(row["global_id"])!=gid: continue
        cam,fid=int(row["camera_id"]),int(row["frame_id"])
        ref_key=next((k for k in REFERENCE if k[0]==cam),None)
        target=reference.get(ref_key,{}).get(fid)
        box=np.array([float(row[k]) for k in ("x1","y1","x2","y2")])
        score=iou(box,target) if target is not None else float(row["confidence"])
        if (cam,int(row["local_id"])) in preferred:
            score += 2
        key=cam,fid
        if key in best:
            if score<=best[key][0]:
                rejected.add(index)
                continue
            rejected.add(best[key][1])
        best[key]=score,index
    if rejected:
        with path.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for i,row in enumerate(rows) if i not in rejected)
    return len(rejected)


def draw_pass(directory, plan, gid):
    """Fuse simultaneous positions for this reviewed identity on the common clock."""
    by_time=defaultdict(list)
    with (directory/"wall_tracks.csv").open() as f:
        for row in csv.DictReader(f):
            if int(row["global_id"])==gid and row["status"]=="approximate":
                by_time[float(row["time_s"])].append(row)
    canvas=plan.copy()
    previous=None
    with (directory/"pass_path.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["global_id","time_s","cameras","x_plan_px","y_plan_px","spread_px","starts_segment"])
        for t,rows in sorted(by_time.items()):
            pts=np.array([[float(r["x_plan_px"]),float(r["y_plan_px"])] for r in rows])
            xy=pts.mean(axis=0)
            spread=float(np.linalg.norm(pts[-1]-pts[0])) if len(pts)>1 else None
            cameras="+".join(r["camera_plan"] for r in rows)
            # A change in contributing views can jump even for a stationary
            # person. Avoid drawing that calibration discontinuity as a walk.
            start=previous is None or t-previous[0]>.2 or cameras!=previous[2]
            writer.writerow([gid,t,cameras,*xy.round(2),
                             "" if spread is None else round(spread,2),int(start)])
            p=tuple(np.rint(xy).astype(int))
            if not start:
                cv2.line(canvas,previous[1],p,(140,45,145),3,cv2.LINE_AA)
            cv2.circle(canvas,p,2,(140,45,145),-1)
            if previous is None:
                cv2.putText(canvas,f"START {t:.1f}s",(p[0]+10,p[1]),0,.56,(92,30,95),2)
            previous=t,p,cameras
    if previous:
        t,p,_=previous
        cv2.putText(canvas,f"END {t:.1f}s",(p[0]+10,p[1]),0,.56,(92,30,95),2)
    cv2.putText(canvas,f"G{gid} | reviewed corridor passage", (15,45),0,1,(30,30,30),2)
    cv2.imencode(".png",canvas)[1].tofile(directory/"pass_path.png")
