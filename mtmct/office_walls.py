"""Rebuild an office floor map from the annotated contact sheet and cached tracks."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .floor_map import _read_image


# Pixel pairs on the 2048 x 1024 working copy: (camera panel, plan inset).
# For cams 3 and 5 the stroke ends denote approximate longitudinal stations.
# Cam 4 uses the corners of the purple passage. Cam 1/2 each have only one line.
PAIRS = {
    3: ([(1539,193),(1596,50),(1634,43),(1725,303)],
        [(1720,688),(1720,905),(1687,900),(1686,686)]),
    4: ([(120,858),(266,592),(356,595),(353,884)],
        [(1689,833),(1543,836),(1541,808),(1688,805)]),
    5: ([(964,980),(974,795),(1042,544),(1310,894)],
        [(1685,970),(1685,905),(1720,688),(1720,905)]),
}


def transform(points, matrix):
    return cv2.perspectiveTransform(np.array(points, np.float64).reshape(-1,1,2),
                                    matrix).reshape(-1,2)


def register(reference, panel, origin):
    """Register two copies of the SAME image (not camera scenery to a drawing)."""
    h, w = reference.shape[:2]
    small = cv2.resize(reference, (760, round(760*h/w)), interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=4500)
    a, da = sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
    b, db = sift.detectAndCompute(cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY), None)
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p)==2 and p[0].distance < .72*p[1].distance]
    if len(good)<12:
        raise ValueError("Too few matching pixels to locate panel")
    src = np.float32([a[m.queryIdx].pt for m in good])
    dst = np.float32([b[m.trainIdx].pt for m in good])+origin
    affine, mask = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=2)
    if affine is None or mask.sum()<12:
        raise ValueError("Panel registration failed")
    M = np.vstack([affine,[0,0,1]]) @ np.diag([small.shape[1]/w,small.shape[0]/h,1])
    return M, int(mask.sum())


def calibrate(sheet, plan, videos):
    sheet = cv2.resize(_read_image(sheet),(2048,1024))
    to_inset, votes = register(plan,sheet[512:,1366:],(1366,512))
    result = {"plan_size":list(plan.shape[1::-1]),"plan_registration_inliers":votes,"cameras":{},
              "walls_only":{str(cam):transform(ends,np.linalg.inv(to_inset)).round(2).tolist()
                            for cam,ends in [(1,[(1537,684),(1684,687)]),
                                             (2,[(1559,623),(1612,622)])]}}
    for cam,(src,dst) in PAIRS.items():
        x,y = {3:(1366,0),4:(0,514),5:(683,514)}[cam]
        cap = cv2.VideoCapture(str(videos/f"camera_new_{cam}.mkv"))
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC,40000)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError(f"Cannot read camera {cam}")
        to_panel, votes = register(frame,sheet[y:y+508,x:x+682],(x,y))
        result["cameras"][str(cam)] = {
            "image_points":transform(src,np.linalg.inv(to_panel)).round(2).tolist(),
            "plan_points":transform(dst,np.linalg.inv(to_inset)).round(2).tolist(),
            "panel_registration_inliers":votes,
        }
    return result


def marker(sheet, roi, bgr, floor_end=False):
    x, y, w, h = roi
    patch = sheet[y:y+h,x:x+w].astype(float)
    yy, xx = np.where(np.max(np.abs(patch-bgr), axis=2) < 20)
    if len(xx) < 8:
        raise ValueError(f"Missing colour mark in {roi}")
    if floor_end:
        tip = yy >= np.percentile(yy, 95)
        xx, yy = xx[tip], yy[tip]
    return [[float(x+np.median(xx)),float(y+np.median(yy))]]


def refine_marks(sheet_path, plan, videos, base_path):
    config=json.loads(base_path.read_text())
    if config["plan_size"] != list(plan.shape[1::-1]):
        raise ValueError("Plan size differs from base calibration")
    sheet=cv2.resize(_read_image(sheet_path),(2048,1024))
    inset,_=register(plan,sheet[512:,1366:],(1366,512))
    # Replace the estimated endpoint on the same wall. Each explicit mark
    # is a single correspondence even though the brush covers several pixels.
    marks=[
        (3,3,"red",(36,28,237),(1708,238,40,95),(1680,664,13,35)),
        (5,1,"green",(29,230,181),(965,800,31,92),(1677,906,18,23)),
    ]
    for cam,index,name,color,frame_roi,plan_roi in marks:
        x,y={3:(1366,0),5:(683,514)}[cam]
        cap=cv2.VideoCapture(str(videos/f"camera_new_{cam}.mkv"))
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC,40000)
            ok,frame=cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError(f"Cannot register camera {cam}")
        panel,_=register(frame,sheet[y:y+508,x:x+682],(x,y))
        source=transform(marker(sheet,frame_roi,color,True),np.linalg.inv(panel))[0]
        target=transform(marker(sheet,plan_roi,color),np.linalg.inv(inset))[0]
        row=config["cameras"][str(cam)]
        old=cv2.getPerspectiveTransform(np.float32(row["image_points"]),np.float32(row["plan_points"]))
        projected=transform([source],old)[0]
        row["image_points"][index]=source.round(2).tolist()
        row["plan_points"][index]=target.round(2).tolist()
        row["mark"]={"color":name,"index":index,"previous_projection":projected.round(2).tolist(),
                     "previous_offset_px":round(float(np.linalg.norm(projected-target)),2)}
    return config


def render(run, plan, config, out):
    out.mkdir(parents=True,exist_ok=True)
    performance=json.loads((run/"performance.json").read_text())
    clock=json.loads((Path(__file__).resolve().parents[1]/"configs/hackaton_office_map_approx.json").read_text())
    fps={int(r["camera_id"]):r.get("source_fps",clock["camera_fps"][str(r["camera_id"])]) for r in performance}
    groups=defaultdict(list)
    counts=Counter()
    fits={}
    canvas=plan.copy()
    for cam,wall in config["walls_only"].items():
        a,b=np.rint(wall).astype(int)
        color=(30,35,150) if cam=="1" else (0,155,135)
        cv2.line(canvas,tuple(a),tuple(b),color,4,cv2.LINE_AA)
        cv2.putText(canvas,f"C{cam} wall",tuple(a+[4,-6]),0,.53,color,2,cv2.LINE_AA)
    for cam,row in config["cameras"].items():
        a,b=(np.float32(row[k]) for k in ("image_points","plan_points"))
        fits[int(cam)]=(cv2.getPerspectiveTransform(a,b),cv2.convexHull(a))
        cv2.polylines(canvas,[cv2.convexHull(b).astype(np.int32)],True,(150,150,150),1)
    with (run/"global_tracks_2d.csv").open(newline="") as f, (out/"wall_tracks.csv").open("w",newline="") as dest:
        writer=csv.writer(dest)
        writer.writerow(["camera_plan","global_id","local_id","frame","time_s","x_plan_px","y_plan_px","status"])
        for r in csv.DictReader(f):
            i,fid,g,l=map(int,(r["camera_id"],r["frame_id"],r["global_id"],r["local_id"]))
            cam=i+1
            t=fid/fps[i]+clock["camera_offsets_s"][str(i)]
            foot=((float(r["x1"])+float(r["x2"]))/2,float(r["y2"]))
            xy=("","")
            status="single_wall" if cam not in fits else "outside_anchors"
            if cam in fits:
                H,hull=fits[cam]
                if cv2.pointPolygonTest(hull,foot,False)>=0:
                    xy=tuple(transform([foot],H)[0].round(2))
                    status="approximate"
                    groups[cam,g,l].append((t,xy))
            counts[cam,status]+=1
            writer.writerow([cam,g,l,fid,round(t,3),*xy,status])
    for (cam,g,l),pts in groups.items():
        color=tuple(map(int,cv2.cvtColor(np.uint8([[[(g*37)%180,220,190]]]),cv2.COLOR_HSV2BGR)[0,0]))
        prev=None
        for t,xy in sorted(pts):
            p=tuple(map(round,xy))
            if prev and t-prev[0]<=1.1/fps[cam-1]*2:
                cv2.line(canvas,prev[1],p,color,3,cv2.LINE_AA)
            cv2.circle(canvas,p,2,color,-1)
            prev=t,p
        cv2.putText(canvas,f"G{g}",p,cv2.FONT_HERSHEY_SIMPLEX,.48,color,2,cv2.LINE_AA)
    for cam,row in config["cameras"].items():
        if "mark" in row:
            info=row["mark"]
            p=tuple(np.rint(row["plan_points"][info["index"]]).astype(int))
            color=(20,30,230) if info["color"]=="red" else (30,160,60)
            cv2.drawMarker(canvas,p,color,cv2.MARKER_DIAMOND,17,3)
            cv2.putText(canvas,f"C{cam}",(p[0]-47,p[1]-10),0,.55,color,2,cv2.LINE_AA)
    cv2.rectangle(canvas,(0,0),(canvas.shape[1],55),(255,255,255),-1)
    cv2.putText(canvas,"Wall anchors | approximate | cams 1, 2: no XY", (12,36),0,.85,(35,35,35),2)
    cv2.imencode(".png",canvas)[1].tofile(out/"wall_tracks.png")
    return dict(sorted(counts.items()))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sheet",type=Path,required=True)
    p.add_argument("--refine-from",type=Path,help="Base wall_anchors.json for the red/green point sheet")
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--videos",type=Path,default=Path("data/Hackaton Data"))
    p.add_argument("--run",type=Path,default=Path("runs/hackaton_5cam_60s_v10"))
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    plan=_read_image(args.plan)
    config=(refine_marks(args.sheet,plan,args.videos,args.refine_from) if args.refine_from
            else calibrate(args.sheet,plan,args.videos))
    print(render(args.run,plan,config,args.out))
    (args.out/"wall_anchors.json").write_text(json.dumps(config,indent=2))
