"""Office routes after excluding detections contained in the camera 3 TV."""
import csv
from collections import Counter, defaultdict

import cv2
import numpy as np

from .floor_map import homographies, load_calibration, project_foot


# Screen outline on a 648 x 486 reference frame, including the bezel.
SCREEN = np.float32([[442, 99], [606, 121], [595, 181], [427, 171]])


def map_office_routes(run, plan, calibration):
    config = load_calibration(calibration)
    rough = homographies(config)
    with (run / "wall_tracks.csv").open(newline="") as stream:
        anchors = {(int(r["camera_plan"])-1, int(r["frame"]), int(r["local_id"])): r
                   for r in csv.DictReader(stream)}
    canvas = plan.copy()
    previous = {}
    counts = defaultdict(Counter)
    removed = Counter()
    height, width = canvas.shape[:2]
    with (run / "global_tracks_2d.csv").open(newline="") as source, (run / "office_routes.csv").open("w", newline="") as dest:
        writer = csv.writer(dest)
        writer.writerow(["global_id","camera","frame","time_s","x_plan_px","y_plan_px","projection"])
        for row in csv.DictReader(source):
            cam, frame, local, gid = (int(row[k]) for k in ("camera_id","frame_id","local_id","global_id"))
            box = tuple(float(row[k]) for k in ("x1","y1","x2","y2"))
            size = config["cameras"][str(cam)]["image_size"]
            if cam == 2:
                # Require almost the whole box inside the screen so a passer-by can cross it.
                x,y,X,Y = np.array(box) * np.tile([648/size[0],486/size[1]],2)
                rect = np.float32([[x,y],[X,y],[X,Y],[x,Y]])
                overlap, _ = cv2.intersectConvexConvex(SCREEN,rect)
                if overlap >= .90 * (X-x)*(Y-y):
                    removed[gid] += 1
                    continue
            anchor = anchors[cam,frame,local]
            t = float(anchor["time_s"])
            status = anchor["status"]
            point = None
            if status == "approximate":
                point = (float(anchor["x_plan_px"]),float(anchor["y_plan_px"]))
                status = "wall"
            elif status == "single_wall":
                norm = project_foot(rough[cam],box,size)
                if norm:
                    point = (norm[0]*(width-1),norm[1]*(height-1))
                    status = "rough"
            counts[gid][status] += 1
            writer.writerow([gid,cam+1,frame,t,*(point or ("","")),status])
            key = (gid,cam,local)
            if point is None:
                previous.pop(key,None)
                continue
            pixel = tuple(map(round,point))
            color = tuple(map(int,cv2.cvtColor(np.uint8([[[(gid*37)%180,235,175]]]),cv2.COLOR_HSV2BGR)[0,0]))
            prev = previous.get(key)
            if prev and 0 < t-prev[0] < .21:
                cv2.line(canvas,prev[1],pixel,color,4,cv2.LINE_AA)
            cv2.circle(canvas,pixel,2,color,-1)
            previous[key] = (t,pixel)
    for n,(gid,c) in enumerate(sorted(counts.items())):
        color = tuple(map(int,cv2.cvtColor(np.uint8([[[(gid*37)%180,235,175]]]),cv2.COLOR_HSV2BGR)[0,0]))
        y = 110+n*35
        cv2.circle(canvas,(725,y),7,color,-1)
        label = f"G{gid}   {c['wall']+c['rough']}/{sum(c.values())} mapped"
        cv2.putText(canvas,label,(744,y+6),0,.65,(35,35,35),2,cv2.LINE_AA)
    cv2.imencode(".png",canvas)[1].tofile(run/"office_routes.png")
    return dict(removed),dict(counts)
