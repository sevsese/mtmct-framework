"""One full view and one fixed far-field crop feed a single camera tracker."""
import csv
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml


class ZoomTracker:
    def __init__(self, model, roi, tracker, log, exclude=None):
        self.model = model
        self.roi = roi
        self.exclude = [] if exclude is None else (
            [exclude] if len(exclude) == 4 and isinstance(exclude[0], (int, float)) else exclude
        )
        settings = YAML.load(check_yaml(str(tracker)))
        if settings["tracker_type"] != "bytetrack":
            raise ValueError("Corridor zoom requires ByteTrack")
        self.tracker = BYTETracker(SimpleNamespace(**settings))
        self.log = csv.writer(log)
        self.log.writerow(["frame", "pass", "x1", "y1", "x2", "y2", "score"])
        self.full_seconds = self.zoom_seconds = 0.

    def step(self, frame, fid, imgsz, conf, device):
        h, w = frame.shape[:2]
        params = dict(classes=[0], conf=conf, device=device, verbose=False)
        started = time.perf_counter()
        result = self.model.predict(frame,imgsz=imgsz,**params)[0]
        full = result.boxes.data.cpu().numpy()
        self.full_seconds += time.perf_counter()-started
        started = time.perf_counter()
        combined = full
        if self.roi is not None:
            x,y,r,b = np.rint(np.array(self.roi)*[w,h,w,h]).astype(int)
            if not (0 <= x < r <= w and 0 <= y < b <= h):
                raise ValueError(f"Invalid normalized ROI {self.roi}")
            crop = self.model.predict(frame[y:b,x:r],imgsz=960,**params)[0].boxes.data.cpu().numpy().copy()
            # Discard boxes clipped by an artificial crop edge; the full pass covers it.
            crop = crop[(crop[:,0]>2) & (crop[:,2]<r-x-2) &
                        ((y==0) | (crop[:,1]>2)) & (crop[:,3]<b-y-2)]
            crop[:,:4] += [x,y,x,y]
            combined = np.concatenate((full,crop))
            if len(crop):
                rects=combined[:,:4].copy()
                rects[:,2:] -= rects[:,:2]
                keep=cv2.dnn.NMSBoxes(rects.tolist(),combined[:,4].tolist(),0.,.50)
                combined=combined[np.asarray(keep,dtype=int).ravel()]
        for excluded in self.exclude:
            left, top, right, bottom = np.asarray(excluded) * [w,h,w,h]
            foot_x = (combined[:,0] + combined[:,2]) / 2
            foot_y = combined[:,3]
            combined = combined[~((foot_x >= left) & (foot_x <= right) &
                                  (foot_y >= top) & (foot_y <= bottom))]
        self.zoom_seconds += time.perf_counter()-started
        for name,boxes in (("full",full),("fused",combined)):
            for box in boxes:
                self.log.writerow([fid,name,*box[:5]])
        result.update(boxes=torch.as_tensor(combined))
        tracks=self.tracker.update(result.boxes.numpy(),frame)
        result.update(boxes=torch.as_tensor(tracks[:,:-1]) if len(tracks) else torch.empty((0,7)))
        return result
