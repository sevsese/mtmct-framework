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


def distinct_crop_boxes(full, crop):
    """Keep crop-only detections; retain the full-view box for near-duplicates."""
    if not len(full) or not len(crop):
        return crop
    whole = full[:, :4]
    part = crop[:, :4]
    intersection = np.maximum(
        0, np.minimum(part[:, None, 2:], whole[None, :, 2:]) -
        np.maximum(part[:, None, :2], whole[None, :, :2])
    ).prod(axis=2)
    smaller_area = np.minimum(
        (part[:, 2:] - part[:, :2]).prod(axis=1)[:, None],
        (whole[:, 2:] - whole[:, :2]).prod(axis=1)[None, :]
    )
    part_foot = np.column_stack(((part[:, 0] + part[:, 2]) / 2, part[:, 3]))
    whole_foot = np.column_stack(((whole[:, 0] + whole[:, 2]) / 2, whole[:, 3]))
    distance = np.linalg.norm(part_foot[:, None] - whole_foot[None], axis=2)
    smaller_height = np.minimum((part[:, 3] - part[:, 1])[:, None],
                                (whole[:, 3] - whole[:, 1])[None, :])
    duplicate = (smaller_area > 0) & (intersection >= .85 * smaller_area) & (
        distance <= .15 * smaller_height
    )
    return crop[~duplicate.any(axis=1)]


def filter_person_size(boxes, shape, minimum):
    """Minimum width/height fractions in original camera coordinates."""
    if len(minimum) != 2 or not all(np.isfinite(v) and 0 <= v < 1 for v in minimum):
        raise ValueError("Minimum person size must be two fractions in [0,1)")
    h, w = shape[:2]
    return boxes[((boxes[:, 2]-boxes[:, 0]) >= minimum[0]*w) &
                 ((boxes[:, 3]-boxes[:, 1]) >= minimum[1]*h)]


class ZoomTracker:
    def __init__(self, model, roi, tracker, log, exclude=None, minimum=(0., 0.)):
        self.model = model
        self.roi = roi
        self.minimum = minimum
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
            crop = distinct_crop_boxes(full,crop)
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
        combined = filter_person_size(combined, frame.shape, self.minimum)
        self.zoom_seconds += time.perf_counter()-started
        for name,boxes in (("full",full),("fused",combined)):
            for box in boxes:
                self.log.writerow([fid,name,*box[:5]])
        result.update(boxes=torch.as_tensor(combined))
        tracks=self.tracker.update(result.boxes.numpy(),frame)
        result.update(boxes=torch.as_tensor(tracks[:,:-1]) if len(tracks) else torch.empty((0,7)))
        return result
