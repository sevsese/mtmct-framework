"""TrackEval HOTA implementation with an explicit custom multicamera 2D protocol."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .evaluate import iou
from .nvidia import NvidiaDatasetAdapter


def evaluate(predictions, gt_path, cameras, max_frames, output):
    from trackeval.metrics import HOTA
    pred,gt=defaultdict(list),defaultdict(list)
    for row in csv.DictReader(Path(predictions).open(encoding='utf-8-sig')):
        camera,frame=int(row['camera_id']),int(row['frame_id'])
        if camera in cameras and 0<=frame<max_frames:
            pred[frame,camera].append((int(row['global_id']),[float(row[k]) for k in ('x1','y1','x2','y2')]))
    for obs in NvidiaDatasetAdapter(gt_path).observations(max_frames=max_frames):
        if obs.camera_id in cameras:
            gt[obs.frame_index,obs.camera_id].append((obs.gt_object_id,obs.bbox_xyxy))
    gt_ids=sorted({i for rows in gt.values() for i,_ in rows})
    pred_ids=sorted({i for rows in pred.values() for i,_ in rows})
    gi,pi={v:i for i,v in enumerate(gt_ids)},{v:i for i,v in enumerate(pred_ids)}
    data=dict(gt_ids=[],tracker_ids=[],similarity_scores=[],num_gt_ids=len(gi),num_tracker_ids=len(pi),
              num_gt_dets=sum(map(len,gt.values())),num_tracker_dets=sum(map(len,pred.values())),num_timesteps=max_frames*len(cameras))
    for frame in range(max_frames):
        for camera in cameras:
            a,b=gt[frame,camera],pred[frame,camera]
            if len({i for i,_ in b})!=len(b):
                raise ValueError('Duplicate predicted identity within one camera frame')
            data['gt_ids'].append(np.array([gi[i] for i,_ in a],dtype=int))
            data['tracker_ids'].append(np.array([pi[i] for i,_ in b],dtype=int))
            data['similarity_scores'].append(np.array([[iou(x,y) for _,y in b] for _,x in a]).reshape(len(a),len(b)))
    raw=HOTA().eval_sequence(data)
    result=dict(protocol='trackeval-hota-multicamera-2d-flattened-v1',official_dataset_protocol=False,
                note='TrackEval HOTA implementation; camera-frame flattening is custom, not NVIDIA official 3D protocol',
                cameras=cameras,max_frames=max_frames,
                **{key:float(np.mean(raw[key])) for key in ('HOTA','DetA','AssA','LocA')})
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
