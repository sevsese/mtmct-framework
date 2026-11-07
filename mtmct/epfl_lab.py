"""EPFL Laboratory-6p calibration and sparse ground-plane evaluation.

Research-only, non-official metrics. Does not convert grid labels to fake bbox GT.
The EPFL ground-plane homographies map image pixels -> top-view coordinates.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabGT:
    frame_count: int
    people: int
    grid_width: int
    grid_height: int
    step: int
    first_frame: int
    last_frame: int
    positions: dict[int, dict[int, tuple[float, float]]]


def read_gt(path: str | Path) -> LabGT:
    """Reads CVLab gt_lab_6p.txt. IDs are the column index, NOT grid cell IDs."""
    lines = [line.strip() for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not lines or lines[0] != '1':
        raise ValueError('Not a supported EPFL POM ground-truth file (expected version 1)')
    header = [int(v) for v in lines[1].split()]
    if len(header) != 7:
        raise ValueError('EPFL GT header must have 7 integers')
    nframes, people, gw, gh, step, first, last = header
    if min(nframes,people,gw,gh,step) < 1 or not (0 <= first <= last < nframes):
        raise ValueError('Invalid EPFL GT dimensions or frame range')
    if len(lines) < 2+nframes:
        raise ValueError(f'EPFL GT truncated: {len(lines)-2} data rows, expected {nframes}')
    positions={}
    for frame in range(first, last+1, step):
        vals=[int(v) for v in lines[frame+2].split()]
        if len(vals) != people:
            raise ValueError(f'EPFL GT frame {frame}: expected {people} columns; got {len(vals)}')
        labelled={}
        for person, pos in enumerate(vals):
            if pos >= 0:
                if pos >= gw*gh:
                    raise ValueError(f'EPFL GT invalid grid cell: frame {frame}, pos {pos}')
                labelled[person] = ((pos%gw)+0.5, (pos//gw)+0.5)
        positions[frame] = labelled
    return LabGT(nframes,people,gw,gh,step,first,last,positions)


def read_ground_homographies(path: str | Path) -> dict[int, np.ndarray]:
    text=Path(path).read_text(encoding='utf-8-sig')
    result={}
    camera=None
    lines=text.splitlines()
    for idx,line in enumerate(lines):
        m=re.search(r'^\s*#\s*Camera\s+(\d+)\s*$',line,re.IGNORECASE)
        if m:
            camera=int(m.group(1))
        if camera is not None and re.search(r'^\s*#\s*Ground plane homography\s*$',line,re.IGNORECASE):
            rows=[]
            for value in lines[idx+1:]:
                if not value.strip():
                    continue
                if value.strip().startswith('#'):
                    break
                parts=value.replace(',',' ').split()
                if len(parts) != 3:
                    break
                try:
                    rows.append([float(a) for a in parts])
                except ValueError:
                    break
                if len(rows)==3:
                    break
            if len(rows)!=3:
                raise ValueError(f'EPFL homography for camera {camera} incomplete')
            mat=np.asarray(rows,dtype=float)
            if abs(np.linalg.det(mat))<1e-13:
                raise ValueError(f'EPFL homography for camera {camera} is singular')
            result[camera]=mat
    if len(result)<4:
        raise ValueError(f'Expected four EPFL ground homographies, got {sorted(result)}')
    return result


def _project_pixel_to_grid(u, v, h, gt: LabGT) -> tuple[float,float] | None:
    # EPFL-6p: top-view origin (0,0), width 358, height 360 in author units.
    vec=h@np.array([u,v,1.0],dtype=float)
    if abs(vec[2])<1e-12:
        return None
    tx,ty=float(vec[0]/vec[2]),float(vec[1]/vec[2])
    gx=tx*gt.grid_width/358.0
    gy=ty*gt.grid_height/360.0
    if not (np.isfinite(gx) and np.isfinite(gy)):
        return None
    return gx,gy


def evaluate_sparse_ground(pred_csv: str | Path, gt_path: str | Path,
                           calibration_path: str | Path, out_json: str | Path,
                           cameras: tuple[int, ...] = (0,1,2,3),
                           max_frames: int = 0, max_distance_cells: float = 2.0) -> dict:
    """Unofficial ground-grid IDF1 proxy at EPFL annotated frames only.

    Predictions of a global ID at one time from different cameras are median-fused.
    Hungarian per-frame spatial assignments give eligible GT-person/pred-ID pairs.
    A second Hungarian maximizes temporal identity correspondence.
    Not the original EPFL/POM metric and not bbox-based IDF1.
    """
    from scipy.optimize import linear_sum_assignment
    if max_distance_cells<=0:
        raise ValueError('max_distance_cells must be positive')
    gt=read_gt(gt_path)
    homographies=read_ground_homographies(calibration_path)
    cams=set(cameras)
    if not cams.issubset(homographies):
        raise ValueError(f'No EPFL calibration for cameras {cams-set(homographies)}')
    predictions=defaultdict(list)
    with Path(pred_csv).open(newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            frame=int(row['frame_id'])
            camera=int(row['camera_id'])
            if camera not in cams or frame not in gt.positions or (max_frames>0 and frame>=max_frames):
                continue
            u=(float(row['x1'])+float(row['x2']))/2.0
            v=float(row['y2'])
            coord=_project_pixel_to_grid(u,v,homographies[camera],gt)
            if coord is None:
                continue
            x,y=coord
            if 0 <= x < gt.grid_width and 0 <= y < gt.grid_height:
                predictions[(frame,int(row['global_id']))].append(coord)
    by_frame=defaultdict(list)
    dispersion_conflicts=0
    for (frame,gid), ps in sorted(predictions.items()):
        vals=np.array(ps)
        median=np.median(vals,axis=0)
        if len(vals)>1 and np.max(np.linalg.norm(vals-median,axis=1))>max_distance_cells:
            dispersion_conflicts+=1
        by_frame[frame].append((gid,median))
    matched_pairs=defaultdict(int)
    det_tp=0
    total_gt=0
    total_pred=0
    frames_evaluated=0
    for frame, persons in gt.positions.items():
        if max_frames>0 and frame>=max_frames: continue
        # A frame in which no person was annotated is not assumed fully labelled.
        if not persons: continue
        frames_evaluated+=1
        candidates=by_frame.get(frame,[])
        gitems=list(sorted(persons.items()))
        total_gt+=len(gitems)
        total_pred+=len(candidates)
        if not candidates: continue
        truth=np.array([v for _,v in gitems])
        pred=np.array([v for _,v in candidates])
        dist=np.linalg.norm(truth[:,None,:]-pred[None,:,:],axis=2)
        # Adding dummy columns means no match beyond distance gate; avoid stealing another person's assignment.
        cost=np.concatenate([dist,np.full((len(gitems),len(gitems)),max_distance_cells)],axis=1)
        rr,cc=linear_sum_assignment(cost)
        for i,j in zip(rr,cc):
            if j < len(candidates) and dist[i,j] <= max_distance_cells:
                det_tp+=1
                matched_pairs[(gitems[i][0],candidates[j][0])]+=1
    gts=sorted({key[0] for key in matched_pairs})
    gids=sorted({key[1] for key in matched_pairs})
    idtp=0
    if gts and gids:
        counts=np.zeros((len(gts),len(gids)),dtype=int)
        gi={v:i for i,v in enumerate(gts)}
        pi={v:i for i,v in enumerate(gids)}
        for (gt_id,global_id),n in matched_pairs.items():
            counts[gi[gt_id],pi[global_id]]=n
        rr,cc=linear_sum_assignment(-counts)
        idtp=int(counts[rr,cc].sum())
    def safe_div(a,b): return a/b if b else 0.0
    report={
        'protocol': 'epfl_lab_6p_sparse_ground_grid_v1_UNOFFICIAL',
        'warning': ('NOT NVIDIA bbox IDF1, NOT official EPFL metric. '
                    'GT is sparse ground-grid positions, not image bounding boxes. '
                    'GT -1 is unknown and -2 outside the grid; unlabelled frames ignored.'),
        'cameras': sorted(cams),
        'gt_frame_step': gt.step,
        'frames_with_active_gt': frames_evaluated,
        'max_distance_cells':max_distance_cells,
        'gt_points':total_gt,'predicted_global_points':total_pred,
        'spatial_tp':det_tp,
        'spatial_precision':safe_div(det_tp,total_pred),
        'spatial_recall':safe_div(det_tp,total_gt),
        'spatial_f1':safe_div(2*det_tp,total_pred+total_gt),
        'identity_matched_points':idtp,
        'sparse_ground_idf1_proxy':safe_div(2*idtp,total_pred+total_gt),
        'same_id_cross_camera_dispersion_cases':dispersion_conflicts,
        'gt_unmatched':total_gt-det_tp,
        'pred_unmatched':total_pred-det_tp,
    }
    path=Path(out_json)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report
