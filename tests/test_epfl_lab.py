from pathlib import Path
import csv
import json
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mtmct.epfl_lab import read_gt,read_ground_homographies,evaluate_sparse_ground


def files(tmp_path):
    gt=tmp_path/'gt_lab_6p.txt'
    rows=['1','4 2 56 56 1 0 3','0 5','1 6','2 7','3 8']
    gt.write_text('\n'.join(rows),encoding='utf-8')
    cal=tmp_path/'calibration-6p.txt'
    # Using projection that maps pixel (u,v) to topview (u,v) directly.
    cal.write_text('\n'.join(f'# Camera {i}\n# Ground plane homography\n1 0 0\n0 1 0\n0 0 1\n'
                             for i in range(4)),encoding='utf-8')
    return gt,cal


def test_read_gt_and_calibration(tmp_path):
    gt,cal=files(tmp_path)
    parsed=read_gt(gt)
    assert parsed.positions[0][0]==(0.5,0.5)
    assert parsed.positions[0][1]==(5.5,0.5)
    assert len(read_ground_homographies(cal))==4


def test_spatial_eval_good_and_bad_id(tmp_path):
    gt,cal=files(tmp_path)
    csvp=tmp_path/'global_tracks_2d.csv'
    with csvp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['frame_id','camera_id','global_id','x1','x2','y2'])
        writer.writeheader()
        for fid in range(4):
            for person,cell in enumerate((fid,fid+5)):
                gx=cell+0.5
                # grid to topview inverse for identity homography
                u=gx*358/56
                v=0.5*360/56
                writer.writerow(dict(frame_id=fid,camera_id=0,global_id=person+1,
                                     x1=u-1,x2=u+1,y2=v))
    out=tmp_path/'metrics.json'
    r=evaluate_sparse_ground(csvp,gt,cal,out,(0,),4,1.0)
    assert r['spatial_f1']==1.0
    assert r['sparse_ground_idf1_proxy']==1.0
    assert json.loads(out.read_text())['protocol'].endswith('UNOFFICIAL')


def test_sparse_annotation_ignore_unknowns(tmp_path):
    gt,cal=files(tmp_path)
    gt.write_text('1\n4 2 56 56 2 0 2\n-1 -2\n10 11\n12 -2\n13 14\n')
    x=read_gt(gt)
    assert x.positions=={0:{},2:{0:(12.5,0.5)}}
