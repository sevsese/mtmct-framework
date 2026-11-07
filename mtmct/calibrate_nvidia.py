"""Build calibration pairs from GT-pure local tracks in distinct recordings."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from .association import Config
from .evaluate import _match_boxes
from .nvidia import NvidiaDatasetAdapter
from .probability import fit
from .scoring import Evidence, physical, score


def pairs(features, observations, gt, recording, limit):
    rows=json.loads(Path(features).read_text(encoding='utf-8'))
    pred,truth=defaultdict(list),defaultdict(list)
    counts=Counter()
    for r in csv.DictReader(Path(observations).open(encoding='utf-8-sig')):
        c,f,l=(int(r[k]) for k in ('camera_id','frame_id','local_id'))
        if f<limit:
            pred[c,f].append((l,[float(r[k]) for k in ('x1','y1','x2','y2')]))
            counts[c,l]+=1
    for obs in NvidiaDatasetAdapter(gt).observations(max_frames=limit):
        if (obs.camera_id,obs.frame_index) in pred:
            truth[obs.camera_id,obs.frame_index].append((obs.gt_object_id,obs.bbox_xyxy))
    votes=defaultdict(Counter)
    for key,detections in pred.items():
        for person,local in _match_boxes(truth[key],detections,.5):
            votes[key[0],local][person]+=1
    labels={key:counts_.most_common(1)[0][0] for key,counts_ in votes.items()
            if sum(counts_.values())>=5 and counts_.most_common(1)[0][1]/counts[key]>=.8}
    cfg=Config(appearance_weight=.92,spatial_weight=0,temporal_weight=.08,use_geometry=False,gallery_weight=.25,part_weight=0.)
    tracks=[Evidence(r,cfg) for r in rows if (int(r['camera_id']),int(r['local_id'])) in labels]
    output=[]
    for a,b in combinations(tracks,2):
        if a.key[0]==b.key[0]:
            continue
        result=score(a,b,cfg,physical(a,b,cfg),'max')
        if result['cost'] is not None:
            output.append(dict(cost=result['cost'],label=int(labels[a.key]==labels[b.key]),recording=recording))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',nargs=4,required=True,metavar=('FEATURES','OBSERVATIONS','GT','RECORDING'))
    p.add_argument('--validation',nargs=4,required=True,metavar=('FEATURES','OBSERVATIONS','GT','RECORDING'))
    p.add_argument('--frames',type=int,default=900)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    training=pairs(*a.train,a.frames)
    validation=pairs(*a.validation,a.frames)
    print(fit(training,validation,a.output))


if __name__=='__main__':
    main()
