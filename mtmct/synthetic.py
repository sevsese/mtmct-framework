"""Small SELF-CONTAINED synthetic tracklets to exercise the association code.

These are not AICity data or model predictions. They only test the logic.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from .association import Config, associate,save_mapping
from .export import export_predictions,FIELDS


def make_demo(out: Path):
    out.mkdir(parents=True,exist_ok=True)
    # Person A moves from camera 0 to 1; person B has a different appearance.
    # Another track overlaps camera 0 in time and cannot get A's ID.
    tracks=[
        {"camera_id":0,"local_id":7,"start_frame":0,"end_frame":60,
         "positions":[[0,0,0],[30,1,0],[60,2,0]],"embedding":[1,0,0]},
        {"camera_id":1,"local_id":12,"start_frame":30,"end_frame":90,
         "positions":[[30,1.1,0],[60,2.1,0],[90,3,0]],"embedding":[.99,.01,0]},
        {"camera_id":0,"local_id":8,"start_frame":20,"end_frame":80,
         "positions":[[30,9,9],[60,9,8]],"embedding":[.98,.02,0]},
        {"camera_id":1,"local_id":13,"start_frame":30,"end_frame":70,
         "positions":[[30,9.1,9.0],[60,9.0,8.1]],"embedding":[0,1,0]},
    ]
    with open(out/"local_tracklets.json","w",encoding="utf-8") as f:
        json.dump(tracks,f,indent=2)
    with open(out/"observations.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        for t in tracks:
            for frame,x,y in t["positions"]:
                w.writerow({"camera_id":t["camera_id"],"local_id":t["local_id"],
                            "frame_id":frame,"x1":10,"y1":20,"x2":30,"y2":90,
                            "confidence":.9,"world_x":x,"world_y":y})
    mapping,edges=associate(tracks,Config())
    save_mapping(mapping,out/"global_ids.json")
    export_predictions(out/"observations.csv",out/"global_ids.json",out)
    print("Synthetic test only. Global ID mapping:")
    for k,v in sorted(mapping.items()):print(k,"->",v)
    print("Created",out)
    assert mapping[(0,7)]==mapping[(1,12)]
    assert mapping[(0,7)]!=mapping[(0,8)]

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=Path("runs/synthetic"))
    make_demo(p.parse_args().out)
