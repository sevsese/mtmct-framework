"""Global association CLI; separate from expensive neural inference."""
from __future__ import annotations
import argparse
from pathlib import Path
from .association import Config, associate, load_tracklets, save_mapping
from .export import export_predictions


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--scene-id",type=int,default=0)
    p.add_argument("--max-cost",type=float,default=0.125)
    p.add_argument("--min-similarity",type=float,default=0.80,
                   help="Hard cosine-similarity gate before weighted scoring")
    p.add_argument("--max-gap",type=float,default=12.0)
    p.add_argument("--max-speed",type=float,default=3.5)
    p.add_argument("--max-overlap-distance",type=float,default=2.0)
    p.add_argument("--appearance-weight",type=float,default=0.75)
    p.add_argument("--spatial-weight",type=float,default=0.20)
    p.add_argument("--temporal-weight",type=float,default=0.05)
    p.add_argument("--fps",type=float,default=30.)
    p.add_argument("--require-geometry",action="store_true")
    p.add_argument("--ignore-geometry",action="store_true",
                   help="Ignore projected positions until calibration has been validated")
    a=p.parse_args()
    cfg=Config(max_cost=a.max_cost,min_similarity=a.min_similarity,max_gap_seconds=a.max_gap,
               max_speed_mps=a.max_speed,overlap_distance_m=a.max_overlap_distance,
               appearance_weight=a.appearance_weight,spatial_weight=a.spatial_weight,
               temporal_weight=a.temporal_weight,fps=a.fps,
               require_geometry=a.require_geometry,
               use_geometry=not a.ignore_geometry)
    tracklets=load_tracklets(a.run_dir/"local_tracklets.json")
    mapping,edges=associate(tracklets,cfg)
    save_mapping(mapping,a.run_dir/"global_ids.json")
    out_csv,out_txt=export_predictions(
        a.run_dir/"observations.csv",a.run_dir/"global_ids.json",
        a.run_dir,scene_id=a.scene_id)
    print(f"Local tracklets: {len(tracklets)}, candidate edges: {len(edges)}, global IDs: {len(set(mapping.values()))}")
    print("Results:", out_csv, out_txt)
    print("NOTE: person-only approximate 3D boxes are NOT a competitive full Track-1 solution.")

if __name__=="__main__": main()
