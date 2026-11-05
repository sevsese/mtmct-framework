"""One-command MTMCT pipeline for the validated Warehouse_000 camera pair."""
from __future__ import annotations
import argparse
from pathlib import Path

from .association import Config, associate, load_tracklets, save_mapping
from .download import download
from .export import export_predictions
from .tracking import run
from .visualize import render


def main():
    p=argparse.ArgumentParser(description="Download, track, associate and render MTMCT")
    p.add_argument("--split",choices=["train","val","test"],default="train")
    p.add_argument("--scene",default="Warehouse_000")
    p.add_argument("--cameras",type=int,nargs="+",default=[3,15])
    p.add_argument("--data-root",type=Path,default=Path("data"))
    p.add_argument("--out",type=Path,default=None)
    p.add_argument("--model",default="yolo26l.pt")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--max-frames",type=int,default=900,help="0 = full videos")
    p.add_argument("--imgsz",type=int,default=960)
    p.add_argument("--conf",type=float,default=0.10)
    p.add_argument("--sample-every",type=int,default=5)
    p.add_argument("--min-similarity",type=float,default=0.80)
    p.add_argument("--max-cost",type=float,default=0.125)
    p.add_argument("--geometry",action="store_true",help="Use validated floor geometry")
    p.add_argument("--link-only",action="store_true",help="Reuse existing local_tracklets.json")
    p.add_argument("--no-download",action="store_true")
    p.add_argument("--scale",type=float,default=0.45)
    a=p.parse_args()

    tag="_".join(map(str,a.cameras))
    out=a.out or Path("runs")/f"{a.scene.lower()}_cam{tag}"
    scene_dir=a.data_root/"MTMC_Tracking_2026"/a.split/a.scene
    required=[scene_dir/"calibration.json",*(scene_dir/"videos"/f"Camera_{c:04d}.mp4" for c in a.cameras)]
    missing=[path for path in required if not path.exists()]
    if missing and a.no_download:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    if missing:
        download(a.split,a.scene,a.cameras,a.data_root,False)

    if not a.link_only:
        run(scene_dir,out,a.model,"osnet",None,a.device,a.max_frames,1,
            a.sample_every,a.imgsz,a.conf,cameras=a.cameras)

    tracklets=load_tracklets(out/"local_tracklets.json")
    cfg=Config(appearance_weight=0.95,spatial_weight=0.0 if not a.geometry else 0.20,
               temporal_weight=0.05,max_cost=a.max_cost,
               min_similarity=a.min_similarity,use_geometry=a.geometry)
    mapping,edges=associate(tracklets,cfg)
    save_mapping(mapping,out/"global_ids.json")
    predictions,_=export_predictions(out/"observations.csv",out/"global_ids.json",out)
    render(scene_dir,predictions,out/"mosaic.mp4",a.max_frames,1,a.scale,a.cameras)
    cross=sum(len({c for c,l in mapping if mapping[(c,l)]==gid})>1
              for gid in set(mapping.values()))
    print(f"DONE: {len(tracklets)} local tracks, {len(edges)} candidate edges, "
          f"{cross} cross-camera global IDs")
    print("Video:",out/"mosaic.mp4")


if __name__=="__main__":
    main()
