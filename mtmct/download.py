"""Download only selected AICity 2026 RGB camera videos and calibration.

Default: 2 cameras from Warehouse_000 train; ~363 MB of video plus calibration.
Full dataset and depth maps are intentionally NOT downloaded.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO="nvidia/PhysicalAI-SmartSpaces"


def download(split: str, scene: str, cameras: list[int], output: Path, gt: bool):
    base=f"MTMC_Tracking_2026/{split}/{scene}"
    targets=[f"{base}/calibration.json"]
    if gt:
        targets.append(f"{base}/ground_truth.json")
    for camera in cameras:
        targets.append(f"{base}/videos/Camera_{camera:04d}.mp4")
    for filename in targets:
        print("Downloading", filename, flush=True)
        try:
            local=hf_hub_download(
                repo_id=REPO, repo_type="dataset", filename=filename,
                local_dir=str(output))
        except Exception as exc:
            raise RuntimeError(f"Failed to download {filename}: {exc}") from exc
        print("  ->", local, flush=True)
    print("Scene directory:", output/base)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",choices=["train","val","test"],default="train")
    p.add_argument("--scene",default="Warehouse_000")
    p.add_argument("--cameras",type=int,nargs="+",default=[0,1])
    p.add_argument("--output",type=Path,default=Path("data"))
    p.add_argument("--ground-truth",action="store_true",help="Potentially hundreds of MB")
    a=p.parse_args()
    download(a.split,a.scene,a.cameras,a.output,a.ground_truth)

if __name__=="__main__": main()
