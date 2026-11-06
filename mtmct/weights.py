"""Download Person-ReID (MSMT17) checkpoints; not generic ImageNet weights."""
from __future__ import annotations
import argparse
from pathlib import Path

MODELS = {
    "osnet_x0_25": "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_x0_75": "osnet_x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_x1_0": "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_ibn_x1_0": "osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_ain_x1_0": "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth",
}


def download_weights(model: str = "osnet_x1_0", out: Path = Path("weights")) -> Path:
    from huggingface_hub import hf_hub_download
    if model not in MODELS:
        raise ValueError(f"Unknown model {model}, choose {list(MODELS)}")
    path = hf_hub_download(repo_id="kaiyangzhou/osnet", filename=MODELS[model], local_dir=str(out))
    return Path(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), default="osnet_x1_0")
    p.add_argument("--out", type=Path, default=Path("weights"))
    args = p.parse_args()
    print("ReID weights:", download_weights(args.model, args.out))


if __name__ == "__main__":
    main()
