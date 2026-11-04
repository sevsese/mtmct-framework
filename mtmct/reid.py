"""Appearance embeddings: real OSNet person-ReID or a lightweight color baseline.

The color histogram is for pipeline testing ONLY; it is not a neural ReID model.
"""
from __future__ import annotations
import cv2
import numpy as np


def normalized(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


class ReID:
    def __init__(self, method: str, device: str = "cpu", weights: str | None = None,
                 model_name: str = "osnet_x1_0"):
        self.method = method
        self.extractor = None
        if method == "osnet":
            if not weights:
                raise ValueError("--reid osnet requires --reid-weights pointing to a person-ReID checkpoint")
            try:
                from torchreid.utils import FeatureExtractor
            except ImportError as e:
                raise RuntimeError("Install Torchreid: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git") from e
            self.extractor = FeatureExtractor(
                model_name=model_name, model_path=weights,
                device=device, verbose=False)
        elif method != "color":
            raise ValueError("Choose --reid osnet or --reid color")

    def encode(self, rgb_crops: list[np.ndarray]) -> list[np.ndarray]:
        if not rgb_crops:
            return []
        if self.method == "osnet":
            import torch
            with torch.inference_mode():
                out = self.extractor(rgb_crops).detach().cpu().numpy()
            return [normalized(e.astype(np.float32)) for e in out]
        result=[]
        for crop in rgb_crops:
            # Vertical partitions retain some top/bottom clothing arrangement.
            hsv=cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            parts=np.array_split(hsv, 2, axis=0)
            features=[]
            for part in parts:
                hist=cv2.calcHist([part],[0,1],None,[16,8],[0,180,0,256]).ravel()
                hist=hist/max(hist.sum(),1.)
                features.append(hist)
            result.append(normalized(np.concatenate(features).astype(np.float32)))
        return result
