"""Loads both models once and turns a DICOM into prediction + overlay images."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.dicom_utils import dicom_to_array, extract_metadata, resize
from src.data.transforms import to_cls_tensor, to_seg_tensor
from src.explain.gradcam import compute_cam, heatmap_rgba
from src.models.classifier import CXRClassifier, load_classifier
from src.models.segmenter import build_unet, load_unet

log = logging.getLogger("inference")


class Predictor:
    def __init__(self, cls_path: str, seg_path: str, size: int = 256, allow_untrained: bool = False,
                 device: str | None = None):
        self.size = size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.untrained = False
        if Path(cls_path).exists() and Path(seg_path).exists():
            self.cls, ck = load_classifier(cls_path, self.device)
            self.seg, sk = load_unet(seg_path, self.device)
            self.T, self.thr = ck["temperature"], ck["threshold"]
            self.version = f"{ck['version']}|{sk['version']}"
        elif allow_untrained:
            log.warning("NO TRAINED WEIGHTS FOUND - using random weights. Outputs are meaningless.")
            self.cls = CXRClassifier("densenet121", pretrained=False).to(self.device).eval()
            self.seg = build_unet().to(self.device).eval()
            self.T, self.thr, self.version, self.untrained = 1.0, 0.5, "UNTRAINED-DEMO", True
        else:
            raise FileNotFoundError(f"Missing weights {cls_path} / {seg_path}. Train first, or set ALLOW_UNTRAINED=1.")

    def predict(self, ds) -> dict:
        """ds: pydicom Dataset. Returns dict with arrays + scalars (no file IO)."""
        img = resize(dicom_to_array(ds), self.size)
        x = to_cls_tensor(img)[None].to(self.device)
        with torch.no_grad():
            prob = float(torch.sigmoid(self.cls(x).squeeze() / self.T))
            m = torch.sigmoid(self.seg(to_seg_tensor(img)[None].to(self.device)))[0, 0].cpu().numpy()
        cam = compute_cam(self.cls, x)[0]  # needs grad, so outside no_grad
        return {
            "image": img, "prob": prob, "label": int(prob >= self.thr), "threshold": self.thr,
            "mask": (m > 0.5).astype(np.uint8), "cam": cam, "version": self.version,
            "meta": extract_metadata(ds),
        }


def save_outputs(res: dict, out_dir: Path) -> dict:
    """Write PNGs the viewer stacks: base image, heatmap (RGBA), mask (RGBA)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    size = res["image"].shape[0]
    cv2.imwrite(str(out_dir / "image.png"), (res["image"] * 255).astype(np.uint8))
    hm = heatmap_rgba(res["cam"], size)
    cv2.imwrite(str(out_dir / "heatmap.png"), cv2.cvtColor(hm, cv2.COLOR_RGBA2BGRA))
    mk = np.zeros((size, size, 4), np.uint8)
    mk[res["mask"] > 0] = (255, 40, 40, 140)
    cv2.imwrite(str(out_dir / "mask.png"), cv2.cvtColor(mk, cv2.COLOR_RGBA2BGRA))
    return {"image": "image.png", "heatmap": "heatmap.png", "mask": "mask.png"}


def env_flag(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes")
