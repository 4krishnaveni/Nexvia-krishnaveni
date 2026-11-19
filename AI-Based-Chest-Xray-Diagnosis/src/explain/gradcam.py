from __future__ import annotations

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def compute_cam(model, x: torch.Tensor) -> np.ndarray:
    """x: (B,3,H,W) normalised input. Returns (B,H,W) float in [0,1].
    The single logit is the 'pneumonia-like opacity' score, so we explain output 0."""
    with GradCAM(model=model, target_layers=[model.target_layer]) as cam:
        return cam(input_tensor=x, targets=[ClassifierOutputTarget(0)] * x.shape[0])


def cam_peak(cam: np.ndarray) -> tuple[int, int]:
    """(x, y) of the strongest activation."""
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return int(x), int(y)


def pointing_game_hit(cam: np.ndarray, boxes) -> bool:
    """True if the CAM peak lies inside any ground-truth box (x1,y1,x2,y2)."""
    px, py = cam_peak(cam)
    return any(x1 <= px <= x2 and y1 <= py <= y2 for x1, y1, x2, y2 in boxes)


def heatmap_rgba(cam: np.ndarray, size: int | None = None) -> np.ndarray:
    """Transparent JET heatmap (uint8 RGBA). Alpha follows intensity, so low-activation areas
    stay see-through when stacked on the X-ray in the viewer."""
    if size:
        cam = cv2.resize(cam, (size, size))
    cam = np.clip(cam, 0, 1)
    bgr = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    alpha = (cam * 220).astype(np.uint8)
    return np.dstack([rgb, alpha])


def blend_on_image(img01: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Opaque RGB overlay for reports/slides."""
    rgba = heatmap_rgba(cam).astype(np.float32)
    base = np.repeat((img01 * 255)[..., None], 3, -1)
    a = rgba[..., 3:] / 255
    return (base * (1 - a * 0.6) + rgba[..., :3] * a * 0.6).clip(0, 255).astype(np.uint8)
