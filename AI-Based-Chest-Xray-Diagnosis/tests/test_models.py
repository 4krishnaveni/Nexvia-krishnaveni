import numpy as np
import torch

from src.explain.gradcam import compute_cam, heatmap_rgba, pointing_game_hit
from src.models.classifier import CXRClassifier
from src.models.segmenter import build_unet


def test_classifier_output_shape():
    m = CXRClassifier("densenet121", pretrained=False).eval()
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 1)


def test_unet_output_shape():
    assert build_unet().eval()(torch.randn(2, 1, 64, 64)).shape == (2, 1, 64, 64)


def test_gradcam_runs_and_is_normalised():
    m = CXRClassifier("densenet121", pretrained=False).eval()
    cam = compute_cam(m, torch.randn(2, 3, 64, 64))
    assert cam.shape == (2, 64, 64) and cam.min() >= 0 and cam.max() <= 1.0001
    rgba = heatmap_rgba(cam[0], 128)
    assert rgba.shape == (128, 128, 4) and rgba.dtype == np.uint8


def test_pointing_game():
    cam = np.zeros((10, 10)); cam[5, 7] = 1  # peak at x=7, y=5
    assert pointing_game_hit(cam, [(6, 4, 9, 8)])
    assert not pointing_game_hit(cam, [(0, 0, 3, 3)])
