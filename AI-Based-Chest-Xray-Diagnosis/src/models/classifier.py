from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CXRClassifier(nn.Module):
    """Binary pneumonia-like-opacity classifier (1 logit).

    Wrapped (instead of using torchvision's forward) so that
      * ReLUs are NOT in-place -> Grad-CAM hooks work on DenseNet,
      * the Grad-CAM target layer is always `self.target_layer`.
    """

    def __init__(self, backbone: str = "densenet121", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "densenet121":
            m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
            self.features, self.head, self.relu_after = m.features, nn.Linear(1024, 1), True
        elif backbone == "efficientnet_b0":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            self.features = m.features
            self.head, self.relu_after = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 1)), False
        else:
            raise ValueError(f"unknown backbone {backbone}")
        for mod in self.modules():  # in-place ops break backward hooks
            if hasattr(mod, "inplace"):
                mod.inplace = False

    @property
    def target_layer(self) -> nn.Module:
        return self.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        if self.relu_after:
            f = F.relu(f)
        return self.head(F.adaptive_avg_pool2d(f, 1).flatten(1))  # (B, 1) logits

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.features.parameters():
            p.requires_grad = trainable


def load_classifier(path: str, device="cpu") -> tuple[CXRClassifier, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CXRClassifier(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt  # ckpt has: temperature, threshold, version
