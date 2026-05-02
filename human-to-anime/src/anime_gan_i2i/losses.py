from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids: List[int] | None = None):
        super().__init__()
        weights = None
        try:
            weights = VGG16_Weights.DEFAULT
        except Exception:
            weights = None
        try:
            features = vgg16(weights=weights).features
        except Exception:
            features = vgg16(weights=None).features
        self.backbone = features[:16].eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.criterion = nn.L1Loss()

    def _imagenet_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clamp(-1.0, 1.0).add(1.0).div(2.0).to(dtype=torch.float32, device=tensor.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self.backbone(self._imagenet_normalize(prediction))
        target_features = self.backbone(self._imagenet_normalize(target))
        return self.criterion(pred_features, target_features.detach())
