from __future__ import annotations

from typing import Dict

import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .utils import denormalize


@torch.no_grad()
def paired_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    preds = denormalize(predictions).permute(0, 2, 3, 1).numpy()
    refs = denormalize(targets).permute(0, 2, 3, 1).numpy()
    psnr_values = []
    ssim_values = []
    for pred, ref in zip(preds, refs):
        psnr_values.append(peak_signal_noise_ratio(ref, pred, data_range=1.0))
        ssim_values.append(structural_similarity(ref, pred, channel_axis=2, data_range=1.0))
    return {
        "psnr": float(sum(psnr_values) / len(psnr_values)),
        "ssim": float(sum(ssim_values) / len(ssim_values)),
    }
