from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clamp(-1.0, 1.0).add(1.0).div(2.0)


def save_image_grid(
    images: Iterable[torch.Tensor],
    path: str | Path,
    nrow: int = 4,
    title: str | None = None,
) -> None:
    tensors = [denormalize(image) for image in images]
    grid = make_grid(torch.cat(tensors, dim=0), nrow=nrow)
    save_image(grid, path)
    if title:
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        plt.title(title)
        plt.axis("off")
        plt.show()


def history_to_dataframe(history: List[Dict[str, float]]):
    import pandas as pd

    return pd.DataFrame(history)
