from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT_CANDIDATES = [
    MODEL_DIR / "checkpoint_epoch_80.pth.zip",
    MODEL_DIR / "checkpoint_epoch_80.pth",
    MODEL_DIR / "best.pt",
]

_DEVICE: torch.device | None = None
_GENERATOR: nn.Module | None = None
_LATENT_DIM: int | None = None


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_checkpoint() -> Path:
    for candidate in CHECKPOINT_CANDIDATES:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No random-anime checkpoint file found. Expected one of: "
        + ", ".join(str(path) for path in CHECKPOINT_CANDIDATES)
    )


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0).add(1.0).div(2.0)
    array = tensor.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _load_generator() -> tuple[nn.Module, torch.device, int]:
    global _DEVICE, _GENERATOR, _LATENT_DIM

    if _DEVICE is not None and _GENERATOR is not None and _LATENT_DIM is not None:
        return _GENERATOR, _DEVICE, _LATENT_DIM

    device = _resolve_device()
    checkpoint_path = _find_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict) or "generator_state" not in checkpoint:
        raise ValueError("Checkpoint format unsupported: missing `generator_state`.")

    generator_state = checkpoint["generator_state"]
    if not isinstance(generator_state, dict) or "net.0.weight" not in generator_state:
        raise ValueError("Checkpoint generator_state is invalid.")

    latent_dim = int(generator_state["net.0.weight"].shape[0])
    generator = DCGANGenerator(latent_dim=latent_dim).to(device)
    generator.load_state_dict(generator_state, strict=True)
    generator.eval()

    _DEVICE = device
    _GENERATOR = generator
    _LATENT_DIM = latent_dim
    return generator, device, latent_dim


def generate(output_path: str) -> str:
    generator, device, latent_dim = _load_generator()
    seed = torch.seed()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    noise = torch.randn(1, latent_dim, 1, 1, generator=rng, dtype=torch.float32).to(device)
    with torch.no_grad():
        fake = generator(noise)[0]

    image = _tensor_to_image(fake)

    target_size = int(os.environ.get("ANIME_FACE_RANDOM_IMAGE_SIZE", "512"))
    if target_size > 0 and image.width != target_size:
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return str(output)
