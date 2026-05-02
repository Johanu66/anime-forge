from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms

CHECKPOINT_PATH = Path(__file__).resolve().parent / "best.pt"
IMAGE_SIZE = int(os.environ.get("ANIME_FACE_FACE_MODEL_IMAGE_SIZE", "128"))

_GENERATOR: nn.Module | None = None
_DEVICE: torch.device | None = None


class AdaLIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.rho = nn.Parameter(torch.full((1, num_features, 1, 1), 0.9))
        self.eps = eps

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        in_mean = torch.mean(x, dim=(2, 3), keepdim=True)
        in_var = torch.var(x, dim=(2, 3), keepdim=True, unbiased=False)
        ln_mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        ln_var = torch.var(x, dim=(1, 2, 3), keepdim=True, unbiased=False)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        rho = self.rho.expand(x.size(0), -1, -1, -1).clamp(0.0, 1.0)
        out = rho * out_in + (1.0 - rho) * out_ln
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return out * gamma + beta


class CAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gap_fc = nn.Linear(channels, 1, bias=False)
        self.gmp_fc = nn.Linear(channels, 1, bias=False)
        self.conv1x1 = nn.Conv2d(channels * 2, channels, 1, bias=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.size(0), -1))
        gmp_logit = self.gmp_fc(gmp.view(x.size(0), -1))

        gap_weight = self.gap_fc.weight.unsqueeze(-1).unsqueeze(-1)
        gmp_weight = self.gmp_fc.weight.unsqueeze(-1).unsqueeze(-1)
        gap_features = x * gap_weight
        gmp_features = x * gmp_weight

        cam_logit = torch.cat((gap_logit, gmp_logit), dim=1)
        features = torch.cat((gap_features, gmp_features), dim=1)
        features = self.activation(self.conv1x1(features))
        heatmap = torch.sum(features, dim=1, keepdim=True)
        return features, cam_logit, heatmap


class AdaLINResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, 3, bias=False)
        self.norm1 = AdaLIN(channels)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(channels, channels, 3, bias=False)
        self.norm2 = AdaLIN(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.pad1(x))
        out = self.norm1(out, gamma, beta)
        out = self.activation(out)
        out = self.conv2(self.pad2(out))
        out = self.norm2(out, gamma, beta)
        return out + x


class UGATITGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_blocks: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.cam = CAM(256)
        self.gamma_beta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
        )
        self.resblocks = nn.ModuleList([AdaLINResBlock(256) for _ in range(num_blocks)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(x)
        features, cam_logit, heatmap = self.cam(features)
        gamma_beta = self.gamma_beta(features)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        for block in self.resblocks:
            features = block(features, gamma, beta)
        output = self.decoder(features)
        return {"image": output, "cam_logit": cam_logit, "heatmap": heatmap}


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clamp(-1.0, 1.0).add(1.0).div(2.0)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def _extract_generator_state(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    if "generator_ab" in checkpoint:
        state = checkpoint["generator_ab"]
    elif "generator" in checkpoint:
        state = checkpoint["generator"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise ValueError("Checkpoint invalid: generator state dict not found.")

    cleaned_state = {}
    for key, value in state.items():
        if key.startswith("module."):
            cleaned_state[key[7:]] = value
        else:
            cleaned_state[key] = value
    return cleaned_state


def _infer_num_resblocks(state_dict: Dict[str, torch.Tensor]) -> int:
    indices = []
    for key in state_dict:
        if key.startswith("resblocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                indices.append(int(parts[1]))
    return (max(indices) + 1) if indices else 4


def _load_generator() -> tuple[nn.Module, torch.device]:
    global _GENERATOR, _DEVICE

    if _GENERATOR is not None and _DEVICE is not None:
        return _GENERATOR, _DEVICE

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = _resolve_device()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint format unsupported.")

    state_dict = _extract_generator_state(checkpoint)
    num_blocks = _infer_num_resblocks(state_dict)

    model = UGATITGenerator(num_blocks=num_blocks).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _GENERATOR = model
    _DEVICE = device
    return _GENERATOR, _DEVICE


def transform(image_path: str, output_path: str) -> str:
    model, device = _load_generator()
    transform_pipeline = _build_transform()

    image = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
    tensor = transform_pipeline(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor)
        if isinstance(prediction, dict):
            prediction = prediction["image"]

    output_tensor = _denormalize(prediction)[0]
    output_image = _tensor_to_pil(output_tensor)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)
    return output_path
