from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from PIL import Image

from .config import load_config
from .data import build_transforms, list_images
from .models import build_models
from .text import FrozenCLIPTextEncoder
from .train import load_checkpoint
from .utils import denormalize, ensure_dir, resolve_device


def _to_pil_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def _load_inference_bundle(config: Dict, checkpoint_path: str | Path, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_name = config["model_name"]

    if model_name in {"pix2pix", "anigan", "instruct_pix2pix"}:
        generator, _ = build_models(config)
        generator.load_state_dict(checkpoint["generator"])
        generator = generator.to(device).eval()
        if model_name == "instruct_pix2pix":
            text_cfg = config.get("text_encoder", {})
            text_encoder = FrozenCLIPTextEncoder(
                model_name=text_cfg.get("model_name", "openai/clip-vit-base-patch32"),
                embedding_dim=text_cfg.get("embedding_dim", 512),
                local_files_only=text_cfg.get("local_files_only", False),
            ).to(device)
            return {"generator": generator, "text_encoder": text_encoder}
        return {"generator": generator}

    if model_name in {"cyclegan", "ugatit"}:
        generator_ab, _, _, _ = build_models(config)
        generator_ab.load_state_dict(checkpoint["generator_ab"])
        return {"generator": generator_ab.to(device).eval()}

    raise ValueError(f"Mode non supporte: {model_name}")


@torch.no_grad()
def run_inference(
    config: Dict,
    checkpoint_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    prompt: str | None = None,
) -> Path:
    device = resolve_device()
    output_path = ensure_dir(output_dir)
    transform = build_transforms(config["data"]["image_size"])
    bundle = _load_inference_bundle(config, checkpoint_path, device)
    generator = bundle["generator"]
    model_name = config["model_name"]

    for image_path in list_images(Path(input_dir)):
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        if model_name == "instruct_pix2pix":
            text_encoder = bundle["text_encoder"]
            text_prompt = prompt or config.get("text_encoder", {}).get("default_prompt") or "convert portrait to anime"
            text_embedding = text_encoder.encode([text_prompt], device)
            prediction = generator(tensor, text_embedding)
        else:
            prediction = generator(tensor)
            if isinstance(prediction, dict):
                prediction = prediction["image"]
        result = denormalize(prediction)[0]
        _to_pil_image(result).save(output_path / image_path.name)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference image-vers-image anime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_inference(config, args.checkpoint, args.input_dir, args.output_dir, prompt=args.prompt)


if __name__ == "__main__":
    main()
