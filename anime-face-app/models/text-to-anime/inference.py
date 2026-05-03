from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

MODEL_DIR = Path(__file__).resolve().parent
LORA_DIR = MODEL_DIR / "lora_weights_local"
LORA_MODEL_FILE = LORA_DIR / "adapter_model.safetensors"
LORA_CONFIG_FILE = LORA_DIR / "adapter_config.json"
DEFAULT_LOCAL_BASE_DIR = MODEL_DIR / "base_model_local"

BASE_MODEL_ID = os.environ.get("ANIME_FACE_TEXT_BASE_MODEL", "runwayml/stable-diffusion-v1-5")
# Streamlit-like default behavior: allow first-time download if base model
# is not already cached locally. Set ANIME_FACE_TEXT_LOCAL_ONLY=1 to disable.
LOCAL_FILES_ONLY = os.environ.get("ANIME_FACE_TEXT_LOCAL_ONLY", "0") != "0"
NUM_INFERENCE_STEPS = int(os.environ.get("ANIME_FACE_TEXT_STEPS", "30"))
GUIDANCE_SCALE = float(os.environ.get("ANIME_FACE_TEXT_GUIDANCE", "7.5"))
MAX_PROMPT_LENGTH = int(os.environ.get("ANIME_FACE_TEXT_MAX_PROMPT_LENGTH", "400"))


def _resolve_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _validate_weights() -> None:
    if not LORA_MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing LoRA weights file: {LORA_MODEL_FILE}")
    if not LORA_CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing LoRA config file: {LORA_CONFIG_FILE}")
    if not LORA_DIR.is_dir():
        raise FileNotFoundError(f"Missing LoRA directory: {LORA_DIR}")


def _resolve_base_model_source() -> tuple[str | Path, bool]:
    # Priority 1: explicit override (can be local folder or single checkpoint file).
    base_model_override = os.environ.get("ANIME_FACE_TEXT_BASE_MODEL", "").strip()
    if base_model_override:
        candidate = Path(base_model_override)
        if candidate.exists():
            if candidate.is_file():
                return candidate, True
            return candidate, False
        return base_model_override, False

    # Priority 2: conventional local folder inside this app.
    if (DEFAULT_LOCAL_BASE_DIR / "model_index.json").exists():
        return DEFAULT_LOCAL_BASE_DIR, False

    # Priority 3: default HF model id (still constrained by local_files_only by default).
    return BASE_MODEL_ID, False


@lru_cache(maxsize=1)
def load_model() -> StableDiffusionPipeline:
    _validate_weights()
    device, dtype = _resolve_device()

    base_model_source, is_single_file = _resolve_base_model_source()

    try:
        if is_single_file:
            pipe = StableDiffusionPipeline.from_single_file(
                str(base_model_source),
                torch_dtype=dtype,
                safety_checker=None,
                local_files_only=LOCAL_FILES_ONLY,
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                base_model_source,
                torch_dtype=dtype,
                safety_checker=None,
                local_files_only=LOCAL_FILES_ONLY,
            )
    except OSError as exc:
        if LOCAL_FILES_ONLY:
            raise RuntimeError(
                "LoRA loaded but base Stable Diffusion model is missing or incomplete locally. "
                "Place a full base model locally and set ANIME_FACE_TEXT_BASE_MODEL "
                "(directory with model_index.json or single .safetensors/.ckpt file), "
                "or put it in models/text-to-anime/base_model_local. "
                "Download is disabled by ANIME_FACE_TEXT_LOCAL_ONLY=1."
            ) from exc
        raise

    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(LORA_DIR))
    pipe = pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    return pipe


def generate(prompt: str, output_path: str) -> str:
    text_prompt = (prompt or "").strip()
    if not text_prompt:
        raise ValueError("Prompt cannot be empty.")

    if len(text_prompt) > MAX_PROMPT_LENGTH:
        text_prompt = text_prompt[:MAX_PROMPT_LENGTH]

    pipe = load_model()
    seed_env = os.environ.get("ANIME_FACE_TEXT_SEED")
    generator = None

    if seed_env:
        seed = int(seed_env)
        generator = torch.Generator(device=pipe.device.type)
        generator.manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            text_prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )

    image = result.images[0]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return str(destination)
