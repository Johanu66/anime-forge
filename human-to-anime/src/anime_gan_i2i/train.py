from __future__ import annotations

import argparse
import contextlib
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import load_config
from .data import create_dataloaders
from .losses import PerceptualLoss
from .metrics import paired_metrics
from .models import build_models
from .text import FrozenCLIPTextEncoder
from .utils import ensure_dir, history_to_dataframe, resolve_device, save_image_grid, set_seed


PAIRED_MODELS = {"pix2pix", "anigan", "instruct_pix2pix"}
UNPAIRED_MODELS = {"cyclegan", "ugatit"}

try:
    from torch.amp import GradScaler as _TorchGradScaler
except ImportError:
    from torch.cuda.amp import GradScaler as _TorchGradScaler


def _autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        if hasattr(torch, "autocast"):
            return torch.autocast(device_type="cuda", enabled=enabled)
        from torch.cuda.amp import autocast as cuda_autocast

        return cuda_autocast(enabled=enabled)
    return contextlib.nullcontext()


def save_checkpoint(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: torch.device) -> Dict:
    return torch.load(path, map_location=map_location)


def _model_name(config: Dict) -> str:
    return config.get("model_name", config.get("mode"))


def _criterion_gan_for(model_name: str):
    if model_name in {"cyclegan", "ugatit"}:
        return nn.MSELoss()
    return nn.BCEWithLogitsLoss()


def _extract_image(output):
    if isinstance(output, dict):
        return output["image"]
    return output


def _encode_texts_if_needed(text_encoder, batch: Dict, device: torch.device):
    if text_encoder is None:
        return None
    return text_encoder.encode(batch["text"], device)


def _generator_forward(model_name: str, generator, inputs: torch.Tensor, text_embedding: torch.Tensor | None = None):
    if model_name == "instruct_pix2pix":
        if text_embedding is None:
            raise ValueError("InstructPix2Pix requiert un embedding texte.")
        return generator(inputs, text_embedding)
    return generator(inputs)


def _paired_discriminator_forward(model_name: str, discriminator, image: torch.Tensor, condition: torch.Tensor, text_embedding=None):
    if model_name == "instruct_pix2pix":
        return discriminator(image, condition, text_embedding)
    return discriminator(image, condition)


def _evaluate_paired_generator(generator, loader, device: torch.device, model_name: str, text_encoder=None) -> Dict[str, float]:
    generator.eval()
    l1_loss = nn.L1Loss()
    metrics_sum = {"val_l1": 0.0, "psnr": 0.0, "ssim": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            text_embedding = _encode_texts_if_needed(text_encoder, batch, device)
            predictions = _extract_image(_generator_forward(model_name, generator, inputs, text_embedding))
            batch_metrics = paired_metrics(predictions, targets)
            metrics_sum["val_l1"] += float(l1_loss(predictions, targets).item())
            metrics_sum["psnr"] += batch_metrics["psnr"]
            metrics_sum["ssim"] += batch_metrics["ssim"]
            count += 1
    generator.train()
    return {key: value / max(count, 1) for key, value in metrics_sum.items()}


def _save_paired_samples(generator, loader, device: torch.device, sample_path: Path, model_name: str, text_encoder=None) -> None:
    batch = next(iter(loader))
    inputs = batch["input"][:4].to(device)
    targets = batch["target"][:4].to(device)
    text_embedding = None
    if text_encoder is not None:
        text_embedding = text_encoder.encode(batch["text"][:4], device)
    with torch.no_grad():
        predictions = _extract_image(_generator_forward(model_name, generator, inputs, text_embedding))
    save_image_grid([inputs, predictions, targets], sample_path, nrow=4)


def _save_unpaired_samples(generator_ab, generator_ba, loader, device: torch.device, sample_path: Path) -> None:
    batch = next(iter(loader))
    real_a = batch["domain_a"][:4].to(device)
    real_b = batch["domain_b"][:4].to(device)
    with torch.no_grad():
        fake_b = _extract_image(generator_ab(real_a))
        fake_a = _extract_image(generator_ba(real_b))
    save_image_grid([real_a, fake_b, real_b, fake_a], sample_path, nrow=4)


def _target_like(prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
    return torch.ones_like(prediction) if is_real else torch.zeros_like(prediction)


def _text_encoder_from_config(config: Dict):
    if _model_name(config) != "instruct_pix2pix":
        return None
    text_cfg = config.get("text_encoder", {})
    return FrozenCLIPTextEncoder(
        model_name=text_cfg.get("model_name", "openai/clip-vit-base-patch32"),
        embedding_dim=text_cfg.get("embedding_dim", 512),
        local_files_only=text_cfg.get("local_files_only", False),
    )


def _training_paths(config: Dict):
    runtime_cfg = config["runtime"]
    checkpoint_dir = ensure_dir(runtime_cfg["checkpoint_dir"])
    output_dir = ensure_dir(runtime_cfg["output_dir"])
    sample_dir = ensure_dir(output_dir / Path("samples"))
    return checkpoint_dir, output_dir, sample_dir


def _resolve_resume_path(config: Dict, resume_checkpoint: str | Path | None) -> str | Path | None:
    if resume_checkpoint is not None:
        return resume_checkpoint
    return config.get("runtime", {}).get("resume_checkpoint")


def _save_best_and_last(state: Dict, checkpoint_dir: Path, score: float, best_score: float, maximize: bool) -> float:
    save_checkpoint(state, checkpoint_dir / "last.pt")
    is_better = score > best_score if maximize else score < best_score
    if is_better:
        best_score = score
        best_state = dict(state)
        best_state["best_score"] = best_score
        save_checkpoint(best_state, checkpoint_dir / "best.pt")
    return best_score


def _decay_start_epoch(train_cfg: Dict) -> int:
    epochs = int(train_cfg["epochs"])
    default_start = max(1, epochs // 2)
    return int(train_cfg.get("decay_start_epoch", default_start))


def _build_linear_decay_scheduler(optimizer, total_epochs: int, decay_start_epoch: int):
    start = max(1, min(decay_start_epoch, total_epochs))

    def _lambda(epoch_index: int):
        epoch = epoch_index + 1
        if epoch <= start:
            return 1.0
        span = max(1, total_epochs - start)
        progress = min(epoch - start, span)
        return max(0.0, 1.0 - (progress / span))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lambda)


def _weight_decay(train_cfg: Dict) -> float:
    return float(train_cfg.get("weight_decay", 0.0))


@torch.no_grad()
def _evaluate_unpaired_generators(generator_ab, generator_ba, loader, device: torch.device) -> Dict[str, float]:
    generator_ab.eval()
    generator_ba.eval()
    l1 = nn.L1Loss()
    metrics = {"val_cycle_l1": 0.0, "val_identity_l1": 0.0}
    count = 0
    for batch in loader:
        real_a = batch["domain_a"].to(device)
        real_b = batch["domain_b"].to(device)
        fake_b = _extract_image(generator_ab(real_a))
        fake_a = _extract_image(generator_ba(real_b))
        cycle_a = _extract_image(generator_ba(fake_b))
        cycle_b = _extract_image(generator_ab(fake_a))
        id_a = _extract_image(generator_ba(real_a))
        id_b = _extract_image(generator_ab(real_b))
        metrics["val_cycle_l1"] += float((l1(cycle_a, real_a) + l1(cycle_b, real_b)).item())
        metrics["val_identity_l1"] += float((l1(id_a, real_a) + l1(id_b, real_b)).item())
        count += 1
    generator_ab.train()
    generator_ba.train()
    return {key: value / max(count, 1) for key, value in metrics.items()}


def _apply_text_dropout(texts: List[str], dropout_probability: float) -> List[str]:
    if dropout_probability <= 0.0:
        return texts
    return [text if random.random() >= dropout_probability else "" for text in texts]


def train_pix2pix(config: Dict, resume_checkpoint: str | Path | None = None):
    device = resolve_device()
    set_seed(config["seed"])
    loaders = create_dataloaders(config)
    generator, discriminator = build_models(config)
    generator.to(device)
    discriminator.to(device)

    train_cfg = config["train"]
    checkpoint_dir, output_dir, sample_dir = _training_paths(config)
    criterion_gan = _criterion_gan_for("pix2pix")
    criterion_l1 = nn.L1Loss()
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=_weight_decay(train_cfg),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=_weight_decay(train_cfg),
    )
    scheduler_g = _build_linear_decay_scheduler(optimizer_g, train_cfg["epochs"], _decay_start_epoch(train_cfg))
    scheduler_d = _build_linear_decay_scheduler(optimizer_d, train_cfg["epochs"], _decay_start_epoch(train_cfg))
    scaler = _TorchGradScaler(enabled=device.type == "cuda" and train_cfg.get("mixed_precision", True))

    history: List[Dict[str, float]] = []
    best_score = -math.inf
    start_epoch = 1
    resume_path = _resolve_resume_path(config, resume_checkpoint)
    if resume_path:
        checkpoint = load_checkpoint(resume_path, device)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        history = checkpoint.get("history", [])
        best_score = checkpoint.get("best_score", best_score)
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        progress = tqdm(loaders["train"], desc=f"Pix2Pix Epoch {epoch}/{train_cfg['epochs']}")
        epoch_metrics = {"epoch": float(epoch), "g_loss": 0.0, "d_loss": 0.0}

        for batch in progress:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            optimizer_g.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.get("mixed_precision", True)):
                fake_images = generator(inputs)
                pred_fake = discriminator(fake_images, inputs)
                real_labels = _target_like(pred_fake, True)
                fake_labels = _target_like(pred_fake, False)
                gan_loss = criterion_gan(pred_fake, real_labels)
                recon_loss = criterion_l1(fake_images, targets) * train_cfg["lambda_l1"]
                g_loss = gan_loss + recon_loss
            scaler.scale(g_loss).backward()
            scaler.step(optimizer_g)

            optimizer_d.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.get("mixed_precision", True)):
                pred_real = discriminator(targets, inputs)
                pred_fake_detached = discriminator(fake_images.detach(), inputs)
                d_loss = 0.5 * (
                    criterion_gan(pred_real, real_labels) + criterion_gan(pred_fake_detached, fake_labels)
                )
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)
            scaler.update()

            epoch_metrics["g_loss"] += float(g_loss.item())
            epoch_metrics["d_loss"] += float(d_loss.item())
            progress.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{d_loss.item():.4f}")

        batches = len(loaders["train"])
        epoch_metrics["g_loss"] /= batches
        epoch_metrics["d_loss"] /= batches
        val_metrics = _evaluate_paired_generator(generator, loaders["val"], device, "pix2pix")
        epoch_metrics.update(val_metrics)
        history.append(epoch_metrics)

        state = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "history": history,
            "config": config,
            "best_score": best_score,
        }
        best_score = _save_best_and_last(state, checkpoint_dir, val_metrics["ssim"] - val_metrics["val_l1"], best_score, True)

        if epoch % train_cfg["sample_interval"] == 0 or epoch == 1:
            _save_paired_samples(generator, loaders["val"], device, sample_dir / f"epoch_{epoch:03d}.png", "pix2pix")
        scheduler_g.step()
        scheduler_d.step()

    history_to_dataframe(history).to_csv(output_dir / "history.csv", index=False)
    return generator, history


def train_anigan(config: Dict, resume_checkpoint: str | Path | None = None):
    device = resolve_device()
    set_seed(config["seed"])
    loaders = create_dataloaders(config)
    generator, discriminator = build_models(config)
    generator.to(device)
    discriminator.to(device)

    train_cfg = config["train"]
    checkpoint_dir, output_dir, sample_dir = _training_paths(config)
    criterion_gan = _criterion_gan_for("anigan")
    criterion_l1 = nn.L1Loss()
    perceptual = PerceptualLoss().to(device)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=_weight_decay(train_cfg),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=_weight_decay(train_cfg),
    )
    scheduler_g = _build_linear_decay_scheduler(optimizer_g, train_cfg["epochs"], _decay_start_epoch(train_cfg))
    scheduler_d = _build_linear_decay_scheduler(optimizer_d, train_cfg["epochs"], _decay_start_epoch(train_cfg))
    scaler = _TorchGradScaler(enabled=device.type == "cuda" and train_cfg.get("mixed_precision", True))
    history: List[Dict[str, float]] = []
    best_score = -math.inf
    start_epoch = 1
    resume_path = _resolve_resume_path(config, resume_checkpoint)
    if resume_path:
        checkpoint = load_checkpoint(resume_path, device)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        history = checkpoint.get("history", [])
        best_score = checkpoint.get("best_score", best_score)
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        progress = tqdm(loaders["train"], desc=f"AniGAN Epoch {epoch}/{train_cfg['epochs']}")
        epoch_metrics = {"epoch": float(epoch), "g_loss": 0.0, "d_loss": 0.0}
        for batch in progress:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            optimizer_g.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.get("mixed_precision", True)):
                fake_images = generator(inputs)
                pred_fake = discriminator(fake_images, inputs)
                real_labels = _target_like(pred_fake, True)
                fake_labels = _target_like(pred_fake, False)
                gan_loss = criterion_gan(pred_fake, real_labels)
                l1_loss = criterion_l1(fake_images, targets) * train_cfg["lambda_l1"]
                perceptual_loss = perceptual(fake_images, targets) * train_cfg["lambda_perceptual"]
                identity_images = generator(targets)
                identity_loss = criterion_l1(identity_images, targets) * train_cfg["lambda_identity"]
                g_loss = gan_loss + l1_loss + perceptual_loss + identity_loss
            scaler.scale(g_loss).backward()
            scaler.step(optimizer_g)

            optimizer_d.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.get("mixed_precision", True)):
                pred_real = discriminator(targets, inputs)
                pred_fake_detached = discriminator(fake_images.detach(), inputs)
                d_loss = 0.5 * (
                    criterion_gan(pred_real, real_labels) + criterion_gan(pred_fake_detached, fake_labels)
                )
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)
            scaler.update()

            epoch_metrics["g_loss"] += float(g_loss.item())
            epoch_metrics["d_loss"] += float(d_loss.item())
            progress.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{d_loss.item():.4f}")

        for key in ("g_loss", "d_loss"):
            epoch_metrics[key] /= len(loaders["train"])
        val_metrics = _evaluate_paired_generator(generator, loaders["val"], device, "anigan")
        epoch_metrics.update(val_metrics)
        history.append(epoch_metrics)

        state = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "history": history,
            "config": config,
            "best_score": best_score,
        }
        best_score = _save_best_and_last(state, checkpoint_dir, val_metrics["ssim"] - val_metrics["val_l1"], best_score, True)

        if epoch % train_cfg["sample_interval"] == 0 or epoch == 1:
            _save_paired_samples(generator, loaders["val"], device, sample_dir / f"epoch_{epoch:03d}.png", "anigan")
        scheduler_g.step()
        scheduler_d.step()

    history_to_dataframe(history).to_csv(output_dir / "history.csv", index=False)
    return generator, history


def train_instruct_pix2pix(config: Dict, resume_checkpoint: str | Path | None = None):
    device = resolve_device()
    set_seed(config["seed"])
    loaders = create_dataloaders(config)
    generator, discriminator = build_models(config)
    text_encoder = _text_encoder_from_config(config)
    text_encoder.to(device)
    generator.to(device)
    discriminator.to(device)

    train_cfg = config["train"]
    checkpoint_dir, output_dir, sample_dir = _training_paths(config)
    criterion_gan = _criterion_gan_for("instruct_pix2pix")
    criterion_l1 = nn.L1Loss()
    perceptual = None
    if train_cfg.get("lambda_perceptual", 0.0) > 0:
        perceptual = PerceptualLoss().to(device)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=_weight_decay(train_cfg),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=_weight_decay(train_cfg),
    )
    scheduler_g = _build_linear_decay_scheduler(optimizer_g, train_cfg["epochs"], _decay_start_epoch(train_cfg))
    scheduler_d = _build_linear_decay_scheduler(optimizer_d, train_cfg["epochs"], _decay_start_epoch(train_cfg))
    scaler = _TorchGradScaler(enabled=device.type == "cuda" and train_cfg.get("mixed_precision", True))
    history: List[Dict[str, float]] = []
    best_score = -math.inf
    start_epoch = 1
    resume_path = _resolve_resume_path(config, resume_checkpoint)
    if resume_path:
        checkpoint = load_checkpoint(resume_path, device)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        history = checkpoint.get("history", [])
        best_score = checkpoint.get("best_score", best_score)
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        progress = tqdm(loaders["train"], desc=f"InstructPix2Pix Epoch {epoch}/{train_cfg['epochs']}")
        epoch_metrics = {"epoch": float(epoch), "g_loss": 0.0, "d_loss": 0.0}
        for batch in progress:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            text_dropout = float(train_cfg.get("text_dropout", 0.0))
            texts = _apply_text_dropout(list(batch["text"]), text_dropout)
            text_embedding = text_encoder.encode(texts, device)

            optimizer_g.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.get("mixed_precision", True)):
                fake_images = generator(inputs, text_embedding)
                pred_fake = discriminator(fake_images, inputs, text_embedding)
                real_labels = _target_like(pred_fake, True)
                fake_labels = _target_like(pred_fake, False)
                gan_loss = criterion_gan(pred_fake, real_labels)
                l1_loss = criterion_l1(fake_images, targets) * train_cfg["lambda_l1"]
                perceptual_loss = torch.tensor(0.0, device=device)
                if perceptual is not None:
                    perceptual_loss = perceptual(fake_images, targets) * train_cfg["lambda_perceptual"]
                g_loss = gan_loss + l1_loss + perceptual_loss
            scaler.scale(g_loss).backward()
            scaler.step(optimizer_g)

            optimizer_d.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.get("mixed_precision", True)):
                pred_real = discriminator(targets, inputs, text_embedding)
                pred_fake_detached = discriminator(fake_images.detach(), inputs, text_embedding)
                d_loss = 0.5 * (
                    criterion_gan(pred_real, real_labels) + criterion_gan(pred_fake_detached, fake_labels)
                )
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)
            scaler.update()

            epoch_metrics["g_loss"] += float(g_loss.item())
            epoch_metrics["d_loss"] += float(d_loss.item())
            progress.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{d_loss.item():.4f}")

        for key in ("g_loss", "d_loss"):
            epoch_metrics[key] /= len(loaders["train"])
        val_metrics = _evaluate_paired_generator(generator, loaders["val"], device, "instruct_pix2pix", text_encoder=text_encoder)
        epoch_metrics.update(val_metrics)
        history.append(epoch_metrics)

        state = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "history": history,
            "config": config,
            "best_score": best_score,
        }
        best_score = _save_best_and_last(state, checkpoint_dir, val_metrics["ssim"] - val_metrics["val_l1"], best_score, True)

        if epoch % train_cfg["sample_interval"] == 0 or epoch == 1:
            _save_paired_samples(
                generator,
                loaders["val"],
                device,
                sample_dir / f"epoch_{epoch:03d}.png",
                "instruct_pix2pix",
                text_encoder=text_encoder,
            )
        scheduler_g.step()
        scheduler_d.step()

    history_to_dataframe(history).to_csv(output_dir / "history.csv", index=False)
    return {"generator": generator, "text_encoder": text_encoder}, history


def _gan_loss_multiscale(outputs: Iterable[Dict[str, torch.Tensor]], criterion, is_real: bool) -> torch.Tensor:
    total = 0.0
    for output in outputs:
        total = total + criterion(output["logit"], _target_like(output["logit"], is_real))
        total = total + criterion(output["cam_logit"], _target_like(output["cam_logit"], is_real))
    return total


def train_cyclegan(config: Dict, resume_checkpoint: str | Path | None = None):
    device = resolve_device()
    set_seed(config["seed"])
    loaders = create_dataloaders(config)
    generator_ab, generator_ba, discriminator_a, discriminator_b = build_models(config)
    generator_ab.to(device)
    generator_ba.to(device)
    discriminator_a.to(device)
    discriminator_b.to(device)

    train_cfg = config["train"]
    checkpoint_dir, output_dir, sample_dir = _training_paths(config)
    criterion_gan = _criterion_gan_for("cyclegan")
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    optimizer_g = torch.optim.Adam(
        list(generator_ab.parameters()) + list(generator_ba.parameters()),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=_weight_decay(train_cfg),
    )
    optimizer_d_a = torch.optim.Adam(
        discriminator_a.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=_weight_decay(train_cfg),
    )
    optimizer_d_b = torch.optim.Adam(
        discriminator_b.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=_weight_decay(train_cfg),
    )
    scheduler_g = _build_linear_decay_scheduler(optimizer_g, train_cfg["epochs"], _decay_start_epoch(train_cfg))
    scheduler_d_a = _build_linear_decay_scheduler(optimizer_d_a, train_cfg["epochs"], _decay_start_epoch(train_cfg))
    scheduler_d_b = _build_linear_decay_scheduler(optimizer_d_b, train_cfg["epochs"], _decay_start_epoch(train_cfg))
    scaler = _TorchGradScaler(enabled=device.type == "cuda" and train_cfg.get("mixed_precision", True))
    history: List[Dict[str, float]] = []
    best_score = math.inf
    start_epoch = 1
    resume_path = _resolve_resume_path(config, resume_checkpoint)
    if resume_path:
        checkpoint = load_checkpoint(resume_path, device)
        generator_ab.load_state_dict(checkpoint["generator_ab"])
        generator_ba.load_state_dict(checkpoint["generator_ba"])
        discriminator_a.load_state_dict(checkpoint["discriminator_a"])
        discriminator_b.load_state_dict(checkpoint["discriminator_b"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d_a.load_state_dict(checkpoint["optimizer_d_a"])
        optimizer_d_b.load_state_dict(checkpoint["optimizer_d_b"])
        history = checkpoint.get("history", [])
        best_score = checkpoint.get("best_score", best_score)
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        progress = tqdm(loaders["train"], desc=f"CycleGAN Epoch {epoch}/{train_cfg['epochs']}")
        epoch_metrics = {"epoch": float(epoch), "g_loss": 0.0, "d_a_loss": 0.0, "d_b_loss": 0.0}

        for batch in progress:
            real_a = batch["domain_a"].to(device)
            real_b = batch["domain_b"].to(device)

            optimizer_g.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.get("mixed_precision", True)):
                identity_a = generator_ba(real_a)
                identity_b = generator_ab(real_b)
                loss_identity = (
                    criterion_identity(identity_a, real_a) + criterion_identity(identity_b, real_b)
                ) * train_cfg["lambda_identity"]

                fake_b = generator_ab(real_a)
                fake_a = generator_ba(real_b)
                loss_gan = criterion_gan(discriminator_b(fake_b), _target_like(discriminator_b(fake_b), True))
                loss_gan = loss_gan + criterion_gan(discriminator_a(fake_a), _target_like(discriminator_a(fake_a), True))

                recovered_a = generator_ba(fake_b)
                recovered_b = generator_ab(fake_a)
                loss_cycle = (
                    criterion_cycle(recovered_a, real_a) + criterion_cycle(recovered_b, real_b)
                ) * train_cfg["lambda_cycle"]
                g_loss = loss_identity + loss_gan + loss_cycle
            scaler.scale(g_loss).backward()
            scaler.step(optimizer_g)

            optimizer_d_a.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.get("mixed_precision", True)):
                pred_real_a = discriminator_a(real_a)
                pred_fake_a = discriminator_a(fake_a.detach())
                d_a_loss = 0.5 * (
                    criterion_gan(pred_real_a, _target_like(pred_real_a, True))
                    + criterion_gan(pred_fake_a, _target_like(pred_fake_a, False))
                )
            scaler.scale(d_a_loss).backward()
            scaler.step(optimizer_d_a)

            optimizer_d_b.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.get("mixed_precision", True)):
                pred_real_b = discriminator_b(real_b)
                pred_fake_b = discriminator_b(fake_b.detach())
                d_b_loss = 0.5 * (
                    criterion_gan(pred_real_b, _target_like(pred_real_b, True))
                    + criterion_gan(pred_fake_b, _target_like(pred_fake_b, False))
                )
            scaler.scale(d_b_loss).backward()
            scaler.step(optimizer_d_b)
            scaler.update()

            epoch_metrics["g_loss"] += float(g_loss.item())
            epoch_metrics["d_a_loss"] += float(d_a_loss.item())
            epoch_metrics["d_b_loss"] += float(d_b_loss.item())
            progress.set_postfix(g_loss=f"{g_loss.item():.4f}", d_a=f"{d_a_loss.item():.4f}", d_b=f"{d_b_loss.item():.4f}")

        for key in ("g_loss", "d_a_loss", "d_b_loss"):
            epoch_metrics[key] /= len(loaders["train"])
        val_metrics = _evaluate_unpaired_generators(generator_ab, generator_ba, loaders["test"], device)
        epoch_metrics.update(val_metrics)
        history.append(epoch_metrics)

        state = {
            "epoch": epoch,
            "generator_ab": generator_ab.state_dict(),
            "generator_ba": generator_ba.state_dict(),
            "discriminator_a": discriminator_a.state_dict(),
            "discriminator_b": discriminator_b.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d_a": optimizer_d_a.state_dict(),
            "optimizer_d_b": optimizer_d_b.state_dict(),
            "history": history,
            "config": config,
            "best_score": best_score,
        }
        proxy_score = epoch_metrics["val_cycle_l1"] + (0.5 * epoch_metrics["val_identity_l1"])
        best_score = _save_best_and_last(state, checkpoint_dir, proxy_score, best_score, False)

        if epoch == 1 or epoch % train_cfg["sample_interval"] == 0:
            _save_unpaired_samples(generator_ab, generator_ba, loaders["test"], device, sample_dir / f"epoch_{epoch:03d}.png")
        scheduler_g.step()
        scheduler_d_a.step()
        scheduler_d_b.step()

    history_to_dataframe(history).to_csv(output_dir / "history.csv", index=False)
    return {"generator_ab": generator_ab, "generator_ba": generator_ba}, history


def train_ugatit(config: Dict, resume_checkpoint: str | Path | None = None):
    device = resolve_device()
    set_seed(config["seed"])
    loaders = create_dataloaders(config)
    generator_ab, generator_ba, discriminator_a, discriminator_b = build_models(config)
    generator_ab.to(device)
    generator_ba.to(device)
    discriminator_a.to(device)
    discriminator_b.to(device)

    train_cfg = config["train"]
    checkpoint_dir, output_dir, sample_dir = _training_paths(config)
    criterion_gan = _criterion_gan_for("ugatit")
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_cam = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(
        list(generator_ab.parameters()) + list(generator_ba.parameters()),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=_weight_decay(train_cfg),
    )
    optimizer_d_a = torch.optim.Adam(
        discriminator_a.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=_weight_decay(train_cfg),
    )
    optimizer_d_b = torch.optim.Adam(
        discriminator_b.parameters(),
        lr=train_cfg["lr"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=_weight_decay(train_cfg),
    )
    scheduler_g = _build_linear_decay_scheduler(optimizer_g, train_cfg["epochs"], _decay_start_epoch(train_cfg))
    scheduler_d_a = _build_linear_decay_scheduler(optimizer_d_a, train_cfg["epochs"], _decay_start_epoch(train_cfg))
    scheduler_d_b = _build_linear_decay_scheduler(optimizer_d_b, train_cfg["epochs"], _decay_start_epoch(train_cfg))
    scaler = _TorchGradScaler(enabled=device.type == "cuda" and train_cfg.get("mixed_precision", True))
    history: List[Dict[str, float]] = []
    best_score = math.inf
    start_epoch = 1
    resume_path = _resolve_resume_path(config, resume_checkpoint)
    if resume_path:
        checkpoint = load_checkpoint(resume_path, device)
        generator_ab.load_state_dict(checkpoint["generator_ab"])
        generator_ba.load_state_dict(checkpoint["generator_ba"])
        discriminator_a.load_state_dict(checkpoint["discriminator_a"])
        discriminator_b.load_state_dict(checkpoint["discriminator_b"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d_a.load_state_dict(checkpoint["optimizer_d_a"])
        optimizer_d_b.load_state_dict(checkpoint["optimizer_d_b"])
        history = checkpoint.get("history", [])
        best_score = checkpoint.get("best_score", best_score)
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        progress = tqdm(loaders["train"], desc=f"UGATIT Epoch {epoch}/{train_cfg['epochs']}")
        epoch_metrics = {"epoch": float(epoch), "g_loss": 0.0, "d_a_loss": 0.0, "d_b_loss": 0.0}

        for batch in progress:
            real_a = batch["domain_a"].to(device)
            real_b = batch["domain_b"].to(device)

            optimizer_g.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.get("mixed_precision", True)):
                id_a = generator_ba(real_a)
                id_b = generator_ab(real_b)
                fake_b = generator_ab(real_a)
                fake_a = generator_ba(real_b)
                cycle_a = generator_ba(fake_b["image"])
                cycle_b = generator_ab(fake_a["image"])

                adv_loss = _gan_loss_multiscale(discriminator_b(fake_b["image"]), criterion_gan, True)
                adv_loss = adv_loss + _gan_loss_multiscale(discriminator_a(fake_a["image"]), criterion_gan, True)
                cycle_loss = (
                    criterion_cycle(cycle_a["image"], real_a) + criterion_cycle(cycle_b["image"], real_b)
                ) * train_cfg["lambda_cycle"]
                identity_loss = (
                    criterion_identity(id_a["image"], real_a) + criterion_identity(id_b["image"], real_b)
                ) * train_cfg["lambda_identity"]
                cam_loss = (
                    criterion_cam(fake_b["cam_logit"], torch.ones_like(fake_b["cam_logit"]))
                    + criterion_cam(fake_a["cam_logit"], torch.ones_like(fake_a["cam_logit"]))
                    + criterion_cam(id_a["cam_logit"], torch.zeros_like(id_a["cam_logit"]))
                    + criterion_cam(id_b["cam_logit"], torch.zeros_like(id_b["cam_logit"]))
                ) * train_cfg["lambda_cam"]
                g_loss = adv_loss + cycle_loss + identity_loss + cam_loss
            scaler.scale(g_loss).backward()
            scaler.step(optimizer_g)

            optimizer_d_a.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.get("mixed_precision", True)):
                d_a_loss = 0.5 * (
                    _gan_loss_multiscale(discriminator_a(real_a), criterion_gan, True)
                    + _gan_loss_multiscale(discriminator_a(fake_a["image"].detach()), criterion_gan, False)
                )
            scaler.scale(d_a_loss).backward()
            scaler.step(optimizer_d_a)

            optimizer_d_b.zero_grad(set_to_none=True)
            with _autocast_context(device, train_cfg.get("mixed_precision", True)):
                d_b_loss = 0.5 * (
                    _gan_loss_multiscale(discriminator_b(real_b), criterion_gan, True)
                    + _gan_loss_multiscale(discriminator_b(fake_b["image"].detach()), criterion_gan, False)
                )
            scaler.scale(d_b_loss).backward()
            scaler.step(optimizer_d_b)
            scaler.update()

            epoch_metrics["g_loss"] += float(g_loss.item())
            epoch_metrics["d_a_loss"] += float(d_a_loss.item())
            epoch_metrics["d_b_loss"] += float(d_b_loss.item())
            progress.set_postfix(g_loss=f"{g_loss.item():.4f}", d_a=f"{d_a_loss.item():.4f}", d_b=f"{d_b_loss.item():.4f}")

        for key in ("g_loss", "d_a_loss", "d_b_loss"):
            epoch_metrics[key] /= len(loaders["train"])
        val_metrics = _evaluate_unpaired_generators(generator_ab, generator_ba, loaders["test"], device)
        epoch_metrics.update(val_metrics)
        history.append(epoch_metrics)

        state = {
            "epoch": epoch,
            "generator_ab": generator_ab.state_dict(),
            "generator_ba": generator_ba.state_dict(),
            "discriminator_a": discriminator_a.state_dict(),
            "discriminator_b": discriminator_b.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d_a": optimizer_d_a.state_dict(),
            "optimizer_d_b": optimizer_d_b.state_dict(),
            "history": history,
            "config": config,
            "best_score": best_score,
        }
        proxy_score = epoch_metrics["val_cycle_l1"] + (0.5 * epoch_metrics["val_identity_l1"])
        best_score = _save_best_and_last(state, checkpoint_dir, proxy_score, best_score, False)

        if epoch == 1 or epoch % train_cfg["sample_interval"] == 0:
            _save_unpaired_samples(generator_ab, generator_ba, loaders["test"], device, sample_dir / f"epoch_{epoch:03d}.png")
        scheduler_g.step()
        scheduler_d_a.step()
        scheduler_d_b.step()

    history_to_dataframe(history).to_csv(output_dir / "history.csv", index=False)
    return {"generator_ab": generator_ab, "generator_ba": generator_ba}, history


TRAINERS = {
    "pix2pix": train_pix2pix,
    "cyclegan": train_cyclegan,
    "ugatit": train_ugatit,
    "anigan": train_anigan,
    "instruct_pix2pix": train_instruct_pix2pix,
}


def run_training(config: Dict, resume_checkpoint: str | Path | None = None):
    model_name = _model_name(config)
    if model_name not in TRAINERS:
        raise ValueError(f"Mode d'entrainement non supporte: {model_name}")
    return TRAINERS[model_name](config, resume_checkpoint=resume_checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrainement image-vers-image anime")
    parser.add_argument("--config", required=True, help="Chemin vers le fichier YAML de configuration")
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Checkpoint à reprendre (ex: checkpoints/<model>/last.pt).",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_training(config, resume_checkpoint=args.resume_checkpoint)


if __name__ == "__main__":
    main()
