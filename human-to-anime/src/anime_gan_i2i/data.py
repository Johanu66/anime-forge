from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAIRED_MODELS = {"pix2pix", "anigan", "instruct_pix2pix"}
UNPAIRED_MODELS = {"cyclegan", "ugatit"}
MODEL_ALIASES = {
    "pix2pix": "pix2pix",
    "cycle_gan": "cyclegan",
    "cyclegan": "cyclegan",
    "u_gat_it": "ugatit",
    "u-gat-it": "ugatit",
    "ugatit": "ugatit",
    "ani_gan": "anigan",
    "anigan": "anigan",
    "instruct-pix2pix": "instruct_pix2pix",
    "instruct_pix2pix": "instruct_pix2pix",
}


def get_model_name(config: Dict) -> str:
    raw_name = config.get("model_name", config.get("mode"))
    if raw_name is None:
        raise ValueError("La configuration doit definir `model_name` ou `mode`.")
    normalized = str(raw_name).strip().lower().replace(" ", "_")
    return MODEL_ALIASES.get(normalized, normalized)


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Dossier introuvable: {folder}")
    files = [path for path in sorted(folder.iterdir()) if path.suffix.lower() in IMG_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"Aucune image detectee dans {folder}")
    return files


def _normalize_tensor():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def build_transforms(image_size: int, is_train: bool = False, augment_cfg: Dict | None = None):
    augment_cfg = augment_cfg or {}
    transform_steps: List[transforms.Compose] = []
    if is_train and augment_cfg.get("random_crop", False):
        # Preserve framing variety without aggressive geometric distortion.
        upscale_size = int(round(image_size * 1.12))
        transform_steps.append(transforms.Resize((upscale_size, upscale_size)))
        transform_steps.append(transforms.RandomCrop((image_size, image_size)))
    else:
        transform_steps.append(transforms.Resize((image_size, image_size)))
    if is_train and augment_cfg.get("random_horizontal_flip", False):
        transform_steps.append(transforms.RandomHorizontalFlip())
    if is_train and augment_cfg.get("color_jitter", 0.0) > 0:
        jitter = float(augment_cfg["color_jitter"])
        transform_steps.append(
            transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=min(0.5 * jitter, 0.2))
        )
    transform_steps.extend([transforms.ToTensor(), _normalize_tensor()])
    return transforms.Compose(transform_steps)


def _load_prompt_map(prompt_file: str | Path | None) -> Dict[str, str]:
    if not prompt_file:
        return {}
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de prompts introuvable: {path}")
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {str(key): str(value) for key, value in payload.items()}
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return {row["filename"]: row["prompt"] for row in reader}
    raise ValueError("Format de fichier de prompts non supporte. Utiliser .json ou .csv")


def _default_prompt_from_name(file_name: str) -> str:
    stem = Path(file_name).stem.replace("_", " ").replace("-", " ")
    if stem.strip():
        return f"convert this portrait to anime style while preserving identity: {stem}"
    return "convert this portrait to anime style while preserving identity"


class PairedImageDataset(Dataset):
    def __init__(
        self,
        real_dir: str | Path,
        anime_dir: str | Path,
        image_size: int,
        is_train: bool = False,
        augment_cfg: Dict | None = None,
    ) -> None:
        self.real_dir = Path(real_dir)
        self.anime_dir = Path(anime_dir)
        self.image_size = image_size
        self.is_train = is_train
        self.augment_cfg = augment_cfg or {}
        self.normalize = _normalize_tensor()
        real_images = list_images(self.real_dir)
        anime_images = list_images(self.anime_dir)
        anime_map = {path.name: path for path in anime_images}
        self.samples: List[Tuple[Path, Path]] = []
        for real_path in real_images:
            match = anime_map.get(real_path.name)
            if match is not None:
                self.samples.append((real_path, match))
        if not self.samples:
            raise ValueError("Aucune paire alignee detectee entre les dossiers real et anime.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        real_path, anime_path = self.samples[index]
        real_image = Image.open(real_path).convert("RGB")
        anime_image = Image.open(anime_path).convert("RGB")
        real_image, anime_image = self._transform_pair(real_image, anime_image)
        return {
            "input": real_image,
            "target": anime_image,
            "input_path": str(real_path),
            "target_path": str(anime_path),
        }

    def _transform_pair(self, real_image: Image.Image, anime_image: Image.Image) -> Tuple:
        real_image = real_image.resize((self.image_size, self.image_size), Image.BICUBIC)
        anime_image = anime_image.resize((self.image_size, self.image_size), Image.BICUBIC)

        if self.is_train and self.augment_cfg.get("random_horizontal_flip", False) and random.random() < 0.5:
            real_image = TF.hflip(real_image)
            anime_image = TF.hflip(anime_image)

        if self.is_train and self.augment_cfg.get("color_jitter", 0.0) > 0:
            jitter = float(self.augment_cfg["color_jitter"])
            brightness = 1.0 + random.uniform(-jitter, jitter)
            contrast = 1.0 + random.uniform(-jitter, jitter)
            saturation = 1.0 + random.uniform(-jitter, jitter)
            hue = random.uniform(-min(0.5 * jitter, 0.2), min(0.5 * jitter, 0.2))
            real_image = TF.adjust_brightness(real_image, brightness)
            real_image = TF.adjust_contrast(real_image, contrast)
            real_image = TF.adjust_saturation(real_image, saturation)
            real_image = TF.adjust_hue(real_image, hue)
            anime_image = TF.adjust_brightness(anime_image, brightness)
            anime_image = TF.adjust_contrast(anime_image, contrast)
            anime_image = TF.adjust_saturation(anime_image, saturation)
            anime_image = TF.adjust_hue(anime_image, hue)

        real_tensor = self.normalize(TF.to_tensor(real_image))
        anime_tensor = self.normalize(TF.to_tensor(anime_image))
        return real_tensor, anime_tensor


class TextPairedImageDataset(PairedImageDataset):
    def __init__(
        self,
        real_dir: str | Path,
        anime_dir: str | Path,
        image_size: int,
        prompt_file: str | Path | None = None,
        default_prompt: str | None = None,
        prompt_templates: List[str] | None = None,
        is_train: bool = False,
        augment_cfg: Dict | None = None,
    ) -> None:
        super().__init__(real_dir, anime_dir, image_size, is_train=is_train, augment_cfg=augment_cfg)
        self.prompt_map = _load_prompt_map(prompt_file)
        self.default_prompt = default_prompt
        self.prompt_templates = [str(template) for template in (prompt_templates or []) if str(template).strip()]

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        file_name = Path(sample["input_path"]).name
        template_prompt = random.choice(self.prompt_templates) if self.prompt_templates else None
        prompt = self.prompt_map.get(file_name) or template_prompt or self.default_prompt or _default_prompt_from_name(file_name)
        sample["text"] = prompt
        return sample


class UnpairedImageDataset(Dataset):
    def __init__(
        self,
        domain_a_dir: str | Path,
        domain_b_dir: str | Path,
        image_size: int,
        is_train: bool = False,
        augment_cfg: Dict | None = None,
        random_pairing: bool = True,
    ) -> None:
        self.domain_a = list_images(Path(domain_a_dir))
        self.domain_b = list_images(Path(domain_b_dir))
        self.transform = build_transforms(image_size, is_train=is_train, augment_cfg=augment_cfg)
        self.random_pairing = random_pairing

    def __len__(self) -> int:
        return max(len(self.domain_a), len(self.domain_b))

    def __getitem__(self, index: int):
        image_a = Image.open(self.domain_a[index % len(self.domain_a)]).convert("RGB")
        if self.random_pairing:
            image_b_path = self.domain_b[random.randrange(len(self.domain_b))]
        else:
            image_b_path = self.domain_b[index % len(self.domain_b)]
        image_b = Image.open(image_b_path).convert("RGB")
        return {
            "domain_a": self.transform(image_a),
            "domain_b": self.transform(image_b),
        }


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def create_dataloaders(config: Dict) -> Dict[str, DataLoader]:
    data_cfg = config["data"]
    root = Path(data_cfg["root"])
    image_size = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    augment_cfg = data_cfg.get("augment", {})
    model_name = get_model_name(config)
    loaders: Dict[str, DataLoader] = {}

    if model_name in PAIRED_MODELS:
        layout = data_cfg["paired_layout"]
        dataset_cls = PairedImageDataset
        dataset_kwargs = {}
        if model_name == "instruct_pix2pix":
            dataset_cls = TextPairedImageDataset
            text_cfg = config.get("text_encoder", {})
            dataset_kwargs = {
                "prompt_file": text_cfg.get("prompt_file"),
                "default_prompt": text_cfg.get("default_prompt"),
                "prompt_templates": text_cfg.get("prompt_templates", []),
            }
        train_dataset = dataset_cls(
            root / layout["train_real"],
            root / layout["train_anime"],
            image_size,
            is_train=True,
            augment_cfg=augment_cfg,
            **dataset_kwargs,
        )
        val_dataset = dataset_cls(root / layout["val_real"], root / layout["val_anime"], image_size, **dataset_kwargs)
        test_dataset = dataset_cls(root / layout["test_real"], root / layout["test_anime"], image_size, **dataset_kwargs)
        loaders["train"] = _loader(train_dataset, batch_size, True, num_workers)
        loaders["val"] = _loader(val_dataset, batch_size, False, num_workers)
        loaders["test"] = _loader(test_dataset, batch_size, False, num_workers)
        return loaders

    if model_name in UNPAIRED_MODELS:
        layout = data_cfg["unpaired_layout"]
        train_dataset = UnpairedImageDataset(
            root / layout["train_a"],
            root / layout["train_b"],
            image_size,
            is_train=True,
            augment_cfg=augment_cfg,
            random_pairing=True,
        )
        test_dataset = UnpairedImageDataset(
            root / layout["test_a"],
            root / layout["test_b"],
            image_size,
            is_train=False,
            augment_cfg=None,
            random_pairing=False,
        )
        loaders["train"] = _loader(train_dataset, batch_size, True, num_workers)
        loaders["test"] = _loader(test_dataset, batch_size, False, num_workers)
        return loaders

    raise ValueError(f"Mode non supporte: {model_name}")
