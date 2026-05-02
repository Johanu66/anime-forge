from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

from .utils import ensure_dir, set_seed


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(folder: Path) -> List[Path]:
    return [path for path in sorted(folder.iterdir()) if path.suffix.lower() in IMG_EXTENSIONS]


def _copy_pairs(pairs: Iterable[Tuple[Path, Path]], destination: Path, split_name: str) -> None:
    real_dir = ensure_dir(destination / split_name / "real")
    anime_dir = ensure_dir(destination / split_name / "anime")
    for real_path, anime_path in pairs:
        shutil.copy2(real_path, real_dir / real_path.name)
        shutil.copy2(anime_path, anime_dir / anime_path.name)


def prepare_paired_dataset(
    source_real: str | Path,
    source_anime: str | Path,
    destination_root: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    set_seed(seed)
    source_real = Path(source_real)
    source_anime = Path(source_anime)
    destination_root = Path(destination_root)

    real_images = _list_images(source_real)
    anime_map = {path.name: path for path in _list_images(source_anime)}
    pairs = [(real_path, anime_map[real_path.name]) for real_path in real_images if real_path.name in anime_map]
    if not pairs:
        raise ValueError("Aucune paire matchee par nom de fichier.")

    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    _copy_pairs(pairs[:train_end], destination_root, "train")
    _copy_pairs(pairs[train_end:val_end], destination_root, "val")
    _copy_pairs(pairs[val_end:], destination_root, "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparation dataset paire pour Pix2Pix")
    parser.add_argument("--source-real", required=True)
    parser.add_argument("--source-anime", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_paired_dataset(
        source_real=args.source_real,
        source_anime=args.source_anime,
        destination_root=args.destination_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
