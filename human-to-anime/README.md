# Fonctionnalite 2 - Image vers Image Anime

Cette base implemente la fonctionnalite image-vers-image du projet GAN Anime avec cinq modes :

- `pix2pix` pour des donnees pairees/alignees
- `cyclegan` pour des donnees non pairees (`trainA/trainB/testA/testB`)
- `ugatit` pour une traduction non pairee avec CAM, AdaLIN et discriminateurs multi-echelles
- `anigan` pour une version orientee preservation de structure avec loss perceptuelle
- `instruct_pix2pix` pour une traduction conditionnee par prompt texte via encodeur CLIP gele

Le code est organise pour que le notebook reste lisible tout en s'appuyant sur des modules PyTorch reutilisables.

## Structure

- `notebooks/feature2_image_to_image_anime.ipynb` : notebook principal
- `src/anime_gan_i2i/` : datasets, modeles, entrainement, inference, utilitaires
- `configs/pix2pix_selfie2anime.yaml` : configuration Pix2Pix
- `configs/cyclegan_selfie2anime.yaml` : configuration CycleGAN
- `configs/ugatit_selfie2anime.yaml` : configuration U-GAT-IT
- `configs/anigan_selfie2anime.yaml` : configuration AniGAN simplifie
- `configs/instruct_pix2pix.yaml` : configuration InstructPix2Pix GAN
- `tests/smoke_test.py` : verification rapide hors GPU

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset attendu

### Option A - Pix2Pix (paires alignees)

Format recommande :

```text
data/selfie2anime/
  train/
    real/
    anime/
  val/
    real/
    anime/
  test/
    real/
    anime/
```

Les noms de fichiers doivent correspondre entre `real` et `anime`.

### Option B - CycleGAN (non paire)

Format classique :

```text
data/selfie2anime/
  trainA/
  trainB/
  testA/
  testB/
```

- `A` = domaine photo reelle
- `B` = domaine anime

## Execution rapide

### Notebook

Ouvrir puis executer :

```bash
jupyter notebook notebooks/feature2_image_to_image_anime.ipynb
```

### Entrainement en script

```bash
PYTHONPATH=src python -m anime_gan_i2i.prepare_data \
  --source-real data/raw/real \
  --source-anime data/raw/anime \
  --destination-root data/selfie2anime

PYTHONPATH=src python -m anime_gan_i2i.train --config configs/pix2pix_selfie2anime.yaml
PYTHONPATH=src python -m anime_gan_i2i.train --config configs/cyclegan_selfie2anime.yaml
PYTHONPATH=src python -m anime_gan_i2i.train --config configs/ugatit_selfie2anime.yaml
PYTHONPATH=src python -m anime_gan_i2i.train --config configs/anigan_selfie2anime.yaml
PYTHONPATH=src python -m anime_gan_i2i.train --config configs/instruct_pix2pix.yaml
```

### Inference

```bash
PYTHONPATH=src python -m anime_gan_i2i.infer \
  --config configs/pix2pix_selfie2anime.yaml \
  --checkpoint checkpoints/pix2pix/best.pt \
  --input-dir data/selfie2anime/test/real \
  --output-dir outputs/inference_pix2pix

PYTHONPATH=src python -m anime_gan_i2i.infer \
  --config configs/instruct_pix2pix.yaml \
  --checkpoint checkpoints/instruct_pix2pix/best.pt \
  --input-dir data/selfie2anime/test/real \
  --output-dir outputs/inference_instruct \
  --prompt "convert this portrait into soft anime style"
```

## Sorties

- `checkpoints/<mode>/last.pt` : dernier checkpoint
- `checkpoints/<mode>/best.pt` : meilleur checkpoint
- `outputs/<mode>/samples/` : grilles d'images au fil des epochs
- `outputs/<mode>/history.csv` : historique des losses et metriques

## Validation locale

```bash
PYTHONPATH=src python tests/smoke_test.py
```

## Notes production

- Le projet choisit automatiquement CPU/CUDA/MPS selon disponibilite
- L'entree et la sortie sont normalisees dans `[-1, 1]`
- Des checkpoints complets sont sauvegardes pour reprise d'entrainement
- Le notebook couvre preparation, entrainement, evaluation qualitative et inference
