# Fonctionnalité 3 — Texte vers image anime

## Approches implémentées

### 1. cGAN + CLIP (Kaggle T4)
- notebooks/cgan_clip.ipynb
- 100 epochs, 128x128, 5000 images

### 2. StackGAN (Kaggle T4)
- notebooks/stackgan.ipynb
- Stage 1 : 64x64 / Stage 2 : 256x256
- 21000 images

### 3. Fine-tuning LoRA — Stable Diffusion (Modal B200)
- modal/train_lora.py
- runwayml/stable-diffusion-v1-5 + LoRA r=4
- 3 epochs, 5000 images

## Poids du modele
- lora_weights/adapter_model.safetensors
- lora_weights/adapter_config.json

