from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class FrozenCLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", embedding_dim: int = 512, local_files_only: bool = False):
        super().__init__()
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as exc:
            raise ImportError("Installer `transformers` pour utiliser InstructPix2Pix GAN.") from exc

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = CLIPTextModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.embedding_dim = embedding_dim
        self.projection = nn.Identity()
        hidden_size = self.model.config.hidden_size
        if hidden_size != embedding_dim:
            self.projection = nn.Linear(hidden_size, embedding_dim, bias=False)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.projection.parameters():
            parameter.requires_grad = False
        self.model.eval()
        self.projection.eval()

    @torch.no_grad()
    def encode(self, texts: Iterable[str], device: torch.device) -> torch.Tensor:
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = self.model(**batch)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return self.projection(embeddings).to(device)
