from __future__ import annotations

try:
    import torch
except ModuleNotFoundError:
    print("Smoke tests ignores: PyTorch n'est pas installe dans l'environnement courant.")
    raise SystemExit(0)

from anime_gan_i2i.models import build_cyclegan_models, build_pix2pix_models


def check_pix2pix() -> None:
    generator, discriminator = build_pix2pix_models()
    x = torch.randn(2, 3, 64, 64)
    y = generator(x)
    pred = discriminator(y, x)
    assert y.shape == (2, 3, 64, 64)
    assert pred.shape[0] == 2


def check_cyclegan() -> None:
    generator_ab, generator_ba, discriminator_a, discriminator_b = build_cyclegan_models()
    a = torch.randn(2, 3, 64, 64)
    b = torch.randn(2, 3, 64, 64)
    fake_b = generator_ab(a)
    fake_a = generator_ba(b)
    pred_a = discriminator_a(fake_a)
    pred_b = discriminator_b(fake_b)
    assert fake_a.shape == (2, 3, 64, 64)
    assert fake_b.shape == (2, 3, 64, 64)
    assert pred_a.shape[0] == 2
    assert pred_b.shape[0] == 2


if __name__ == "__main__":
    check_pix2pix()
    check_cyclegan()
    print("Smoke tests OK")
