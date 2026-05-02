from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def init_weights(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias.data)
    elif "BatchNorm2d" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.zeros_(module.bias.data)


class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.cat((x, skip), dim=1)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        bottleneck = self.down6(d5)

        u1 = self.up1(bottleneck, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        return self.final(u5)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, conditional: bool = False):
        super().__init__()
        channels = in_channels * 2 if conditional else in_channels
        layers = [
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),
        ]
        self.model = nn.Sequential(*layers)
        self.conditional = conditional
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if self.conditional:
            if condition is None:
                raise ValueError("Le discriminateur conditionnel requiert une condition.")
            x = torch.cat((condition, x), dim=1)
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_blocks: int = 6):
        super().__init__()
        layers = [
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
        ]
        for _ in range(num_blocks):
            layers.append(ResnetBlock(256))
        layers.extend(
            [
                nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1, bias=False),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1, bias=False),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, out_channels, 7),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AdaLIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.rho = nn.Parameter(torch.full((1, num_features, 1, 1), 0.9))
        self.eps = eps

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        in_mean, in_var = torch.mean(x, dim=(2, 3), keepdim=True), torch.var(x, dim=(2, 3), keepdim=True, unbiased=False)
        ln_mean, ln_var = torch.mean(x, dim=(1, 2, 3), keepdim=True), torch.var(x, dim=(1, 2, 3), keepdim=True, unbiased=False)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        rho = self.rho.expand(x.size(0), -1, -1, -1).clamp(0.0, 1.0)
        out = rho * out_in + (1.0 - rho) * out_ln
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return out * gamma + beta


class CAM(nn.Module):
    def __init__(self, channels: int, with_spectral_norm: bool = False):
        super().__init__()
        linear = nn.Linear(channels, 1, bias=False)
        self.gap_fc = spectral_norm(linear) if with_spectral_norm else linear
        linear = nn.Linear(channels, 1, bias=False)
        self.gmp_fc = spectral_norm(linear) if with_spectral_norm else linear
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
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(x)
        features, cam_logit, heatmap = self.cam(features)
        gamma_beta = self.gamma_beta(features)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        for block in self.resblocks:
            features = block(features, gamma, beta)
        output = self.decoder(features)
        return {"image": output, "cam_logit": cam_logit, "heatmap": heatmap}


class UGATITDiscriminatorSingle(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_layers: int = 4):
        super().__init__()
        layers: List[nn.Module] = [
            spectral_norm(nn.Conv2d(in_channels, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base_channels
        for idx in range(1, num_layers):
            next_channels = min(base_channels * (2 ** idx), 512)
            stride = 1 if idx == num_layers - 1 else 2
            layers.extend(
                [
                    spectral_norm(nn.Conv2d(channels, next_channels, 4, stride, 1)),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = next_channels
        self.features = nn.Sequential(*layers)
        self.cam = CAM(channels, with_spectral_norm=True)
        self.classifier = spectral_norm(nn.Conv2d(channels, 1, 4, 1, 1))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.features(x)
        features, cam_logit, heatmap = self.cam(features)
        logit = self.classifier(features)
        return {"logit": logit, "cam_logit": cam_logit, "heatmap": heatmap}


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.scale_1 = UGATITDiscriminatorSingle(in_channels=in_channels, base_channels=64, num_layers=4)
        self.scale_2 = UGATITDiscriminatorSingle(in_channels=in_channels, base_channels=32, num_layers=3)
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        outputs = [self.scale_1(x)]
        outputs.append(self.scale_2(self.downsample(x)))
        return outputs


class AniGANGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_blocks: int = 6):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 7, 1, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.resblocks = nn.Sequential(*[ResnetBlock(256) for _ in range(num_blocks)])
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.final = nn.Sequential(nn.Conv2d(128, out_channels, 7, 1, 3), nn.Tanh())
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        bottleneck = self.resblocks(e3)
        d1 = self.dec1(torch.cat((bottleneck, e3), dim=1))
        d2 = self.dec2(torch.cat((d1, e2), dim=1))
        return self.final(torch.cat((d2, e1), dim=1))


class FiLMLayer(nn.Module):
    def __init__(self, feature_dim: int, text_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(text_dim, feature_dim)
        self.to_beta = nn.Linear(text_dim, feature_dim)

    def forward(self, x: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(text_embedding).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(text_embedding).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class TextConditionedUNetGenerator(nn.Module):
    def __init__(self, text_dim: int = 512, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.3)
        self.down5 = UNetDown(512, 512, dropout=0.3)
        self.down6 = UNetDown(512, 512, dropout=0.3)
        self.film_mid = FiLMLayer(512, text_dim)
        self.film_bottleneck = FiLMLayer(512, text_dim)
        self.up1 = UNetUp(512, 512, dropout=0.3)
        self.up2 = UNetUp(1024, 512, dropout=0.3)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d4 = self.film_mid(d4, text_embedding)
        d5 = self.down5(d4)
        bottleneck = self.down6(d5)
        bottleneck = self.film_bottleneck(bottleneck, text_embedding)
        u1 = self.up1(bottleneck, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        return self.final(u5)


class TextConditionedPatchDiscriminator(nn.Module):
    def __init__(self, text_dim: int = 512, in_channels: int = 3, image_size: int = 64):
        super().__init__()
        self.image_size = image_size
        self.text_projection = nn.Linear(text_dim, image_size * image_size)
        self.discriminator = PatchDiscriminator(in_channels=in_channels + 1, conditional=True)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        text_map = self.text_projection(text_embedding).view(text_embedding.size(0), 1, self.image_size, self.image_size)
        conditioned_image = torch.cat((x, text_map), dim=1)
        conditioned_source = torch.cat((condition, text_map), dim=1)
        return self.discriminator(conditioned_image, conditioned_source)


def build_pix2pix_models(config: Dict | None = None):
    generator = UNetGenerator()
    discriminator = PatchDiscriminator(conditional=True)
    return generator, discriminator


def build_cyclegan_models(config: Dict | None = None):
    generator_ab = ResnetGenerator()
    generator_ba = ResnetGenerator()
    discriminator_a = PatchDiscriminator(conditional=False)
    discriminator_b = PatchDiscriminator(conditional=False)
    return generator_ab, generator_ba, discriminator_a, discriminator_b


def build_ugatit_models(config: Dict | None = None):
    num_blocks = 4
    if config is not None:
        model_cfg = config.get("model", {})
        ugatit_cfg = model_cfg.get("ugatit", {})
        num_blocks = int(ugatit_cfg.get("num_blocks", num_blocks))

    generator_ab = UGATITGenerator(num_blocks=num_blocks)
    generator_ba = UGATITGenerator(num_blocks=num_blocks)
    discriminator_a = MultiScaleDiscriminator()
    discriminator_b = MultiScaleDiscriminator()
    return generator_ab, generator_ba, discriminator_a, discriminator_b


def build_anigan_models(config: Dict | None = None):
    generator = AniGANGenerator()
    discriminator = PatchDiscriminator(conditional=True)
    return generator, discriminator


def build_instruct_pix2pix_models(config: Dict | None = None):
    text_dim = 512
    image_size = 64
    if config is not None:
        text_dim = config.get("text_encoder", {}).get("embedding_dim", text_dim)
        image_size = config.get("data", {}).get("image_size", image_size)
    generator = TextConditionedUNetGenerator(text_dim=text_dim)
    discriminator = TextConditionedPatchDiscriminator(text_dim=text_dim, image_size=image_size)
    return generator, discriminator


MODEL_BUILDERS = {
    "pix2pix": build_pix2pix_models,
    "cyclegan": build_cyclegan_models,
    "ugatit": build_ugatit_models,
    "anigan": build_anigan_models,
    "instruct_pix2pix": build_instruct_pix2pix_models,
}


def build_models(config: Dict):
    model_name = config.get("model_name", config.get("mode"))
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Modele non supporte: {model_name}")
    return MODEL_BUILDERS[model_name](config)
