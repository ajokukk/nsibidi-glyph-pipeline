"""
Lightweight CycleGAN/CUT architecture for Nsibidi glyph realism transfer.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    ResNet-based generator: downsample -> residual blocks -> upsample.
    Supports standard forward pass (CycleGAN) and feature-extraction (CUT).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, n_residual_blocks: int = 6):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]

        # Downsampling (2x)
        c = base_channels
        for _ in range(2):
            layers += [
                nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(c * 2),
                nn.ReLU(inplace=True),
            ]
            c *= 2

        # Residual blocks at the bottleneck resolution
        for _ in range(n_residual_blocks):
            layers.append(ResidualBlock(c))

        # Upsampling (2x)
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(c, c // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(c // 2),
                nn.ReLU(inplace=True),
            ]
            c //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, in_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.ModuleList(layers)

    def forward(self, x, layers=None, encode_only=False):
        if layers is None:
            # Standard full forward pass
            for layer in self.model:
                x = layer(x)
            return x

        # Feature-extraction pass for PatchNCE
        feats = []
        max_layer = max(layers)
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in layers:
                feats.append(x)
            if encode_only and i == max_layer:
                return feats
        if encode_only:
            return feats
        return feats, x


class Discriminator(nn.Module):
    """
    PatchGAN discriminator.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        def block(in_c, out_c, stride=2, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, base_channels, normalize=False),
            *block(base_channels, base_channels * 2),
            *block(base_channels * 2, base_channels * 4),
            *block(base_channels * 4, base_channels * 8, stride=1),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


def init_weights(m):
    """Standard weight initialization."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("InstanceNorm2d") != -1:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)