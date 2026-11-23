"""
VGG Gram-matrix texture loss and AdaIN-style color loss.
"""

import torch
import torch.nn as nn
from torchvision import models as tv_models  # Fixed import alias

# Standard ImageNet normalization stats
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _to_vgg_input(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] RGB -> VGG ImageNet input."""
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    x_01 = (x * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
    mean = _IMAGENET_MEAN.to(x.device)
    std = _IMAGENET_STD.to(x.device)
    return (x_01 - mean) / std


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Batched normalized Gram matrix."""
    b, c, h, w = feat.shape
    feat_flat = feat.view(b, c, h * w)
    gram = torch.bmm(feat_flat, feat_flat.transpose(1, 2))
    return gram / (c * h * w)


class VGGFeatureExtractor(nn.Module):
    """Exposes frozen pretrained VGG16 activations."""

    LAYER_NAMES = {
        "3": "relu1_2",
        "8": "relu2_2",
        "15": "relu3_3",
        "22": "relu4_3",
    }

    def __init__(self):
        super().__init__()
        # Changed import source
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict:
        x = _to_vgg_input(x)
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.LAYER_NAMES:
                features[self.LAYER_NAMES[name]] = x
            if name == "22":
                break
        return features


class StyleLoss(nn.Module):
    """Gram-matrix texture loss across VGG."""

    def __init__(self, layer_weights: dict | None = None):
        super().__init__()
        self.extractor = VGGFeatureExtractor()
        self.layer_weights = layer_weights or {
            "relu1_2": 1.0,
            "relu2_2": 1.0,
            "relu3_3": 0.5,
            "relu4_3": 0.25,
        }
        self.mse = nn.MSELoss()

    def forward(self, generated: torch.Tensor, style_target: torch.Tensor) -> torch.Tensor:
        gen_feats = self.extractor(generated)
        target_feats = self.extractor(style_target)

        loss = 0.0
        for layer_name, weight in self.layer_weights.items():
            gram_gen = gram_matrix(gen_feats[layer_name])
            gram_target = gram_matrix(target_feats[layer_name])
            loss = loss + weight * self.mse(gram_gen, gram_target)
        return loss


class ColorLoss(nn.Module):
    """Mean/Std matching direct color transfer in [-1, 1]."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, generated: torch.Tensor, style_target: torch.Tensor) -> torch.Tensor:
        # Avoid dim errors on single channel, enforce RGB stats
        if generated.shape[1] == 1: generated = generated.repeat(1, 3, 1, 1)
        if style_target.shape[1] == 1: style_target = style_target.repeat(1, 3, 1, 1)

        gen_mean = generated.mean(dim=[2, 3])
        gen_std = generated.std(dim=[2, 3])
        target_mean = style_target.mean(dim=[2, 3])
        target_std = style_target.std(dim=[2, 3])

        return self.l1(gen_mean, target_mean) + self.l1(gen_std, target_std)