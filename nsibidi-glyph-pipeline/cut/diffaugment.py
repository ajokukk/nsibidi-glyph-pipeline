"""
Lightweight differentiable augmentations (translation, cutout, color jitter).
Based on DiffAugment idea for GAN training stability.
"""

import torch
import torch.nn.functional as F


def rand_translation(x: torch.Tensor, max_shift_ratio: float = 0.125) -> torch.Tensor:
    """Random reflected padding translation."""
    b, c, h, w = x.shape
    shift_h = int(h * max_shift_ratio)
    shift_w = int(w * max_shift_ratio)

    translation_h = torch.randint(-shift_h, shift_h + 1, (b,), device=x.device)
    translation_w = torch.randint(-shift_w, shift_w + 1, (b,), device=x.device)

    x_padded = F.pad(x, (shift_w, shift_w, shift_h, shift_h), mode="reflect")

    out = torch.empty_like(x)
    for i in range(b):
        th, tw = translation_h[i].item() + shift_h, translation_w[i].item() + shift_w
        out[i] = x_padded[i, :, th:th + h, tw:tw + w]
    return out


def rand_cutout(x: torch.Tensor, cutout_ratio: float = 0.3) -> torch.Tensor:
    """Random square cutout applied locally."""
    b, c, h, w = x.shape
    cut_h, cut_w = int(h * cutout_ratio), int(w * cutout_ratio)

    mask = torch.ones_like(x)
    for i in range(b):
        cy = torch.randint(0, h, (1,)).item()
        cx = torch.randint(0, w, (1,)).item()
        y0, y1 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
        x0, x1 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
        mask[i, :, y0:y1, x0:x1] = 0.0
    return x * mask


def rand_brightness_contrast(x: torch.Tensor, brightness: float = 0.2, contrast: float = 0.2) -> torch.Tensor:
    """Differentiable brightness and contrast jitter."""
    b = x.shape[0]
    # Randomized shifts per batch item
    brightness_shift = (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 2 * brightness
    contrast_scale = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 2 * contrast

    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - x_mean) * contrast_scale + x_mean + brightness_shift


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """Applies translation, color jitter, and cutout."""
    x = rand_translation(x)
    x = rand_brightness_contrast(x)
    x = rand_cutout(x)
    return x