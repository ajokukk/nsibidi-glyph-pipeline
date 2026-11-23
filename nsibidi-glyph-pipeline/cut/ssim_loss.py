"""
Differentiable SSIM for structural preservation.
"""

import torch
import torch.nn.functional as F


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_1d = g.unsqueeze(0)
    kernel_2d = kernel_1d.t() @ kernel_1d
    kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel_2d


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, data_range: float = 2.0) -> torch.Tensor:
    """Mean SSIM between image batches."""
    channels = img1.shape[1]
    device = img1.device
    window = _gaussian_kernel(window_size, channels=channels).to(device)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


class SSIMLoss(torch.nn.Module):
    """Structural loss: 1 - SSIM."""

    def __init__(self, window_size: int = 11, data_range: float = 2.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(img1, img2, window_size=self.window_size, data_range=self.data_range)