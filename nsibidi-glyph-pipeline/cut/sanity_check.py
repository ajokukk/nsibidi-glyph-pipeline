"""
Lightweight ink-coverage sanity check filter.
"""

import numpy as np
from PIL import Image


def ink_coverage_ratio(img: Image.Image, ink_threshold: int = 235) -> float:
    """Returns fraction of 'ink' pixels (darker than threshold in L)."""
    gray = np.array(img.convert("L")).astype(float)
    ink_mask = gray < ink_threshold
    return ink_mask.sum() / ink_mask.size


def passes_sanity_check(
    img: Image.Image,
    min_ink_ratio: float = 0.00,
    max_ink_ratio: float = 1.00,
    ink_threshold: int = 235,
) -> bool:
    """True if image ink coverage falls within calibrated bounds."""
    ratio = ink_coverage_ratio(img, ink_threshold=ink_threshold)
    return min_ink_ratio <= ratio <= max_ink_ratio


def calibrate_bounds(sample_images, ink_threshold: int = 235, margin: float = 0.25):
    """Calibrates min/max bounds from known-good samples with margin."""
    if not sample_images:
        return 0.0, 1.0
    ratios = [ink_coverage_ratio(img, ink_threshold=ink_threshold) for img in sample_images]
    obs_min, obs_max = min(ratios), max(ratios)
    span = obs_max - obs_min
    return max(0.0, obs_min - span * margin), min(1.0, obs_max + span * margin)