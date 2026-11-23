"""
Unpaired dataset loader for Nsibidi glyphs.
"""

import os
import random
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms


def get_real_train_holdout_split(real_dir: str, random_state: int = 42, n_splits: int = 5):
    """Reproduces the exact fold-0 split for OCR evaluation holding out."""
    classes = sorted(
        d for d in os.listdir(real_dir) if os.path.isdir(os.path.join(real_dir, d))
    )
    if len(classes) == 0:
        raise ValueError(f"No class subfolders found under real_dir='{real_dir}'.")

    all_paths, all_labels = [], []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(real_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                all_paths.append(os.path.join(class_dir, fname))
                all_labels.append(class_idx)

    if len(all_paths) == 0:
        raise ValueError(f"Found {len(classes)} folders, but no images under '{real_dir}'.")

    all_labels = np.array(all_labels)
    min_class_count = min(Counter(all_labels.tolist()).values())
    actual_splits = min(n_splits, min_class_count)

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(np.zeros(len(all_labels)), all_labels))
    train_idx, holdout_idx = splits[0]

    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    holdout_paths = [all_paths[i] for i in holdout_idx]
    holdout_labels = [all_labels[i] for i in holdout_idx]

    return train_paths, train_labels, holdout_paths, holdout_labels, classes


class ConvertImageMode:
    """Pickleable replacement for transforms.Lambda(lambda img: img.convert(mode))."""
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, img):
        return img.convert(self.mode)

class UnpairedGlyphDataset(Dataset):
    """Returns unpaired (img_A, img_B) tuples."""

    def __init__(self, domain_a_paths, domain_b_paths, image_size: int = 100, channels: int = 3,
                 steps_per_epoch: int | None = None):
        self.domain_a_paths = domain_a_paths
        self.domain_b_paths = domain_b_paths
        self.channels = channels
        self.steps_per_epoch = steps_per_epoch or len(domain_a_paths)

        mode = "RGB" if channels == 3 else "L"
        # Standard normalization: maps [0,1] -> [-1,1]
        mean = [0.5] * channels
        std = [0.5] * channels

        # Fix: Lambdas cause pickle errors on Windows. Replace with proper classes.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            ConvertImageMode(mode), # Named class is pickleable
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        # idx is loop counter; random sample for true unpairing
        a_path = random.choice(self.domain_a_paths)
        b_path = random.choice(self.domain_b_paths)

        img_a = self.transform(Image.open(a_path))
        img_b = self.transform(Image.open(b_path))

        return img_a, img_b


def list_image_paths(root_dir: str):
    """Collects paths root_dir/<class>/*.png."""
    paths = []
    if not os.path.isdir(root_dir):
        return paths
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(class_dir, fname))
    return paths


def list_style_reference_paths(style_dir: str):
    """Collects style reference images (flat or single nest)."""
    paths = []
    if not os.path.isdir(style_dir):
        return paths

    for entry in sorted(os.listdir(style_dir)):
        entry_path = os.path.join(style_dir, entry)
        if os.path.isdir(entry_path):
            for fname in sorted(os.listdir(entry_path)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    paths.append(os.path.join(entry_path, fname))
        elif entry.lower().endswith((".png", ".jpg", ".jpeg")):
            paths.append(entry_path)
    return paths