"""
Applies a trained CUT generator G to the full LPGS synthetic dataset,
producing a style-refined version (e.g. "synthetic_stone_carving/").
Each output is screened with the same lightweight sanity check used
during LPGS generation before being saved.

Usage on Colab:
    python generate_refined_dataset_cut.py
(edit the CONFIG block below)
"""

"""
Applies one or more trained CUT generator checkpoints to the LPGS
synthetic dataset, producing style-refined output per checkpoint. Each
output is screened with the same lightweight sanity check used during
LPGS generation before being saved.

Two modes, controlled by SAMPLE_SIZE_PER_CLASS below:

  - COMPARISON MODE (SAMPLE_SIZE_PER_CLASS = a small int, e.g. 5-10):
    Refines only a small, fixed sample of images per class, for EVERY
    checkpoint found in the checkpoints directory. Fast -- intended for
    visually comparing checkpoints side by side to pick the best one,
    without waiting for a full dataset run per checkpoint.

  - FULL MODE (SAMPLE_SIZE_PER_CLASS = None):
    Refines the ENTIRE synthetic dataset for every checkpoint selected
    via CHECKPOINT_EPOCHS. This is slow and storage-heavy if run across
    many checkpoints -- normally you'd set CHECKPOINT_EPOCHS to a single
    epoch (your chosen best checkpoint) once you've picked one using
    comparison mode first.

Usage on Colab:
    python generate_refined_dataset_cut.py
(edit the CONFIG block below)
"""

import csv
import os
import re

import torch
from PIL import Image
from torchvision import transforms

from models import Generator
from sanity_check import ink_coverage_ratio, passes_sanity_check

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DRIVE_ROOT = "<ROOT>"
STYLE_NAME = "textile"

SYNTHETIC_DIR = f"{DRIVE_ROOT}/dataset/synthetic"
CHECKPOINT_DIR = f"{DRIVE_ROOT}/cut_output_{STYLE_NAME}/checkpoints"
OUTPUT_ROOT = f"{DRIVE_ROOT}/dataset/synthetic_{STYLE_NAME}"
LOG_CSV_PATH = f"{DRIVE_ROOT}/cut_output_{STYLE_NAME}/refinement_log.csv"

# Which checkpoints to run:
#   None            -> auto-discover and run ALL G_epoch*.pth files found
#   [10, 30, 50]     -> run only these specific epochs
CHECKPOINT_EPOCHS = None

# Comparison mode (small int) vs full mode (None) -- see module docstring.
SAMPLE_SIZE_PER_CLASS = 400

IMAGE_SIZE = 100
CHANNELS = 3
BASE_CHANNELS = 64      # must match what was used in train_cut.py
N_RESIDUAL_BLOCKS = 9   # must match what was used in train_cut.py

# Wide open by default -- accept nearly everything automatically, and rely
# on the CSV log for manual triage instead.
MIN_INK_RATIO = 0.0
MAX_INK_RATIO = 1.0
# ------------------------------------------------------------------


def discover_checkpoints(checkpoint_dir, wanted_epochs=None):
    """Finds G_epoch{N}.pth files, returns sorted list of (epoch, path)."""
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

    pattern = re.compile(r"^G_epoch(\d+)\.pth$")
    found = []
    for fname in os.listdir(checkpoint_dir):
        m = pattern.match(fname)
        if m:
            epoch = int(m.group(1))
            if wanted_epochs is None or epoch in wanted_epochs:
                found.append((epoch, os.path.join(checkpoint_dir, fname)))

    found.sort(key=lambda x: x[0])
    if not found:
        raise ValueError(
            f"No matching checkpoints found in {checkpoint_dir}. "
            f"Expected files named like 'G_epoch10.pth'."
        )
    return found


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Style: {STYLE_NAME}")

    checkpoints = discover_checkpoints(CHECKPOINT_DIR, CHECKPOINT_EPOCHS)
    print(f"Found {len(checkpoints)} checkpoint(s) to process: "
          f"{[e for e, _ in checkpoints]}")
    if SAMPLE_SIZE_PER_CLASS is not None:
        print(f"Running in COMPARISON MODE: {SAMPLE_SIZE_PER_CLASS} images/class per checkpoint.")
    else:
        print("Running in FULL MODE: entire dataset per checkpoint. This may be slow/storage-heavy "
              "if multiple checkpoints are selected.")

    mode = "RGB" if CHANNELS == 3 else "L"
    norm = ([0.5] * CHANNELS, [0.5] * CHANNELS)
    to_tensor = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(lambda img: img.convert(mode)),
        transforms.ToTensor(),
        transforms.Normalize(*norm),
    ])

    class_names = sorted(
        d for d in os.listdir(SYNTHETIC_DIR) if os.path.isdir(os.path.join(SYNTHETIC_DIR, d))
    )

    # Build the per-class file list once; comparison mode takes a fixed
    # subset so the SAME images are compared across every checkpoint.
    class_files = {}
    for class_name in class_names:
        in_dir = os.path.join(SYNTHETIC_DIR, class_name)
        fnames = sorted(f for f in os.listdir(in_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if SAMPLE_SIZE_PER_CLASS is not None:
            fnames = fnames[:SAMPLE_SIZE_PER_CLASS]
        class_files[class_name] = fnames

    G = Generator(CHANNELS, BASE_CHANNELS, N_RESIDUAL_BLOCKS).to(device)

    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)
    with open(LOG_CSV_PATH, "w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["epoch", "class_name", "filename", "output_path", "ink_ratio", "passed_check"])

        for epoch, ckpt_path in checkpoints:
            print(f"\n========== Checkpoint: epoch {epoch} ==========")
            G.load_state_dict(torch.load(ckpt_path, map_location=device))
            G.eval()

            epoch_output_dir = f"{OUTPUT_ROOT}_epoch{epoch}"
            total_generated, total_rejected = 0, 0

            for class_name in class_names:
                in_dir = os.path.join(SYNTHETIC_DIR, class_name)
                out_dir = os.path.join(epoch_output_dir, class_name)
                os.makedirs(out_dir, exist_ok=True)

                fnames = class_files[class_name]
                for fname in fnames:
                    img_path = os.path.join(in_dir, fname)
                    img = Image.open(img_path)
                    tensor = to_tensor(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        refined = G(tensor)

                    refined_denorm = (refined.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
                    refined_pil = transforms.ToPILImage()(refined_denorm)

                    ratio = ink_coverage_ratio(refined_pil)
                    passed = passes_sanity_check(refined_pil, min_ink_ratio=MIN_INK_RATIO, max_ink_ratio=MAX_INK_RATIO)

                    if not passed:
                        total_rejected += 1
                        log_writer.writerow([epoch, class_name, fname, "", f"{ratio:.4f}", False])
                        continue

                    out_path = os.path.join(out_dir, fname)
                    refined_pil.save(out_path)
                    total_generated += 1
                    log_writer.writerow([epoch, class_name, fname, out_path, f"{ratio:.4f}", True])

            print(f"  Epoch {epoch}: {total_generated} saved, {total_rejected} rejected -> {epoch_output_dir}")

    print(f"\nAll checkpoints processed. Combined ink-ratio log: {LOG_CSV_PATH}")
    if SAMPLE_SIZE_PER_CLASS is not None:
        print("Comparison mode: review the per-epoch output folders "
              f"({OUTPUT_ROOT}_epoch<N>/) side by side, pick the best checkpoint, "
              "then re-run with SAMPLE_SIZE_PER_CLASS = None and "
              "CHECKPOINT_EPOCHS = [<chosen epoch>] for a full-dataset refinement.")
    else:
        print("Full mode: each *_epoch<N>/ folder is ready to use as a drop-in "
              "replacement for 'synthetic' in the k-fold OCR evaluation script.")
