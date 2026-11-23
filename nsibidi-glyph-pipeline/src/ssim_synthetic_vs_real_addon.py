"""
Computes synthetic-vs-real SSIM per class -- the one comparison axis
missing from the original six-comparison request:

  1. Real vs. real                    (already in ssim_multisource_table.csv)
  2. Prototype vs. real               (already in ssim_multisource_table.csv)
  3. Historical vs. prototype         (already in ssim_multisource_table.csv)
  4. Historical vs. synthetic         (already in ssim_multisource_table.csv)
  5. Historical vs. real              (already in ssim_multisource_table.csv)
  6. Synthetic vs. real               (computed here)

Directly compares synthesized glyphs against genuine handwritten/web-
sourced glyphs of the same class -- the most direct test of whether
LPGS output resembles authentic handwriting, without routing the
comparison through the dictionary prototype.

To avoid a combinatorial explosion (500 synthetic x up to ~22 real
images per class), a capped number of random synthetic-real pairs are
sampled per class rather than computing every possible pair.

Requires scikit-image:
    pip install scikit-image

Usage:
    python ssim_synthetic_vs_real_addon.py
(edit the CONFIG block below)
"""

import csv
import os
import random

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DRIVE_ROOT = "<ROOT>"

SYNTHETIC_DIR = f"{DRIVE_ROOT}/IEEE kfold CORRECTIONS/LPGSvariations_augmentation"
REAL_DIR = f"{DRIVE_ROOT}/dataset/real"

IMAGE_SIZE = 100
SSIM_WINDOW_SIZE = 7
MAX_PAIRS_PER_CLASS = 200  # caps combinatorial blowup; random sample of pairs

# Path to the existing multisource table, so this script's results can be
# merged into it rather than kept as a separate disconnected file
EXISTING_TABLE_CSV = f"{DRIVE_ROOT}/ssim_multisource_table.csv"

OUTPUT_TABLE_CSV = f"{DRIVE_ROOT}/ssim_synthetic_vs_real_table.csv"
MERGED_TABLE_CSV = f"{DRIVE_ROOT}/ssim_multisource_table_complete.csv"
# ------------------------------------------------------------------


def load_grayscale(path, size=IMAGE_SIZE):
    img = Image.open(path).convert("L").resize((size, size))
    return np.asarray(img, dtype=np.float64)


def compute_ssim(path1, path2):
    img1 = load_grayscale(path1)
    img2 = load_grayscale(path2)
    return sk_ssim(img1, img2, data_range=255, win_size=SSIM_WINDOW_SIZE)


def list_images(class_dir):
    if not os.path.isdir(class_dir):
        return []
    return sorted(f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))


if __name__ == "__main__":
    class_names = sorted(
        d for d in os.listdir(REAL_DIR) if os.path.isdir(os.path.join(REAL_DIR, d))
    )
    print(f"Computing synthetic-vs-real SSIM for {len(class_names)} classes...\n")

    results = {}  # class_name -> (mean, std, min, max, n_pairs)

    with open(OUTPUT_TABLE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "n_pairs", "mean_ssim_synthetic_vs_real",
                          "std_ssim_synthetic_vs_real", "min_ssim_synthetic_vs_real",
                          "max_ssim_synthetic_vs_real"])

        for class_name in class_names:
            synth_dir = os.path.join(SYNTHETIC_DIR, class_name)
            real_dir_cls = os.path.join(REAL_DIR, class_name)

            synth_fnames = list_images(synth_dir)
            real_fnames = list_images(real_dir_cls)

            if not synth_fnames or not real_fnames:
                print(f"  {class_name}: skipping (missing synthetic or real images)")
                continue

            # Sample random synthetic-real pairs, capped at MAX_PAIRS_PER_CLASS
            max_possible = len(synth_fnames) * len(real_fnames)
            n_pairs = min(MAX_PAIRS_PER_CLASS, max_possible)

            scores = []
            seen_pairs = set()
            attempts = 0
            while len(scores) < n_pairs and attempts < n_pairs * 5:
                attempts += 1
                s_fname = random.choice(synth_fnames)
                r_fname = random.choice(real_fnames)
                pair_key = (s_fname, r_fname)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                score = compute_ssim(
                    os.path.join(synth_dir, s_fname),
                    os.path.join(real_dir_cls, r_fname),
                )
                scores.append(score)

            arr = np.array(scores)
            results[class_name] = (arr.mean(), arr.std(), arr.min(), arr.max(), len(scores))
            writer.writerow([class_name, len(scores), f"{arr.mean():.4f}", f"{arr.std():.4f}",
                              f"{arr.min():.4f}", f"{arr.max():.4f}"])
            print(f"  {class_name:<16} n_pairs={len(scores):<5} mean={arr.mean():.4f} "
                  f"[{arr.min():.4f}, {arr.max():.4f}]")

    print(f"\nSaved: {OUTPUT_TABLE_CSV}")

    # --- Merge into the existing multisource table, if present ---
    if os.path.exists(EXISTING_TABLE_CSV):
        with open(EXISTING_TABLE_CSV) as f:
            existing_rows = list(csv.DictReader(f))
            fieldnames = list(existing_rows[0].keys()) + [
                "n_pairs_synthetic_vs_real", "mean_ssim_synthetic_vs_real",
                "min_ssim_synthetic_vs_real", "max_ssim_synthetic_vs_real",
            ]

        for row in existing_rows:
            cls = row["class_name"]
            if cls in results:
                mean, std, mn, mx, n = results[cls]
                row["n_pairs_synthetic_vs_real"] = n
                row["mean_ssim_synthetic_vs_real"] = f"{mean:.4f}"
                row["min_ssim_synthetic_vs_real"] = f"{mn:.4f}"
                row["max_ssim_synthetic_vs_real"] = f"{mx:.4f}"
            else:
                row["n_pairs_synthetic_vs_real"] = "N/A"
                row["mean_ssim_synthetic_vs_real"] = "N/A"
                row["min_ssim_synthetic_vs_real"] = "N/A"
                row["max_ssim_synthetic_vs_real"] = "N/A"

        with open(MERGED_TABLE_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)

        print(f"Merged complete six-comparison table (all classes): {MERGED_TABLE_CSV}")
    else:
        print(f"NOTE: {EXISTING_TABLE_CSV} not found -- only the standalone "
              f"synthetic-vs-real table was saved. Re-run with the existing "
              f"multisource table present to get a single merged file.")
