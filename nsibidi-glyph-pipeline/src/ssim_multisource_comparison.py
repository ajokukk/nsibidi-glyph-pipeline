"""
Multi-source SSIM structural fidelity evaluation:

  1. Synthetic vs. Dictionary Prototype -- original Table I methodology
  2. Real vs. Dictionary Prototype
  3. Real vs. Real (intra-class, pairwise) -- natural variability baseline
  4. Synthetic vs. Historical Source      -- LPGS output vs. glyphs
     illustrated in independent historical publications (e.g. Macgregor's
     "Some Notes on Nsibidi" / "Further Notes on Nsibidi"), where covered
  5. Real vs. Historical Source
  6. Dictionary Prototype vs. Historical Source -- cross-source agreement:
     do two independent scholarly sources depict the same symbol
     consistently? This validates the "ground truth" itself, not just
     LPGS's fidelity to one reference.

The historical source will typically have PARTIAL class coverage (only
symbols actually illustrated in those specific papers) -- classes without
a historical-source image are simply skipped for axes 4-6, not treated
as an error.

If LPGS's synthetic-vs-prototype SSIM falls within the real-vs-real
range for a class, that's quantitative evidence of realistic variation.
If it falls well outside that range (either direction), that's flagged
for discussion -- too high suggests insufficient diversity, too low
suggests loss of structural identity.

Requires scikit-image:
    pip install scikit-image

Usage:
    python ssim_threeway_comparison.py
(edit the CONFIG block below)
"""

import csv
import itertools
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DRIVE_ROOT = "<ROOT>"

PROTOTYPE_DIR = f"{DRIVE_ROOT}/IEEE kfold CORRECTIONS/Prototypes"  # the extracted LpGs.rar folder
SYNTHETIC_DIR = f"{DRIVE_ROOT}/IEEE kfold CORRECTIONS/LPGSvariations_augmentation"
REAL_DIR = f"{DRIVE_ROOT}/dataset/real"

# Historical publication source (Macgregor's Nsibidi papers, etc.). Expects
# files named directly by canonical class name, e.g. "Leopard.png" --
# partial coverage is fine; missing classes are skipped, not errors.
HISTORICAL_DIR = f"{DRIVE_ROOT}/dataset/historical_sources"

IMAGE_SIZE = 100
SSIM_WINDOW_SIZE = 7

# Max real-vs-real pairs per class (guards against blowup: n images -> n*(n-1)/2
# pairs; with ~5-20 real images per class this is small regardless, but capped
# for safety if any class has substantially more).
MAX_REAL_PAIRS_PER_CLASS = 200

OUTPUT_TABLE_CSV = f"{DRIVE_ROOT}/ssim_multisource_table.csv"
OUTPUT_LOG_CSV = f"{DRIVE_ROOT}/ssim_multisource_per_image_log.csv"
# ------------------------------------------------------------------

# Maps prototype filenames (from LpGs.rar, with synonym suffixes) to the
# canonical 20 class names used throughout the project. Verified as a
# complete, unambiguous one-to-one mapping against Table I's class list.
PROTOTYPE_FILENAME_TO_CLASS = {
    "0001 leopard.png": "Leopard",
    "0003 chief.png": "Chief",
    "0005 bush_woods__forest_grove.png": "Forest",
    "0013 female.png": "Female",
    "0014 male.png": "Male",
    "0016 breast_milk.png": "Breast_milk",
    "0074 vagina_vulva.png": "Vagina",
    "0075 penis.png": "Penis",
    "0078 unity.png": "Unity",
    "0079 lover.png": "Lovers",
    "0081 journey.png": "Journey",
    "0088 eye.png": "Eye",
    "0089 see.png": "See",
    "0109 love_affection__admiration.png": "Love_affection",
    "0409 trade.png": "Trade",
    "0411 trader__businessman.png": "Trader",
    "0434 staff_walking_stick.png": "Staff",
    "0627 female_maiden.png": "Maiden",
    "0889 crossbow.png": "Crossbow",
    "0959 mans_quarters_building_base.png": "MansQuarters",
}


def load_grayscale(path, size=IMAGE_SIZE):
    img = Image.open(path).convert("L").resize((size, size))
    return np.asarray(img, dtype=np.float64)


def compute_ssim(path1, path2):
    img1 = load_grayscale(path1)
    img2 = load_grayscale(path2)
    return sk_ssim(img1, img2, data_range=255, win_size=SSIM_WINDOW_SIZE)


def build_prototype_lookup():
    """Returns {class_name: full_path_to_prototype_image}."""
    lookup = {}
    missing = []
    for fname, class_name in PROTOTYPE_FILENAME_TO_CLASS.items():
        path = os.path.join(PROTOTYPE_DIR, fname)
        if os.path.exists(path):
            lookup[class_name] = path
        else:
            missing.append(fname)
    if missing:
        print(f"WARNING: {len(missing)} prototype file(s) not found in {PROTOTYPE_DIR}: {missing}")
    return lookup


def build_historical_lookup(historical_dir, canonical_class_names):
    """
    Returns {class_name: full_path_to_historical_image} for whichever
    classes have a matching file directly named <ClassName>.<ext> in
    historical_dir. Partial coverage is expected -- classes without a
    file here are simply absent from the returned dict.
    """
    lookup = {}
    if not os.path.isdir(historical_dir):
        print(f"NOTE: historical source directory not found ({historical_dir}) -- "
              f"skipping historical-source comparison axes entirely.")
        return lookup

    for class_name in canonical_class_names:
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(historical_dir, f"{class_name}{ext}")
            if os.path.exists(path):
                lookup[class_name] = path
                break

    covered = len(lookup)
    print(f"Historical source: {covered}/{len(canonical_class_names)} classes covered "
          f"({sorted(lookup.keys())}).")
    return lookup


def list_images(class_dir):
    if not os.path.isdir(class_dir):
        return []
    return sorted(f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))


if __name__ == "__main__":
    prototype_lookup = build_prototype_lookup()
    class_names = sorted(prototype_lookup.keys())
    historical_lookup = build_historical_lookup(HISTORICAL_DIR, class_names)
    print(f"\nEvaluating {len(class_names)} classes with matched dictionary prototypes.\n")

    per_image_rows = []  # (class, comparison_type, filename_or_pair, ssim)
    summary_rows = []

    def stats(scores):
        if not scores:
            return (None, None, None, None)
        arr = np.array(scores)
        return arr.mean(), arr.std(), arr.min(), arr.max()

    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    for class_name in class_names:
        prototype_path = prototype_lookup[class_name]
        historical_path = historical_lookup.get(class_name)  # may be None

        # --- 1. Synthetic vs Prototype ---
        synth_dir = os.path.join(SYNTHETIC_DIR, class_name)
        synth_fnames = list_images(synth_dir)
        synth_proto_scores = []
        synth_hist_scores = []
        for fname in synth_fnames:
            score = compute_ssim(os.path.join(synth_dir, fname), prototype_path)
            synth_proto_scores.append(score)
            per_image_rows.append([class_name, "synthetic_vs_prototype", fname, f"{score:.4f}"])

            if historical_path:
                h_score = compute_ssim(os.path.join(synth_dir, fname), historical_path)
                synth_hist_scores.append(h_score)
                per_image_rows.append([class_name, "synthetic_vs_historical", fname, f"{h_score:.4f}"])

        # --- 2. Real vs Prototype, and Real vs Historical ---
        real_dir = os.path.join(REAL_DIR, class_name)
        real_fnames = list_images(real_dir)
        real_proto_scores = []
        real_hist_scores = []
        for fname in real_fnames:
            score = compute_ssim(os.path.join(real_dir, fname), prototype_path)
            real_proto_scores.append(score)
            per_image_rows.append([class_name, "real_vs_prototype", fname, f"{score:.4f}"])

            if historical_path:
                h_score = compute_ssim(os.path.join(real_dir, fname), historical_path)
                real_hist_scores.append(h_score)
                per_image_rows.append([class_name, "real_vs_historical", fname, f"{h_score:.4f}"])

        # --- 3. Real vs Real (intra-class, pairwise) ---
        real_real_scores = []
        pairs = list(itertools.combinations(real_fnames, 2))[:MAX_REAL_PAIRS_PER_CLASS]
        for fname_a, fname_b in pairs:
            score = compute_ssim(os.path.join(real_dir, fname_a), os.path.join(real_dir, fname_b))
            real_real_scores.append(score)
            per_image_rows.append([class_name, "real_vs_real", f"{fname_a}|{fname_b}", f"{score:.4f}"])

        # --- 4. Prototype vs Historical (cross-source agreement, single pair if available) ---
        proto_hist_score = None
        if historical_path:
            proto_hist_score = compute_ssim(prototype_path, historical_path)
            per_image_rows.append([class_name, "prototype_vs_historical", "single_pair", f"{proto_hist_score:.4f}"])

        s_mean, s_std, s_min, s_max = stats(synth_proto_scores)
        rp_mean, rp_std, rp_min, rp_max = stats(real_proto_scores)
        rr_mean, rr_std, rr_min, rr_max = stats(real_real_scores)
        sh_mean, _, _, _ = stats(synth_hist_scores)
        rh_mean, _, _, _ = stats(real_hist_scores)

        # Flag: does synthetic-vs-prototype mean fall within the real-vs-real range?
        if rr_min is not None and s_mean is not None:
            in_range = rr_min <= s_mean <= rr_max
            flag = "within natural range" if in_range else "OUTSIDE natural range"
        else:
            flag = "insufficient real pairs to compare"

        summary_rows.append([
            class_name, len(synth_fnames), len(real_fnames), len(pairs),
            "yes" if historical_path else "no",
            fmt(s_mean), fmt(rp_mean), fmt(rr_mean), fmt(rr_min), fmt(rr_max),
            fmt(sh_mean), fmt(rh_mean), fmt(proto_hist_score),
            flag,
        ])

        print(f"{class_name:<16} synth_vs_proto={fmt(s_mean):<8} real_vs_proto={fmt(rp_mean):<8} "
              f"real_vs_real={fmt(rr_mean):<8} [{fmt(rr_min)}, {fmt(rr_max)}]  "
              f"hist={'Y' if historical_path else 'n/a'}  -> {flag}")

    with open(OUTPUT_TABLE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_name", "n_synthetic", "n_real", "n_real_pairs", "has_historical_source",
            "mean_ssim_synthetic_vs_prototype", "mean_ssim_real_vs_prototype",
            "mean_ssim_real_vs_real", "min_ssim_real_vs_real", "max_ssim_real_vs_real",
            "mean_ssim_synthetic_vs_historical", "mean_ssim_real_vs_historical",
            "ssim_prototype_vs_historical",
            "synthetic_within_natural_range",
        ])
        writer.writerows(summary_rows)

    with open(OUTPUT_LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "comparison_type", "identifier", "ssim"])
        writer.writerows(per_image_rows)

    n_historical = sum(1 for row in summary_rows if row[4] == "yes")
    print(f"\nSummary table: {OUTPUT_TABLE_CSV}")
    print(f"Full per-image/per-pair log: {OUTPUT_LOG_CSV}")
    print(f"Historical-source comparison available for {n_historical}/{len(class_names)} classes.")
    if n_historical > 0:
        proto_hist_scores = [float(r[12]) for r in summary_rows if r[12] != "N/A"]
        if proto_hist_scores:
            print(f"Mean prototype-vs-historical agreement (cross-source consistency): "
                  f"{np.mean(proto_hist_scores):.4f}")
