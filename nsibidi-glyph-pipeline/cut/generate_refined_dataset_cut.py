"""
Applies a trained CUT generator G to synthesize a style-refined dataset variant.
Output screened by ink sanity check.
"""

import csv
import os

import torch
from PIL import Image
from torchvision import transforms

# Imports match local structure
from models import Generator
from glyph_datasets import ConvertImageMode
from sanity_check import ink_coverage_ratio, passes_sanity_check

# ------------------------------------------------------------------
# CONFIG - Edit these paths for local use
# ------------------------------------------------------------------
ROOT = "<ROOT>"  # Example local root
STYLE_NAME = "textile"

SYNTHETIC_DIR = f"{ROOT}/dataset/synthetic"
OUTPUT_DIR = f"{ROOT}/dataset/CUT_synthetic_{STYLE_NAME}_5"
# Ensure this checkpoint actually exists
CHECKPOINT_PATH = f"{ROOT}/cut_output_{STYLE_NAME}/checkpoints/G_epoch5.pth"
LOG_CSV_PATH = f"{ROOT}/cut_output_{STYLE_NAME}/refinement_log.csv"

IMAGE_SIZE = 100
CHANNELS = 3
BASE_CHANNELS = 64  # Sized up to T4 capacity
N_RESIDUAL_BLOCKS = 9  # original paper depth

# Wide open default bounds - rely on CSV log triage instead
MIN_INK_RATIO = 0.0
MAX_INK_RATIO = 1.0


# ------------------------------------------------------------------

def get_generation_transform(size, channels):
    mode = "RGB" if channels == 3 else "L"
    norm = ([0.5] * channels, [0.5] * channels)
    return transforms.Compose([
        transforms.Resize((size, size)),
        ConvertImageMode(mode),  # Pickleable
        transforms.ToTensor(),
        transforms.Normalize(*norm),
    ])


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Missing G checkpoint: {CHECKPOINT_PATH}")

    print(f"Refining synthetic dataset with style: {STYLE_NAME}")

    G = Generator(CHANNELS, BASE_CHANNELS, N_RESIDUAL_BLOCKS).to(device)
    G.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    G.eval()

    to_tensor = get_generation_transform(IMAGE_SIZE, CHANNELS)
    # Norm definition needed for denormalization
    denorm_mean = torch.tensor([0.5] * CHANNELS).view(CHANNELS, 1, 1).to(device)
    denorm_std = torch.tensor([0.5] * CHANNELS).view(CHANNELS, 1, 1).to(device)

    if not os.path.isdir(SYNTHETIC_DIR):
        raise FileNotFoundError(f"Synthetic dir not found: {SYNTHETIC_DIR}")

    class_names = sorted(
        d for d in os.listdir(SYNTHETIC_DIR) if os.path.isdir(os.path.join(SYNTHETIC_DIR, d))
    )

    total_generated, total_rejected = 0, 0
    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)

    with open(LOG_CSV_PATH, "w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["class_name", "filename", "output_path", "ink_ratio", "passed_check"])

        for class_name in class_names:
            in_dir = os.path.join(SYNTHETIC_DIR, class_name)
            out_dir = os.path.join(OUTPUT_DIR, class_name)
            os.makedirs(out_dir, exist_ok=True)

            fnames = sorted(f for f in os.listdir(in_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
            print(f"\n[{class_name}] refining {len(fnames)} images...")

            for fname in fnames:
                img_path = os.path.join(in_dir, fname)
                img = Image.open(img_path)
                tensor = to_tensor(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    refined = G(tensor)

                # Denormalize manually [-1,1] -> [0,1]
                refined_01 = (refined.squeeze(0) * denorm_std + denorm_mean).clamp(0, 1)
                refined_pil = transforms.ToPILImage()(refined_01.cpu())

                ratio = ink_coverage_ratio(refined_pil)
                # Open bounds by default
                passed = passes_sanity_check(refined_pil, min_ink_ratio=MIN_INK_RATIO, max_ink_ratio=MAX_INK_RATIO)

                if not passed:
                    total_rejected += 1
                    log_writer.writerow([class_name, fname, "", f"{ratio:.4f}", False])
                    continue

                out_path = os.path.join(out_dir, fname)
                refined_pil.save(out_path)
                total_generated += 1
                log_writer.writerow([class_name, fname, out_path, f"{ratio:.4f}", True])

    print(f"\nDone. {total_generated} refined glyphs saved to {OUTPUT_DIR}")
    print(f"{total_rejected} rejected. Log: {LOG_CSV_PATH}")