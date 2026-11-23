"""
CUT (Contrastive Unpaired Translation) training for realistic material transfer
on Nsibidi glyphs. Sized for T4 capacity (16GB VRAM).
"""

import itertools
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Local modular imports
from glyph_datasets import (
    UnpairedGlyphDataset,
    get_real_train_holdout_split,
    list_image_paths,
    list_style_reference_paths,
)
from diffaugment import diff_augment
from models import Discriminator, Generator, init_weights
from patchnce import PatchNCELoss, PatchSampleMLP
from ssim_loss import SSIMLoss
from style_loss import ColorLoss, StyleLoss

# ------------------------------------------------------------------
# CONFIG -- Edit for local PC paths (e.g. from C:/...)
# ------------------------------------------------------------------
# ROOT = "<ROOT>" # Local usage
DRIVE_ROOT = "<ROOT>"  # Paste the config you had here
SYNTHETIC_DIR = f"{DRIVE_ROOT}/dataset/synthetic"
REAL_DIR = f"{DRIVE_ROOT}/dataset/assume"
STYLE_REFERENCE_DIR = f"{DRIVE_ROOT}/dataset/style_refs/stone_carving"

STYLE_NAME = "stone_carving_test"
OUTPUT_DIR = f"{DRIVE_ROOT}/cut_output_{STYLE_NAME}"
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

IMAGE_SIZE = 100
CHANNELS = 3
BASE_CHANNELS = 64  # Colab T4 capacity, original CycleGAN capacities
N_RESIDUAL_BLOCKS = 9  # CycleGAN depth
BATCH_SIZE = 4
NUM_EPOCHS = 100
STEPS_PER_EPOCH = 400
LR = 2e-4
BETA1, BETA2 = 0.5, 0.999

# NCE config
NCE_LAYERS = [3, 6, 9, 12, 15]  # Span encoder bottleneck
NUM_PATCHES = 256

LAMBDA_GAN = 1.0
LAMBDA_NCE = 1.0
LAMBDA_STRUCT = 1.0  # SSIM preservation
LAMBDA_STYLE = 50.0  # Gram texture
LAMBDA_COLOR = 10.0  # Explicit statistics
USE_NCE_IDENTITY = True  # Stability term

SAMPLE_EVERY = 1
CHECKPOINT_EVERY = 1  # Less frequent locally than Colab
SEED = 42

RESUME_FROM_EPOCH = None
# ------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Style target: {STYLE_NAME}")

    # --- Data Prep ---
    domain_a_paths = list_image_paths(SYNTHETIC_DIR)
    if not domain_a_paths:
        raise ValueError(f"Domain A empty at {SYNTHETIC_DIR}")

    real_train_paths, _, real_holdout_paths, _, _ = get_real_train_holdout_split(REAL_DIR)

    if isinstance(STYLE_REFERENCE_DIR, (list, tuple)):
        style_paths = []
        for d in STYLE_REFERENCE_DIR: style_paths += list_style_reference_paths(d)
    else:
        style_paths = list_style_reference_paths(STYLE_REFERENCE_DIR)

    domain_b_paths = real_train_paths + style_paths
    if not domain_b_paths:
        raise ValueError("Domain B empty.")

    print(f"Domain A: {len(domain_a_paths)} images")
    print(f"Domain B total: {len(domain_b_paths)}")

    dataset = UnpairedGlyphDataset(
        domain_a_paths, domain_b_paths, image_size=IMAGE_SIZE, channels=CHANNELS,
        steps_per_epoch=STEPS_PER_EPOCH,
    )
    # num_workers=2 works now because lambda is removed.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    fixed_sample_a, _ = next(iter(dataloader))
    # Standard denorm tensor for visualization [BATCH_SIZE, CHANNELS, 1, 1]
    denorm_mean = torch.tensor([0.5] * CHANNELS).view(CHANNELS, 1, 1).to(device)
    denorm_std = torch.tensor([0.5] * CHANNELS).view(CHANNELS, 1, 1).to(device)

    # --- Models ---
    G = Generator(CHANNELS, BASE_CHANNELS, N_RESIDUAL_BLOCKS).to(device)
    D = Discriminator(CHANNELS, BASE_CHANNELS).to(device)
    patch_mlp = PatchSampleMLP(num_patches=NUM_PATCHES).to(device)

    G.apply(init_weights)
    D.apply(init_weights)

    # --- Losses ---
    criterion_gan = nn.MSELoss()
    criterion_nce = PatchNCELoss()
    criterion_struct = SSIMLoss(data_range=2.0)
    criterion_style = StyleLoss().to(device)
    criterion_color = ColorLoss()

    # --- Optimizers ---
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_MLP = None  # created lazily


    def lr_lambda(epoch, decay_start=NUM_EPOCHS // 2):
        if epoch < decay_start: return 1.0
        return 1.0 - (epoch - decay_start) / (NUM_EPOCHS - decay_start)


    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    start_epoch = 1
    if RESUME_FROM_EPOCH is not None:
        print(f"Resuming epoch {RESUME_FROM_EPOCH}...")
        G.load_state_dict(
            torch.load(os.path.join(CHECKPOINT_DIR, f"G_epoch{RESUME_FROM_EPOCH}.pth"), map_location=device))
        D.load_state_dict(
            torch.load(os.path.join(CHECKPOINT_DIR, f"D_epoch{RESUME_FROM_EPOCH}.pth"), map_location=device))
        mlp_path = os.path.join(CHECKPOINT_DIR, f"MLP_epoch{RESUME_FROM_EPOCH}.pth")
        if os.path.exists(mlp_path): patch_mlp.load_state_dict(torch.load(mlp_path, map_location=device))
        start_epoch = RESUME_FROM_EPOCH

    print(f"\n--- Starting CUT training ({STYLE_NAME}) ---")

    for epoch in range(start_epoch, NUM_EPOCHS):
        running = {"G": 0.0, "D": 0.0, "nce": 0.0}
        n_batches = 0

        G.train()
        for real_A, real_B in dataloader:
            real_A, real_B = real_A.to(device), real_B.to(device)

            with torch.no_grad():
                patch_shape = D(real_A).shape  # Output shape needed for targets
            valid = torch.ones(patch_shape, device=device)
            fake = torch.zeros(patch_shape, device=device)

            # --- Train G ---
            opt_G.zero_grad()
            fake_B = G(real_A)
            loss_gan = criterion_gan(D(diff_augment(fake_B)), valid)

            # PatchNCE content
            nce_total = 0.0
            for b in range(real_A.size(0)):
                feats_k = G(real_A[b:b + 1], layers=NCE_LAYERS, encode_only=True)
                feats_q = G(fake_B[b:b + 1], layers=NCE_LAYERS, encode_only=True)
                embed_k, patch_ids = patch_mlp(feats_k)
                embed_q, _ = patch_mlp(feats_q, patch_ids=patch_ids)
                nce_total = nce_total + criterion_nce(embed_q, embed_k)
            loss_nce = nce_total / real_A.size(0)

            if USE_NCE_IDENTITY:
                idt_B = G(real_B)
                nce_idt_total = 0.0
                for b in range(real_B.size(0)):
                    feats_k = G(real_B[b:b + 1], layers=NCE_LAYERS, encode_only=True)
                    feats_q = G(idt_B[b:b + 1], layers=NCE_LAYERS, encode_only=True)
                    embed_k, patch_ids = patch_mlp(feats_k)
                    embed_q, _ = patch_mlp(feats_q, patch_ids=patch_ids)
                    nce_idt_total = nce_idt_total + criterion_nce(embed_q, embed_k)
                loss_nce = (loss_nce + nce_idt_total / real_B.size(0)) / 2

            loss_G = LAMBDA_GAN * loss_gan + LAMBDA_NCE * loss_nce
            # Add other losses optionally
            loss_G = loss_G + LAMBDA_STRUCT * criterion_struct(fake_B, real_A)
            loss_G = loss_G + LAMBDA_STYLE * criterion_style(fake_B, real_B)
            loss_G = loss_G + LAMBDA_COLOR * criterion_color(fake_B, real_B)

            loss_G.backward()
            opt_G.step()

            # MLP Opt lazy init
            if opt_MLP is None and len(list(patch_mlp.parameters())) > 0:
                opt_MLP = torch.optim.Adam(patch_mlp.parameters(), lr=LR, betas=(BETA1, BETA2))
            if opt_MLP is not None:
                opt_MLP.step()
                opt_MLP.zero_grad()

            # --- Train D ---
            opt_D.zero_grad()
            loss_real = criterion_gan(D(diff_augment(real_B)), valid)
            loss_fake = criterion_gan(D(diff_augment(fake_B.detach())), fake)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            opt_D.step()

            running["G"] += loss_G.item();
            running["D"] += loss_D.item();
            running["nce"] += loss_nce.item()
            n_batches += 1

        sched_G.step();
        sched_D.step()

        print(
            f"Epoch {epoch + 1:03d} | G: {running['G'] / n_batches:.4f} D: {running['D'] / n_batches:.4f} NCE: {running['nce'] / n_batches:.4f}")

        if (epoch + 1) % SAMPLE_EVERY == 0 or epoch == 0:
            G.eval()
            with torch.no_grad():
                sample_fake_b = G(fixed_sample_a.to(device))
                grid = torch.cat([fixed_sample_a.to(device), sample_fake_b], dim=0)
                # Apply standard denorm
                grid = (grid * denorm_std + denorm_mean).clamp(0, 1)
                save_image(grid.cpu(), os.path.join(SAMPLES_DIR, f"epoch_{epoch + 1:03d}.png"), nrow=BATCH_SIZE)

        if (epoch + 1) % CHECKPOINT_EVERY == 0 or (epoch + 1) == NUM_EPOCHS:
            torch.save(G.state_dict(), os.path.join(CHECKPOINT_DIR, f"G_epoch{epoch + 1}.pth"))
            torch.save(D.state_dict(), os.path.join(CHECKPOINT_DIR, f"D_epoch{epoch + 1}.pth"))
            torch.save(patch_mlp.state_dict(), os.path.join(CHECKPOINT_DIR, f"MLP_epoch{epoch + 1}.pth"))
            print(f"Checkpoints epoch {epoch + 1}")