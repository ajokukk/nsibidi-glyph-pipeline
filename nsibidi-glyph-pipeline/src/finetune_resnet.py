"""
finetune_resnet.py  --  fine-tune the Adinkra ResNet-18 on synthetic Nsibidi glyphs (LPGS + CUT)

Question this answers
---------------------
Frozen-prototype tests showed that adding synthetic glyphs to the support set does NOT help.
This script asks the next question: does TRAINING the network on synthetic glyphs make its
features better for REAL glyphs?  Real test crops are never used for training.

Run it from the same folder as classify_detections.py (it re-uses that file's image loading and
--gray-norm preprocessing, so training and evaluation see exactly the same pixels).

Experiment A (synthetic only):
  python finetune_resnet.py --synth-dirs "D:\\dataset\\synthetic" "D:\\dataset\\Synthetic Stone Carving_cut" ^
      --test-dir "D:\\dataset\\real_split\\test" --resnet-ckpt adinkra_resnet18.pth ^
      --out finetuned_synth.pth --max-per-class 150 --epochs 4

Experiment B (synthetic + real train crops, real oversampled x5):
  add  --real-train-dir "D:\\dataset\\real_split\\train" --real-repeat 5  --out finetuned_synth_real.pth

Then measure with the SAME eval you used before (frozen prototypes from real train crops):
  python classify_detections.py eval --support-dir <real_split\\train> --query-dir <real_split\\test> ^
      --shots 1 2 5 --episodes 20 --backbone resnet --resnet-ckpt finetuned_synth.pth --gray-norm ^
      --out-csv ft_synth_eval.csv
Baseline to beat (Adinkra ResNet, gray-norm, real support): 42.0 / 55.1 / 67.4 % at 1/2/5 shots.

Notes
-----
* The final-epoch weights are saved. Test accuracy is printed each epoch for information only;
  the script does NOT pick the best epoch on the test set (that would leak the test set).
* --test-dir signs must also exist in the training folders; missing ones are reported.
* Needs a GPU for the default settings to be comfortable. On CPU use --max-per-class 100 --epochs 3.
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
from PIL import Image, ImageFilter

# Same loading / preprocessing as the evaluation script.
from classify_detections import (load_rgb, list_class_files, pad_to_square,
                                 normalise_polarity, load_resnet)

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Building the training list
# ---------------------------------------------------------------------------

def build_items(synth_dirs, real_dir, real_repeat, max_per_class, seed):
    """Returns (items, classes). items = [(path, class_index, is_real)].

    Synthetic images are capped at max_per_class per sign (random but seeded), split evenly
    across the synthetic folders so LPGS and CUT are both represented.
    """
    rng = random.Random(seed)
    synth = [list_class_files(d) for d in synth_dirs]
    real = list_class_files(real_dir) if real_dir else {}

    names = set()
    for s in synth:
        names |= set(s)
    names |= set(real)
    classes = sorted(names)
    idx = {c: i for i, c in enumerate(classes)}

    items = []
    for c in classes:
        pools = [s.get(c, []) for s in synth]
        pools = [p for p in pools if p]
        if pools:
            per_pool = max(1, max_per_class // len(pools)) if max_per_class else None
            for p in pools:
                p = list(p)
                rng.shuffle(p)
                if per_pool:
                    p = p[:per_pool]
                items += [(f, idx[c], False) for f in p]
        for f in real.get(c, []):
            items += [(f, idx[c], True)] * max(1, real_repeat)
    return items, classes


def class_counts(items, n):
    cnt = np.zeros(n, dtype=int)
    for _, y, _ in items:
        cnt[y] += 1
    return cnt


# ---------------------------------------------------------------------------
# Dataset / augmentation
# ---------------------------------------------------------------------------

def augment(im, rng):
    """Mild augmentation on the grey-normalised image; strokes stay recognisable."""
    w, h = im.size
    # random scale + shift crop
    s = rng.uniform(0.75, 1.0)
    cw, ch = int(w * s), int(h * s)
    x0 = rng.randint(0, w - cw)
    y0 = rng.randint(0, h - ch)
    im = im.crop((x0, y0, x0 + cw, y0 + ch))
    # small rotation, filled with the border colour (light ground after normalisation)
    ang = rng.uniform(-12, 12)
    im = im.rotate(ang, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    # random blur / noise-like softening
    if rng.random() < 0.35:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))
    # brightness / contrast jitter
    arr = np.asarray(im, dtype=np.float32)
    arr = (arr - 128.0) * rng.uniform(0.75, 1.25) + 128.0 + rng.uniform(-20, 20)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_dataset(items, train, seed):
    import torch
    from torch.utils.data import Dataset

    class GlyphDS(Dataset):
        def __init__(self):
            self.items = items
            self.train = train
            self.base_seed = seed
            self.epoch = 0

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, y, _ = self.items[i]
            im = load_rgb(path)
            im = normalise_polarity(im)                     # same as --gray-norm at eval time
            im = pad_to_square(im)
            if self.train:
                rng = random.Random(self.base_seed * 1000003 + self.epoch * 7919 + i)
                im = augment(im, rng)
                im = pad_to_square(im)
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            a = np.asarray(im, dtype=np.float32) / 255.0
            a = (a - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
            return torch.from_numpy(a).permute(2, 0, 1), y

    return GlyphDS()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(resnet_ckpt, n_classes, freeze, device):
    import torch
    net = load_resnet(resnet_ckpt, torch.device("cpu"))     # fc is Identity -> 512-d features
    net.fc = torch.nn.Linear(512, n_classes)
    torch.nn.init.normal_(net.fc.weight, std=0.01)
    torch.nn.init.zeros_(net.fc.bias)

    frozen = []
    if freeze in ("stem", "stem+1"):
        frozen += [net.conv1, net.bn1, net.layer1]
    if freeze == "stem+1":
        frozen += [net.layer2]
    for m in frozen:
        for p in m.parameters():
            p.requires_grad = False
    return net.to(device), frozen


def set_train_mode(net, frozen):
    net.train()
    for m in frozen:            # keep frozen blocks' BatchNorm statistics fixed
        m.eval()


def evaluate(net, loader, device, n_classes):
    import torch
    net.eval()
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)
    with torch.no_grad():
        for x, y in loader:
            pred = net(x.to(device)).argmax(1).cpu().numpy()
            for p, t in zip(pred, y.numpy()):
                total[t] += 1
                correct[t] += int(p == t)
    seen = total > 0
    micro = correct.sum() / max(total.sum(), 1)
    macro = (correct[seen] / total[seen]).mean() if seen.any() else float("nan")
    return micro, macro


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dirs", nargs="*", default=[],
                    help="synthetic glyph folders (one sub-folder per sign): LPGS and/or CUT. "
                         "May be omitted for a real-only control (use --real-train-dir)")
    ap.add_argument("--real-train-dir", default=None, help="optional real TRAIN crops (never the test folder)")
    ap.add_argument("--real-repeat", type=int, default=5, help="oversampling factor for real train crops")
    ap.add_argument("--test-dir", default=None, help="real TEST crops, only for reporting accuracy")
    ap.add_argument("--resnet-ckpt", required=True, help="Adinkra ResNet-18 .pth to start from")
    ap.add_argument("--out", default="finetuned_resnet18.pth")
    ap.add_argument("--max-per-class", type=int, default=150,
                    help="max synthetic images per sign per epoch pool (0 = no cap)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--freeze", choices=["none", "stem", "stem+1"], default="stem",
                    help="freeze early layers (faster, keeps Adinkra low-level features)")
    ap.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + ("" if device.type == "cuda" else "   (CPU: consider --max-per-class 100 --epochs 3)"))

    if not a.synth_dirs and not a.real_train_dir:
        raise SystemExit("Give --synth-dirs and/or --real-train-dir")
    items, classes = build_items(a.synth_dirs, a.real_train_dir, a.real_repeat, a.max_per_class, a.seed)
    cnt = class_counts(items, len(classes))
    n_real = sum(1 for _, _, r in items if r)
    print(f"{len(classes)} signs, {len(items)} training images ({len(items) - n_real} synthetic, {n_real} real after repeat)")
    if n_real == 0:
        print("Training on SYNTHETIC ONLY.")
    for c, n in zip(classes, cnt):
        if n == 0:
            print(f"  WARNING: no training images for {c}")

    test_loader = None
    if a.test_dir:
        test_files = list_class_files(a.test_dir)
        missing = [c for c in test_files if c not in classes]
        if missing:
            print(f"  WARNING: test signs with no training class (will count as wrong): {missing}")
        cidx = {c: i for i, c in enumerate(classes)}
        # test signs absent from training get label -1 -> never predicted -> counted wrong
        t_items = [(f, cidx.get(c, -1), False) for c, fs in test_files.items() for f in fs]
        t_items = [(f, y if y >= 0 else len(classes), r) for f, y, r in t_items]
        test_ds = make_dataset(t_items, train=False, seed=a.seed)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=a.workers)
        print(f"Test crops (reporting only): {len(t_items)}")

    train_ds = make_dataset(items, train=True, seed=a.seed)
    # class-balanced sampling: every sign is drawn equally often regardless of folder size
    w = np.array([1.0 / max(cnt[y], 1) for _, y, _ in items], dtype=np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(items), replacement=True)
    loader = DataLoader(train_ds, batch_size=a.batch_size, sampler=sampler,
                        num_workers=a.workers, drop_last=len(items) > a.batch_size)

    net, frozen = build_model(a.resnet_ckpt, len(classes), a.freeze, device)
    head = list(net.fc.parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in net.parameters() if p.requires_grad and id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.head_lr}],
                            weight_decay=1e-4)
    steps = a.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr, a.head_lr], total_steps=max(steps, 1), pct_start=0.15)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Training {a.epochs} epochs, {len(loader)} steps/epoch, freeze={a.freeze}")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        train_ds.epoch = ep
        set_train_mode(net, frozen)
        run_loss, run_acc, seen = 0.0, 0.0, 0
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = net(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            run_acc += (out.argmax(1) == y).float().sum().item()
            seen += len(y)
            if step % 20 == 0 or step == len(loader):
                print(f"  epoch {ep} step {step}/{len(loader)}  loss {run_loss / seen:.3f}  "
                      f"train-acc {run_acc / seen:.3f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        msg = f"epoch {ep}: train loss {run_loss / seen:.3f}, train acc {run_acc / seen:.3f}"
        if test_loader is not None:
            micro, macro = evaluate(net, test_loader, device, len(classes) + 1)
            msg += f"  |  real-test accuracy {micro:.3f} (macro {macro:.3f})  [information only]"
        print(msg, flush=True)

    # Save the FINAL weights. load_resnet() in classify_detections.py reads this file directly.
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, a.out)
    meta = {"classes": classes, "synth_dirs": a.synth_dirs, "real_train_dir": a.real_train_dir,
            "epochs": a.epochs, "freeze": a.freeze, "max_per_class": a.max_per_class,
            "start_checkpoint": a.resnet_ckpt}
    with open(os.path.splitext(a.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {a.out}  (classes + settings in {os.path.splitext(a.out)[0]}_meta.json)")
    print("Next: run classify_detections.py eval with --resnet-ckpt " + a.out)


if __name__ == "__main__":
    main()
