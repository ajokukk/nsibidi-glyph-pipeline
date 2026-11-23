"""
Local Faster R-CNN glyph detector: train + predict, one file.
================================================================

Trains on the composited detection pages made by compose_detection_pages.py
(COCO-format images + boxes built from your CUT-refined glyph crops).

Setup (once):
    pip install torch torchvision pillow numpy
    (for an NVIDIA GPU, install the CUDA build of torch from pytorch.org)

Workflow:
    1. python compose_detection_pages.py          # builds detection_pages/
    2. python train_rcnn_local.py train           # trains, saves best/last checkpoints
    3. python train_rcnn_local.py predict --image some_plate.png

Edit CONFIG below once, or override any value on the command line, e.g.:
    python train_rcnn_local.py train --arch resnet50 --batch-size 2 --epochs 30

First run with PRETRAINED=True downloads COCO-pretrained detector weights
(about 75 MB for mobilenet, 160 MB for resnet50) into torch's cache folder;
needs internet that one time only.

Model choice / memory:
    mobilenet : light, fits a small GPU (2-4 GB) and is OK on CPU. Default.
    resnet50  : more accurate, needs roughly 6+ GB VRAM at batch size 2.
If you hit "CUDA out of memory": lower BATCH_SIZE, lower MIN_SIZE/MAX_SIZE,
or switch to mobilenet.
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF
from torchvision.ops import box_iou, nms
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# CONFIG -- edit once (or override via command-line flags)
# ---------------------------------------------------------------------------

DETECTION_PAGES_DIR = r"<ROOT>\nsibidi_training_data\detection_pages"
OUTPUT_DIR = r"<ROOT>\nsibidi_training_data\rcnn_output_new"

ARCH = "mobilenet"        # "mobilenet" or "resnet50"
PRETRAINED = True         # start from COCO-pretrained detector weights
SINGLE_CLASS = True       # True: detector finds "glyph" only (a classifier names it);
                          # False: one detector class per sign. Override with --single-class.

# The detector internally rescales every page so its short side is MIN_SIZE
# (long side capped at MAX_SIZE). Pages from compose_detection_pages.py are
# 1200x900; glyphs are small, so don't shrink much below the native size.
MIN_SIZE = 800
MAX_SIZE = 1200

VAL_FRACTION = 0.1
BATCH_SIZE = 4
NUM_EPOCHS = 20
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 8          # decay LR every this many epochs
LR_GAMMA = 0.1

NUM_WORKERS = 2           # set 0 if you get multiprocessing errors on Windows
USE_AMP = True            # mixed precision; only used when a CUDA GPU is present
EVAL_EVERY = 1            # compute validation mAP every N epochs
SEED = 42

# ---------------------------------------------------------------------------


class CocoDetectionDataset(Dataset):
    """Reads exactly the COCO-style JSON compose_detection_pages.py writes."""

    def __init__(self, images_dir, coco_json_path, image_ids=None, single_class=False):
        self.images_dir = images_dir
        self.single_class = single_class
        with open(coco_json_path) as f:
            coco = json.load(f)

        self.images_by_id = {im["id"]: im for im in coco["images"]}
        self.anns_by_image = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        categories = sorted(coco["categories"], key=lambda c: c["id"])
        if single_class:
            # Every sign is just "glyph": the detector only has to FIND signs;
            # naming them is left to the classifier (prototypical network).
            self.class_names = {1: "glyph"}
            self.num_classes = 2                                     # glyph + background
        else:
            self.class_names = {c["id"]: c["name"] for c in categories}  # 1-indexed ids
            self.num_classes = len(categories) + 1                       # +1 background
        self.image_ids = image_ids if image_ids is not None else sorted(self.images_by_id)

    def ids_for_split(self, split):
        """Page ids tagged with this split by compose_detection_pages.py (or [])."""
        return [i for i in sorted(self.images_by_id)
                if self.images_by_id[i].get("split") == split]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        info = self.images_by_id[image_id]
        img = Image.open(os.path.join(self.images_dir, info["file_name"])).convert("RGB")

        boxes, labels = [], []
        for ann in self.anns_by_image.get(image_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(1 if self.single_class else ann["category_id"])

        boxes_t = (torch.as_tensor(boxes, dtype=torch.float32)
                   if boxes else torch.zeros((0, 4), dtype=torch.float32))
        labels_t = (torch.as_tensor(labels, dtype=torch.int64)
                    if labels else torch.zeros((0,), dtype=torch.int64))
        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([image_id])}
        return TF.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(arch, num_classes, pretrained, min_size, max_size):
    if arch == "mobilenet":
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights, weights_backbone=None,
            min_size=min_size, max_size=max_size)
    elif arch == "resnet50":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights, weights_backbone=None,
            min_size=min_size, max_size=max_size)
    else:
        raise ValueError(f"Unknown ARCH '{arch}' (use 'mobilenet' or 'resnet50')")

    # Swap the COCO classification head for one sized to our glyph classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Evaluation: mAP@0.5 (VOC-style all-point interpolation, no extra packages)
# ---------------------------------------------------------------------------

def _average_precision(records, n_gt):
    if n_gt == 0:
        return None
    if not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([1.0 if r[1] else 0.0 for r in records])
    fp = np.cumsum([0.0 if r[1] else 1.0 for r in records])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def evaluate_map50(model, loader, device, num_classes, iou_thresh=0.5):
    model.eval()
    records = {c: [] for c in range(1, num_classes)}
    n_gt = {c: 0 for c in range(1, num_classes)}

    for images, targets in loader:
        outputs = model([im.to(device) for im in images])
        for out, tgt in zip(outputs, targets):
            pb, ps, pl = out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()
            gb, gl = tgt["boxes"], tgt["labels"]
            for c in gl.tolist():
                n_gt[c] += 1
            for c in range(1, num_classes):
                cb, cs = pb[pl == c], ps[pl == c]
                order = cs.argsort(descending=True)
                cb, cs = cb[order], cs[order]
                g = gb[gl == c]
                matched = torch.zeros(len(g), dtype=torch.bool)
                ious = box_iou(cb, g) if len(cb) and len(g) else None
                for i in range(len(cb)):
                    is_tp = False
                    if ious is not None:
                        best, j = ious[i].max(0)
                        if best >= iou_thresh and not matched[j]:
                            matched[j] = True
                            is_tp = True
                    records[c].append((cs[i].item(), is_tp))

    aps = [ap for c in range(1, num_classes)
           if (ap := _average_precision(records[c], n_gt[c])) is not None]
    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train(args):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cpu":
        print("  No GPU found -- training on CPU works but is slow; consider --arch mobilenet "
              "and a small --epochs for a first run.")

    images_dir = os.path.join(args.pages_dir, "images")
    ann_path = os.path.join(args.pages_dir, "annotations_coco.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"{ann_path} not found -- run compose_detection_pages.py first.")

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    probe = CocoDetectionDataset(images_dir, ann_path, single_class=args.single_class)
    all_ids = probe.image_ids

    # Prefer the split written by compose_detection_pages.py: its validation pages
    # are built from glyph files the training pages never use. Fall back to a
    # random page split for JSON files made by the older compositor.
    train_ids, val_ids = probe.ids_for_split("train"), probe.ids_for_split("val")
    if train_ids and val_ids:
        print("Using the train/val split stored in annotations_coco.json "
              "(validation pages use held-out glyph files).")
    else:
        n_val = max(1, int(len(all_ids) * VAL_FRACTION))
        perm = torch.randperm(len(all_ids), generator=torch.Generator().manual_seed(SEED)).tolist()
        val_ids = [all_ids[i] for i in perm[:n_val]]
        train_ids = [all_ids[i] for i in perm[n_val:]]
        print("No stored split found -- using a random page split (glyph crops may be shared "
              "between train and val, so val mAP will be optimistic).")
    train_ds = CocoDetectionDataset(images_dir, ann_path, train_ids, single_class=args.single_class)
    val_ds = CocoDetectionDataset(images_dir, ann_path, val_ids, single_class=args.single_class)
    print(f"Pages -> train: {len(train_ds)}  val: {len(val_ds)}  | glyph classes: {train_ds.num_classes - 1}"
          + ("  (single-class mode)" if args.single_class else ""))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    model = build_model(args.arch, train_ds.num_classes, args.pretrained,
                        args.min_size, args.max_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    scaler = make_scaler(use_amp)

    start_epoch, best_map = 1, -1.0
    last_path = os.path.join(ckpt_dir, "last.pth")
    if args.resume:
        if not os.path.exists(last_path):
            raise FileNotFoundError(f"--resume given but {last_path} does not exist.")
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        best_map = ck.get("best_map", -1.0)
        start_epoch = ck["epoch"] + 1
        print(f"Resumed after epoch {ck['epoch']} (best mAP50 so far: {best_map:.4f})")

    meta = {
        "arch": args.arch, "num_classes": train_ds.num_classes,
        "class_names": train_ds.class_names,
        "min_size": args.min_size, "max_size": args.max_size,
        "single_class": args.single_class,
    }

    print(f"\n--- Training {args.arch} for {args.epochs} epochs ---")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running, n_batches, t0 = 0.0, 0, time.time()
        for images, targets in train_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = sum(model(images, targets).values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            n_batches += 1
        scheduler.step()

        msg = (f"Epoch {epoch:03d}/{args.epochs} | train_loss {running / max(n_batches, 1):.4f} "
               f"| lr {scheduler.get_last_lr()[0]:.5f} | {time.time() - t0:.0f}s")

        val_map = None
        if epoch % EVAL_EVERY == 0 or epoch == args.epochs:
            val_map = evaluate_map50(model, val_loader, device, train_ds.num_classes)
            msg += f" | val mAP@0.5 {val_map:.4f}"
        print(msg)

        ck = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "scheduler": scheduler.state_dict(), "epoch": epoch,
              "best_map": max(best_map, val_map if val_map is not None else -1.0), "meta": meta}
        torch.save(ck, last_path)
        if val_map is not None and val_map > best_map:
            best_map = val_map
            torch.save({"model": model.state_dict(), "meta": meta, "epoch": epoch,
                        "val_map50": val_map}, os.path.join(ckpt_dir, "best.pth"))
            print(f"  new best (mAP@0.5 {best_map:.4f}) -> best.pth")

    print(f"\nDone. Best validation mAP@0.5: {best_map:.4f}")
    print(f"Checkpoints in: {ckpt_dir}  (best.pth = best val mAP, last.pth = resumable)")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or os.path.join(args.output_dir, "checkpoints", "best.pth")
    ck = torch.load(ckpt_path, map_location=device)
    meta = ck["meta"]
    model = build_model(meta["arch"], meta["num_classes"], False,
                        meta["min_size"], meta["max_size"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    class_names = {int(k): v for k, v in meta["class_names"].items()}

    img = Image.open(args.image).convert("RGB")
    out = model([TF.to_tensor(img).to(device)])[0]

    boxes, scores, labels = out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()
    keep = scores >= args.threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    # Class-agnostic NMS: the detector's own NMS runs per class, so one glyph can
    # keep two boxes with different labels. Keep only the best box per location.
    if len(boxes) and args.nms_iou < 1.0:
        k = nms(boxes, scores, args.nms_iou)
        boxes, scores, labels = boxes[k], scores[k], labels[k]

    draw = ImageDraw.Draw(img)
    rows = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(v) for v in box]
        name = class_names.get(int(label), str(int(label)))
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1, max(0, y1 - 11)), f"{name} {score:.2f}", fill=(255, 0, 0))
        rows.append((name, float(score), x1, y1, x2, y2))

    out_path = args.out or os.path.splitext(args.image)[0] + "_detected.png"
    img.save(out_path)
    print(f"{len(rows)} detections >= {args.threshold} -> {out_path}")
    for r in rows[:20]:
        print(f"  {r[0]:<20} {r[1]:.2f}  box=({r[2]:.0f},{r[3]:.0f},{r[4]:.0f},{r[5]:.0f})")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Local Faster R-CNN glyph detector")
    sub = ap.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="train the detector")
    t.add_argument("--pages-dir", default=DETECTION_PAGES_DIR)
    t.add_argument("--output-dir", default=OUTPUT_DIR)
    t.add_argument("--arch", default=ARCH, choices=["mobilenet", "resnet50"])
    t.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    t.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    t.add_argument("--lr", type=float, default=LR)
    t.add_argument("--min-size", type=int, default=MIN_SIZE)
    t.add_argument("--max-size", type=int, default=MAX_SIZE)
    t.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    t.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                   default=PRETRAINED, help="random init instead of COCO-pretrained")
    t.add_argument("--resume", action="store_true", help="continue from checkpoints/last.pth")
    t.add_argument("--single-class", action=argparse.BooleanOptionalAction, default=SINGLE_CLASS,
                   help="--single-class: one 'glyph' class; --no-single-class: one class per sign")

    p = sub.add_parser("predict", help="run the trained detector on an image")
    p.add_argument("--image", required=True)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--checkpoint", default=None, help="defaults to <output-dir>/checkpoints/best.pth")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--nms-iou", type=float, default=0.5,
                   help="class-agnostic NMS overlap; boxes overlapping more than this are merged "
                        "(1.0 turns it off)")
    p.add_argument("--out", default=None)

    args = ap.parse_args()
    if args.cmd is None:           # bare `python train_rcnn_local.py` just trains
        args = ap.parse_args(["train"])
    (train if args.cmd == "train" else predict)(args)


if __name__ == "__main__":
    main()
