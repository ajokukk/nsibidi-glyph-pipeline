"""
Classification stage: name the glyphs the detector finds  (prototypical network)
================================================================================

The single-class detector (train_rcnn_local.py) says WHERE a glyph is. This script
says WHICH sign it is, with a few-shot prototypical classifier on frozen features:

    detector box -> crop -> frozen backbone embedding -> nearest class prototype

Backbone (--backbone):
    dino    frozen DINOv2 (default, small ViT-S/14, 384-d)
    resnet  your Adinkra-pretrained ResNet-18 (512-d)         needs --resnet-ckpt <path>
            (use --resnet-ckpt imagenet for the plain ImageNet ResNet-18 control)
    both    DINOv2 + ResNet-18 concatenated (each L2-normalised first)
Nothing is fine-tuned. Class prototypes are the mean embedding of each sign's
labelled example images; a crop gets the class with the highest cosine similarity.

Two commands
------------
1) classify  - detect on an image, name every box, write a labelled picture + CSV:

    python classify_detections.py classify --image plate.png ^
        --support-dir "C:\\...\\real_glyphs" "C:\\...\\dataset\\synthetic" ^
        --shots 20 --threshold 0.3

   --support-dir: one or more folders with one sub-folder per sign (the same layout
   as your glyph folders). --shots is how many images per sign are taken from EACH
   folder to build the prototypes. Prototypes can be cached with --prototypes file.npz.

   The plate has more signs than the 20 you have names for, so every box is forced
   into the nearest of your classes. Use --min-sim (e.g. 0.5) to label boxes whose
   best similarity is low as "unknown"; the similarity values are printed and saved
   in the CSV so you can choose the number from your own data.

2) eval  - k-shot accuracy on labelled glyph crops (the thesis metric):

    python classify_detections.py eval ^
        --support-dir "C:\\...\\real_glyphs_train" --query-dir "C:\\...\\real_glyphs_test" ^
        --shots 1 2 5 10 --episodes 20 --backbone dino

   Each episode draws k support images per sign to make prototypes and classifies
   the query images; the script reports macro accuracy (mean over signs) as mean +/- std
   over episodes. Keep support and query files DISJOINT (the script also drops any query
   file that was chosen as a support image). Run it once per --backbone for the ablation.

Setup (once):  pip install torch torchvision pillow numpy
First run downloads DINOv2 code and weights from Meta's official repository
(torch.hub, facebookresearch/dinov2) into torch's cache folder; needs internet that
one time only.
"""

import os
import csv
import sys
import random
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ---------------------------------------------------------------------------
# CONFIG -- edit once (or override on the command line)
# ---------------------------------------------------------------------------

DETECTOR_CKPT = r"<ROOT>\nsibidi_training_data\rcnn_output_new\checkpoints\best.pth"
SUPPORT_DIRS = [r"<ROOT>\dataset\real" , r"<ROOT>\dataset\synthetic"]            # e.g. [r"C:\...\real_glyphs", r"C:\...\dataset\synthetic"]
RESNET_CKPT = r"<HOME>\PycharmProjects\NsibidiDataEfficientPipeline\adinkra_resnet18.pth"           # your Adinkra-pretrained ResNet-18 weights (.pth), if you have them
DINO_MODEL = "dinov2_vits14"  # also: dinov2_vitb14 (bigger, slower)
IMG_SIZE = 224               # crops are padded to a square, then resized to this
SEED = 42

# ---------------------------------------------------------------------------

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# Images and files
# ---------------------------------------------------------------------------

def load_rgb(path):
    """Open an image as RGB; transparent pixels become white (not black)."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        return bg.convert("RGB")
    return im.convert("RGB")


def list_class_files(root):
    """{class_name: [image paths]} for a folder with one sub-folder per class."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Folder not found: {root}")
    out = {}
    for c in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        cdir = os.path.join(root, c)
        files = sorted(os.path.join(cdir, f) for f in os.listdir(cdir)
                       if f.lower().endswith(IMG_EXT))
        if files:
            out[c] = files
    return out


def pad_to_square(img):
    """Pad to a square with the border colour so the glyph is not stretched."""
    w, h = img.size
    if w == h:
        return img
    arr = np.asarray(img)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
    fill = tuple(int(v) for v in np.median(border, axis=0))
    s = max(w, h)
    out = Image.new("RGB", (s, s), fill)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def normalise_polarity(im):
    """Colour-free view of a crop: grayscale, stretched contrast, dark strokes on a light ground.

    Your glyphs come as light strokes on dark blue tiles, black ink on white, blue pen on grey
    paper, and so on. Without this the embedding can end up describing the background colour
    instead of the sign's shape. It is applied to support and query images alike (--gray-norm).
    """
    g = ImageOps.autocontrast(im.convert("L"), cutoff=2)
    if np.median(np.asarray(g)) < 128:      # dark ground -> invert so strokes are dark
        g = ImageOps.invert(g)
    return g.convert("RGB")


def l2_normalise(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# Frozen embedding backbone (torch is imported lazily so the numpy logic below
# can be used and tested without it)
# ---------------------------------------------------------------------------

def load_resnet(ckpt_path, device):
    import torch
    import torchvision
    if not ckpt_path:
        raise ValueError("--resnet-ckpt is required for --backbone resnet or both "
                         "(pass the path to your Adinkra weights, or the word 'imagenet' "
                         "for the plain ImageNet ResNet-18 control)")
    if ckpt_path == "imagenet":
        # Control for the ablation: ResNet-18 with ImageNet weights only, no Adinkra training.
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
        return net.to(device).eval()
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck
    if isinstance(ck, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in ck and isinstance(ck[key], dict):
                state = ck[key]
                break
    if isinstance(state, dict):
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        n_out = state["fc.weight"].shape[0] if "fc.weight" in state else 1000
        net = torchvision.models.resnet18(weights=None, num_classes=n_out)
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  WARNING: ResNet-18 weights did not match exactly "
                  f"(missing {len(missing)}, unexpected {len(unexpected)} keys)")
    else:                       # a whole pickled model
        net = state
    net.fc = torch.nn.Identity()    # 512-d pooled features instead of the Adinkra class scores
    return net.to(device).eval()


class Embedder:
    """PIL images -> L2-normalised numpy features, from frozen backbones."""

    def __init__(self, backbone="dino", dino_model=DINO_MODEL, resnet_ckpt=RESNET_CKPT,
                 gray_norm=False):
        import torch
        self.torch = torch
        self.gray_norm = gray_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dino = self.resnet = None
        if backbone in ("dino", "both"):
            try:
                self.dino = torch.hub.load("facebookresearch/dinov2", dino_model, trust_repo=True)
            except TypeError:       # very old torch without trust_repo
                self.dino = torch.hub.load("facebookresearch/dinov2", dino_model)
            self.dino = self.dino.to(self.device).eval()
        if backbone in ("resnet", "both"):
            self.resnet = load_resnet(resnet_ckpt, self.device)
        if self.dino is None and self.resnet is None:
            raise ValueError(f"Unknown backbone '{backbone}' (use dino, resnet or both)")
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _tensor(self, pil_images):
        torch = self.torch
        arrs = []
        for im in pil_images:
            im = im.convert("RGB")
            if self.gray_norm:
                im = normalise_polarity(im)
            im = pad_to_square(im).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            arrs.append(np.asarray(im, dtype=np.float32) / 255.0)
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to(self.device)
        return (x - self.mean) / self.std

    def __call__(self, pil_images):
        torch = self.torch
        x = self._tensor(pil_images)
        feats = []
        with torch.no_grad():
            for net in (self.dino, self.resnet):
                if net is not None:
                    f = net(x)
                    f = f / (f.norm(dim=1, keepdim=True) + 1e-12)
                    feats.append(f)
            out = torch.cat(feats, dim=1)
            out = out / (out.norm(dim=1, keepdim=True) + 1e-12)
        return out.cpu().numpy().astype(np.float32)


def embed_files(embedder, files, batch_size=32):
    """Embed image files; returns (features [N, D], list of the files that loaded)."""
    out, ok, batch = [], [], []
    for p in files:
        try:
            batch.append(load_rgb(p))
            ok.append(p)
        except Exception as ex:
            print(f"  skipped {p}: {ex}")
            continue
        if len(batch) >= batch_size:
            out.append(embedder(batch))
            batch = []
    if batch:
        out.append(embedder(batch))
    return (np.concatenate(out, 0) if out else np.zeros((0, 0), np.float32)), ok


# ---------------------------------------------------------------------------
# Prototypes
# ---------------------------------------------------------------------------

def sample_support(dirs, shots, seed):
    """Up to `shots` random images per class from EACH folder (0 = use all)."""
    rng = random.Random(seed)
    merged = {}
    for d in dirs:
        for c, files in list_class_files(d).items():
            files = list(files)
            rng.shuffle(files)
            merged.setdefault(c, []).extend(files[:shots] if shots else files)
    return merged


def build_prototypes(embedder, class_files):
    names, protos = [], []
    for c in sorted(class_files):
        embs, ok = embed_files(embedder, class_files[c])
        if len(ok) == 0:
            print(f"  WARNING: no usable images for class '{c}', skipped")
            continue
        names.append(c)
        protos.append(l2_normalise(embs.mean(axis=0)))
        print(f"  prototype '{c}' from {len(ok)} image(s)")
    if not names:
        raise ValueError("No prototypes could be built -- check --support-dir")
    return names, np.stack(protos)


def get_prototypes(embedder, args):
    tag = f"{args.backbone}:{args.dino_model}:{args.shots}:{args.seed}:{int(args.gray_norm)}"
    if args.prototypes and os.path.exists(args.prototypes):
        z = np.load(args.prototypes, allow_pickle=False)
        if str(z["tag"]) == tag:
            print(f"Loaded cached prototypes from {args.prototypes}")
            return [str(n) for n in z["names"]], z["protos"]
        print("Cached prototypes were built with different settings; rebuilding.")
    dirs = args.support_dir or SUPPORT_DIRS
    if not dirs:
        raise ValueError("Give --support-dir (folders with one sub-folder per sign), "
                         "or set SUPPORT_DIRS at the top of the file.")
    class_files = sample_support(dirs, args.shots, args.seed)
    print(f"Building prototypes for {len(class_files)} classes ...")
    names, protos = build_prototypes(embedder, class_files)
    if args.prototypes:
        np.savez(args.prototypes, names=np.array(names), protos=protos, tag=np.array(tag))
        print(f"Saved prototypes to {args.prototypes}")
    return names, protos


# ---------------------------------------------------------------------------
# Detector (same model the trainer saves; single-class glyph finder)
# ---------------------------------------------------------------------------

def detect(img, ckpt_path, threshold, nms_iou, min_box):
    import torch
    from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2,
                                              fasterrcnn_mobilenet_v3_large_fpn)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.ops import nms
    from torchvision.transforms import functional as TF

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device)
    meta = ck["meta"]
    maker = fasterrcnn_mobilenet_v3_large_fpn if meta["arch"] == "mobilenet" else fasterrcnn_resnet50_fpn_v2
    model = maker(weights=None, weights_backbone=None,
                  min_size=meta["min_size"], max_size=meta["max_size"])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, meta["num_classes"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    with torch.no_grad():
        out = model([TF.to_tensor(img).to(device)])[0]
    boxes, scores = out["boxes"].cpu(), out["scores"].cpu()
    keep = scores >= threshold
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) and nms_iou < 1.0:
        k = nms(boxes, scores, nms_iou)
        boxes, scores = boxes[k], scores[k]
    dets = []
    for b, s in zip(boxes.tolist(), scores.tolist()):
        if (b[2] - b[0]) >= min_box and (b[3] - b[1]) >= min_box:
            dets.append((b, float(s)))
    return dets


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def classify_boxes(img, dets, embedder, names, protos, pad, min_sim, min_margin=0.0):
    """dets: [(box, det_score)] -> list of result dicts, one per box.

    A box is labelled "unknown" if its best similarity is below min_sim, or if the best class
    beats the runner-up by less than min_margin (the classifier is not really choosing)."""
    if not dets:
        return []
    w, h = img.size
    crops = []
    for (x1, y1, x2, y2), _s in dets:
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        box = (max(0, int(x1 - px)), max(0, int(y1 - py)),
               min(w, int(x2 + px)), min(h, int(y2 + py)))
        crops.append(img.crop(box))
    embs = embedder(crops)
    sims = embs @ protos.T
    order = np.argsort(-sims, axis=1)
    results = []
    for i, ((box, det_score), row) in enumerate(zip(dets, order)):
        s1 = float(sims[i, row[0]])
        s2 = float(sims[i, row[1]]) if len(row) > 1 else float("nan")
        margin = s1 - s2 if len(row) > 1 else float("inf")
        label = names[row[0]] if (s1 >= min_sim and margin >= min_margin) else "unknown"
        results.append({
            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
            "det_score": det_score, "label": label, "best_label": names[row[0]], "sim": s1,
            "second_label": names[row[1]] if len(row) > 1 else "", "second_sim": s2,
        })
    return results


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:               # older Pillow
        return ImageFont.load_default()


def draw_results(img, results):
    out = img.copy()
    d = ImageDraw.Draw(out)
    font = _font(max(11, out.width // 70))
    for r in results:
        colour = (200, 120, 0) if r["label"] == "unknown" else (220, 0, 0)
        d.rectangle([r["x1"], r["y1"], r["x2"], r["y2"]], outline=colour, width=2)
        text = f'{r["label"]} {r["sim"]:.2f}'
        tb = d.textbbox((r["x1"], r["y1"]), text, font=font)
        ty = max(0, r["y1"] - (tb[3] - tb[1]) - 2)
        d.rectangle([r["x1"], ty, r["x1"] + (tb[2] - tb[0]) + 4, ty + (tb[3] - tb[1]) + 2], fill=colour)
        d.text((r["x1"] + 2, ty), text, fill=(255, 255, 255), font=font)
    return out


def save_results(img, results, out_path):
    draw_results(img, results).save(out_path)
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    cols = ["x1", "y1", "x2", "y2", "det_score", "label", "best_label", "sim", "second_label", "second_sim"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in results:
            wr.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})
    return csv_path


def cmd_classify(args):
    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)
    embedder = Embedder(args.backbone, args.dino_model, args.resnet_ckpt, args.gray_norm)
    names, protos = get_prototypes(embedder, args)

    img = load_rgb(args.image)
    dets = detect(img, args.detector_ckpt, args.threshold, args.nms_iou, args.min_box)
    print(f"{len(dets)} detections >= {args.threshold}")
    results = classify_boxes(img, dets, embedder, names, protos, args.pad, args.min_sim,
                             args.min_margin)

    out_path = args.out or os.path.splitext(args.image)[0] + "_classified.png"
    csv_path = save_results(img, results, out_path)
    print(f"Labelled image -> {out_path}\nTable          -> {csv_path}")
    if results:
        sims = np.array([r["sim"] for r in results])
        print(f"Similarity of the best class: min {sims.min():.2f}  median {np.median(sims):.2f}  "
              f"max {sims.max():.2f}  (use --min-sim to mark low ones 'unknown')")
        margins = np.array([r["sim"] - r["second_sim"] for r in results])
        print(f"Margin over the runner-up class: median {np.median(margins):.3f}; "
              f"{int((margins < 0.03).sum())} of {len(margins)} boxes are within 0.03 "
              f"(near-ties; use --min-margin to mark them 'unknown')")
        counts = {}
        for r in results:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        print("Boxes per label:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))


# ---------------------------------------------------------------------------
# eval: k-shot episodes
# ---------------------------------------------------------------------------

def write_confusion(conf, classes, base, k):
    """Row-normalised confusion matrix + per-sign accuracy, summed over all episodes."""
    tot = conf.sum(axis=1, keepdims=True)
    norm = np.divide(conf, tot, out=np.zeros_like(conf), where=tot > 0)
    with open(base + "_matrix.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["true\\pred"] + classes)
        for i, c in enumerate(classes):
            wr.writerow([c] + [f"{v:.3f}" for v in norm[i]])
    acc = np.diag(norm)
    with open(base + "_per_sign.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["sign", "accuracy", "most_confused_with", "confused_fraction"])
        for i, c in enumerate(classes):
            off = norm[i].copy()
            off[i] = 0
            j = int(np.argmax(off))
            wr.writerow([c, f"{acc[i]:.3f}", classes[j] if off[j] > 0 else "", f"{off[j]:.3f}"])
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and norm[i, j] > 0:
                pairs.append((norm[i, j], classes[i], classes[j]))
    pairs.sort(reverse=True)
    print(f"  top confusions at {k}-shot (true -> predicted, share of that sign's test crops):")
    for v, a, b in pairs[:8]:
        print(f"     {a} -> {b}: {v:.2f}")
    print(f"  wrote {base}_matrix.csv and {base}_per_sign.csv")


def run_eval(embedder, support_dirs, query_dirs, ks, episodes, pool, seed, out_csv=None, tag="",
             confusion_base=None):
    rng = random.Random(seed)

    def pool_of(dirs):
        merged = {}
        for d in dirs:
            for c, files in list_class_files(d).items():
                merged.setdefault(c, []).extend(files)
        out = {}
        for c, files in merged.items():
            files = sorted(set(files))
            rng.shuffle(files)
            out[c] = files[:pool] if pool else files
        return out

    sup, qry = pool_of(support_dirs), pool_of(query_dirs)
    classes = sorted(set(sup) & set(qry))
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes present in both --support-dir and --query-dir.")
    print(f"{len(classes)} classes in common; embedding up to {pool or 'all'} images per class "
          f"from each side ...")

    cache = {}
    for pool_dict in (sup, qry):
        for c in classes:
            need = [p for p in pool_dict[c] if p not in cache]
            if need:
                embs, ok = embed_files(embedder, need)
                for p, e in zip(ok, embs):
                    cache[p] = e
    classes = [c for c in classes
               if any(p in cache for p in sup[c]) and any(p in cache for p in qry[c])]
    chance = 1.0 / len(classes)

    rows = []
    print(f"\nChance level (1/{len(classes)}): {chance:.3f}")
    for k in ks:
        accs = []
        conf = np.zeros((len(classes), len(classes)), dtype=np.float64)   # rows true, cols predicted
        for ep in range(episodes):
            r = random.Random(seed * 100003 + k * 1009 + ep)
            protos, used = [], set()
            for c in classes:
                avail = [p for p in sup[c] if p in cache]
                chosen = r.sample(avail, min(k, len(avail)))
                used.update(chosen)
                protos.append(l2_normalise(np.mean([cache[p] for p in chosen], axis=0)))
            P = np.stack(protos)
            per_class = []
            for ci, c in enumerate(classes):
                qs = [p for p in qry[c] if p in cache and p not in used]
                if not qs:
                    continue
                E = np.stack([cache[p] for p in qs])
                pred = np.argmax(E @ P.T, axis=1)
                per_class.append(float(np.mean(pred == ci)))
                for pj in pred:
                    conf[ci, pj] += 1
            if per_class:
                accs.append(float(np.mean(per_class)))
        mean, std = float(np.mean(accs)), float(np.std(accs))
        print(f"  {k:>2}-shot: macro accuracy {mean:.3f} +/- {std:.3f}  over {len(accs)} episodes")
        if confusion_base:
            write_confusion(conf, classes, f"{confusion_base}_k{k}", k)
        rows.append({"backbone": tag, "k_shot": k, "mean_accuracy": f"{mean:.4f}",
                     "std": f"{std:.4f}", "episodes": len(accs), "n_classes": len(classes),
                     "chance": f"{chance:.4f}"})
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nResults saved to {out_csv}")
    return rows


def cmd_eval(args):
    embedder = Embedder(args.backbone, args.dino_model, args.resnet_ckpt, args.gray_norm)
    sup_dirs = args.support_dir or SUPPORT_DIRS
    if not sup_dirs or not args.query_dir:
        raise ValueError("eval needs --support-dir and --query-dir.")
    parts = []
    if args.backbone in ("dino", "both"):
        parts.append(args.dino_model)
    if args.backbone in ("resnet", "both"):
        parts.append("resnet_" + os.path.splitext(os.path.basename(str(args.resnet_ckpt)))[0])
    tag = f"{args.backbone}:" + "+".join(parts) + (":gray" if args.gray_norm else "")
    run_eval(embedder, sup_dirs, args.query_dir, args.shots_list, args.episodes,
             args.pool_per_class, args.seed, args.out_csv, tag=tag,
             confusion_base=args.confusion_base)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Few-shot prototypical classification of detected glyphs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--backbone", default="dino", choices=["dino", "resnet", "both"])
        p.add_argument("--dino-model", default=DINO_MODEL)
        p.add_argument("--resnet-ckpt", default=RESNET_CKPT)
        p.add_argument("--gray-norm", action="store_true",
                       help="embed a colour-free view (grayscale, stretched contrast, dark strokes "
                            "on light) so tile / paper colour does not dominate")
        p.add_argument("--support-dir", nargs="+", default=None,
                       help="folder(s) with one sub-folder of labelled glyph images per sign")
        p.add_argument("--seed", type=int, default=SEED)

    c = sub.add_parser("classify", help="detect, then name every box")
    common(c)
    c.add_argument("--image", required=True)
    c.add_argument("--detector-ckpt", default=DETECTOR_CKPT)
    c.add_argument("--threshold", type=float, default=0.3, help="detector confidence cut-off")
    c.add_argument("--nms-iou", type=float, default=0.4)
    c.add_argument("--min-box", type=int, default=10, help="ignore boxes smaller than this (px)")
    c.add_argument("--pad", type=float, default=0.04, help="extra margin around each crop (fraction)")
    c.add_argument("--shots", type=int, default=20,
                   help="images per sign taken from EACH support folder (0 = all)")
    c.add_argument("--min-sim", type=float, default=0.0,
                   help="best-class similarity below this is labelled 'unknown' (0 = never)")
    c.add_argument("--min-margin", type=float, default=0.0,
                   help="best class must beat the runner-up by this much, else 'unknown' (0 = never)")
    c.add_argument("--prototypes", default=None, help="cache file (.npz) for the prototypes")
    c.add_argument("--out", default=None)

    e = sub.add_parser("eval", help="k-shot accuracy on labelled crops")
    common(e)
    e.add_argument("--query-dir", nargs="+", default=None,
                   help="labelled crops to classify (must not overlap the support set)")
    e.add_argument("--shots", dest="shots_list", type=int, nargs="+", default=[1, 2, 5, 10],
                   help="k values to test, e.g. 1 2 5 10")
    e.add_argument("--episodes", type=int, default=20)
    e.add_argument("--pool-per-class", type=int, default=60,
                   help="cap on images embedded per class on each side (0 = all)")
    e.add_argument("--out-csv", default=None)
    e.add_argument("--confusion-base", default=None,
                   help="write confusion matrices + per-sign accuracy as <base>_k<K>_matrix.csv / "
                        "_per_sign.csv for every k, and print the top confusions")

    args = ap.parse_args()
    (cmd_classify if args.cmd == "classify" else cmd_eval)(args)


if __name__ == "__main__":
    main()
