"""
Composite glyph crops -> synthetic multi-glyph detection pages  (v2)
====================================================================

A Faster R-CNN detector needs multi-object SCENES with bounding boxes, but
your pipeline produces one image per glyph. This script pastes many glyphs onto
a page and writes the page images plus a COCO annotation file. The boxes are
free: we chose where each glyph goes.

What changed from v1 (all aimed at the synthetic-to-real gap you measured):

  * THREE glyph sources, mixed by weight, instead of only the CUT tiles:
        cut  - CUT-refined crops           (REFINED_CROPS_DIR)
        ink  - raw LPGS ink-on-white crops (INK_CROPS_DIR, optional)
        real - real photos / scans of glyphs: pen on paper, powder on stone,
               historical ink drawings        (REAL_CROPS_DIR, optional)
    Each source is a folder with one sub-folder per sign class; class folder
    names must match across sources.
  * WIDE size range. Each glyph gets a target size drawn log-uniformly from
    SIZE_RANGE_PX (default 50-500 px), so the detector sees tiny and huge signs
    instead of only ~100 px tiles.
  * REAL backgrounds (optional BACKGROUND_DIR of paper / cloth / stone photos),
    otherwise a tinted paper texture with random colour.
  * Ink-on-white crops are pasted with a stroke mask, so no white square shows,
    and every box is TIGHT around the visible glyph (after rotation), not the
    padded crop.
  * Photometric jitter (brightness, contrast, blur) per glyph.
  * A few EMPTY pages (no glyphs) as negatives.
  * Train/val split by GLYPH FILE: validation pages only use glyph files that no
    training page uses. Each image in the JSON carries "split": "train"/"val",
    which train_rcnn_local.py reads automatically.

Usage:
    Edit CONFIG below, then:
        python compose_detection_pages.py
"""

import os
import json
import math
import random

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------

# Glyph sources: folders containing one sub-folder per sign class.
REFINED_CROPS_DIR = r"<ROOT>\dataset\Synthetic Stone Carving_cut"  # CUT output
INK_CROPS_DIR = r"<ROOT>\dataset\synthetic"
REAL_CROPS_DIR = r"<ROOT>\dataset\real"

# Relative chance of drawing a glyph from each source. Sources you leave as
# None are skipped and the rest are renormalised automatically.
SOURCE_WEIGHTS = {"cut": 0.5, "ink": 0.3, "real": 0.2}

# Optional folder of background photos (paper, cloth, stone ...). Any images.
BACKGROUND_DIR = r"<ROOT>\dataset\reference\assume_stone_carving_2\Trader"
BACKGROUND_IMAGE_PROB = 0.6   # chance a page uses a photo background (if the folder is set)

OUTPUT_DIR = r"<ROOT>\nsibidi_training_data\detection_pages"

NUM_PAGES = 800                 # TRAIN pages
VAL_FRACTION = 0.1              # extra validation pages = NUM_PAGES * VAL_FRACTION
HOLDOUT_FRACTION = 0.1          # share of glyph files per class reserved for validation pages
EMPTY_PAGE_FRACTION = 0.03      # share of pages with no glyphs at all

GLYPHS_PER_PAGE_RANGE = (6, 22)
PAGE_SIZE = (1200, 900)

# Target size of a glyph's LONGEST side in pixels, drawn log-uniformly.
SIZE_RANGE_PX = (50, 500)
ROTATION_RANGE = (-15, 15)      # degrees
GAP_RANGE_PX = (2, 14)          # minimum gap between glyphs, random per page
MAX_PLACEMENT_ATTEMPTS = 60

SEED = 42

# ---------------------------------------------------------------------------

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
ALPHA_VISIBLE = 40   # alpha above this counts as "visible" when fitting boxes


def discover_classes(root):
    """{class_name: [image paths]} for a folder of class sub-folders."""
    out = {}
    for c in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        cdir = os.path.join(root, c)
        files = sorted(os.path.join(cdir, f) for f in os.listdir(cdir)
                       if f.lower().endswith(IMG_EXT))
        if files:
            out[c] = files
    return out


def split_files(files):
    """Hold out some files for validation pages. Classes with <5 files keep all for training."""
    files = sorted(files)
    random.shuffle(files)
    n_val = max(1, round(len(files) * HOLDOUT_FRACTION)) if len(files) >= 5 else 0
    return files[n_val:], files[:n_val]


def build_pools():
    """pools[split][class][source] = [paths]"""
    sources = {"cut": REFINED_CROPS_DIR, "ink": INK_CROPS_DIR, "real": REAL_CROPS_DIR}
    pools = {"train": {}, "val": {}}
    counts = {}
    for src, root in sources.items():
        if not root:
            continue
        if not os.path.isdir(root):
            print(f"WARNING: {src} folder not found, skipping: {root}")
            continue
        found = discover_classes(root)
        counts[src] = sum(len(v) for v in found.values())
        for c, files in found.items():
            tr, va = split_files(files)
            if tr:
                pools["train"].setdefault(c, {})[src] = tr
            if va:
                pools["val"].setdefault(c, {})[src] = va
    if not pools["train"]:
        raise ValueError("No glyph images found. Check REFINED_CROPS_DIR / INK_CROPS_DIR / REAL_CROPS_DIR.")
    for src, n in counts.items():
        print(f"  source '{src}': {n} glyph files")
    if not pools["val"]:
        print("WARNING: no files could be held out (classes too small) -- validation pages will "
              "reuse training glyphs, so validation mAP will be optimistic.")
        pools["val"] = pools["train"]
    return pools


def pick_glyph(pool):
    cname = random.choice(sorted(pool))
    srcs = pool[cname]
    names = sorted(srcs)
    weights = [max(SOURCE_WEIGHTS.get(s, 1.0), 1e-6) for s in names]
    src = random.choices(names, weights=weights)[0]
    return cname, src, random.choice(srcs[src])


def list_backgrounds():
    if not BACKGROUND_DIR:
        return []
    if not os.path.isdir(BACKGROUND_DIR):
        print(f"WARNING: BACKGROUND_DIR not found, using generated paper: {BACKGROUND_DIR}")
        return []
    files = [os.path.join(BACKGROUND_DIR, f) for f in sorted(os.listdir(BACKGROUND_DIR))
             if f.lower().endswith(IMG_EXT)]
    print(f"  backgrounds: {len(files)} images")
    return files


def make_background(size, bg_files):
    w, h = size
    if bg_files and random.random() < BACKGROUND_IMAGE_PROB:
        im = Image.open(random.choice(bg_files)).convert("RGB")
        iw, ih = im.size
        cw = int(iw * random.uniform(0.4, 1.0))
        ch = int(cw * h / w)
        if ch > ih:
            ch = ih
            cw = int(ch * w / h)
        cw, ch = max(cw, 8), max(ch, 8)
        x0 = random.randint(0, max(0, iw - cw))
        y0 = random.randint(0, max(0, ih - ch))
        im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize(size, Image.LANCZOS)
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return im
    # generated paper: random off-white / cream / grey tint plus grain
    level = random.uniform(225, 252)
    tint = np.array([random.uniform(-8, 8) for _ in range(3)])
    base = level + tint[None, None, :] + np.random.normal(0, random.uniform(2, 8), size=(h, w, 1))
    img = Image.fromarray(base.clip(180, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.6, 1.6)))


def jitter(im):
    """Brightness / contrast / blur on the colour channels, alpha untouched."""
    alpha = im.getchannel("A")
    rgb = im.convert("RGB")
    if random.random() < 0.7:
        rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.75, 1.15))
    if random.random() < 0.5:
        rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.75, 1.2))
    if random.random() < 0.25:
        rgb = rgb.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.4)))
    rgb.putalpha(alpha)
    return rgb


def prepare_glyph(path, target_side, rotation):
    """Load, mask, resize, rotate, jitter, then crop tight to the visible pixels."""
    im = Image.open(path).convert("RGBA")
    # Images with no real alpha (flat PNG/JPG): derive one from distance to white
    # so ink-on-white glyphs paste as strokes only. Tiles / photos that fill the
    # frame stay (almost) fully opaque. float32 avoids int16 overflow in the square.
    if im.getextrema()[3] == (255, 255):
        rgb = np.array(im.convert("RGB")).astype(np.float32)
        white_dist = np.sqrt(((rgb - 255) ** 2).sum(axis=2))
        alpha = np.clip((white_dist - 8) * 6, 0, 255).astype(np.uint8)
        im.putalpha(Image.fromarray(alpha))

    scale = target_side / max(im.size)
    new_w, new_h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    im = im.resize((new_w, new_h), Image.LANCZOS)
    im = im.rotate(rotation, expand=True, resample=Image.BICUBIC)   # corners become transparent
    im = jitter(im)

    visible = im.getchannel("A").point(lambda v: 255 if v > ALPHA_VISIBLE else 0)
    bbox = visible.getbbox()
    if bbox is None:            # nothing visible (very faint / tiny): skip this glyph
        return None
    im = im.crop(bbox)
    if im.width < 6 or im.height < 6:
        return None
    return im


def boxes_overlap(a, b, gap):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + gap < bx1 or bx2 + gap < ax1 or ay2 + gap < by1 or by2 + gap < ay1)


def compose_page(pool, bg_files, page_w, page_h, empty=False):
    page = make_background((page_w, page_h), bg_files)
    if empty:
        return page, []

    n_glyphs = random.randint(*GLYPHS_PER_PAGE_RANGE)
    log_lo, log_hi = math.log(SIZE_RANGE_PX[0]), math.log(SIZE_RANGE_PX[1])
    max_side = 0.8 * min(page_w, page_h)

    prepared = []
    for _ in range(n_glyphs):
        cname, _src, path = pick_glyph(pool)
        side = min(math.exp(random.uniform(log_lo, log_hi)), max_side)
        glyph = prepare_glyph(path, side, random.uniform(*ROTATION_RANGE))
        if glyph is not None:
            prepared.append((cname, glyph))
    prepared.sort(key=lambda t: -(t[1].width * t[1].height))   # place big ones first

    gap = random.randint(*GAP_RANGE_PX)
    placed, annotations = [], []
    for cname, glyph in prepared:
        gw, gh = glyph.size
        if gw >= page_w or gh >= page_h:
            continue
        for _attempt in range(MAX_PLACEMENT_ATTEMPTS):
            x1 = random.randint(0, page_w - gw)
            y1 = random.randint(0, page_h - gh)
            cand = (x1, y1, x1 + gw, y1 + gh)
            if not any(boxes_overlap(cand, b, gap) for b in placed):
                page.paste(glyph, (x1, y1), glyph)
                placed.append(cand)
                annotations.append((cname, x1, y1, x1 + gw, y1 + gh))
                break
        # no free spot found: this glyph is skipped for this page
    return page, annotations


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading glyph sources ...")
    pools = build_pools()
    bg_files = list_backgrounds()

    class_names = sorted(set(pools["train"]) | set(pools["val"]))
    cat_id = {c: i + 1 for i, c in enumerate(class_names)}
    print(f"{len(class_names)} classes: {class_names[:10]}{' ...' if len(class_names) > 10 else ''}")

    images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    coco = {"images": [], "annotations": [],
            "categories": [{"id": cat_id[c], "name": c} for c in class_names]}
    n_val_pages = max(1, int(NUM_PAGES * VAL_FRACTION))
    plan = [("train", i) for i in range(NUM_PAGES)] + [("val", i) for i in range(n_val_pages)]

    image_id, ann_id = 0, 1
    for split, i in plan:
        image_id += 1
        empty = random.random() < EMPTY_PAGE_FRACTION
        page, annotations = compose_page(pools[split], bg_files, *PAGE_SIZE, empty=empty)
        fname = f"{split}_{i:05d}.png"
        page.convert("RGB").save(os.path.join(images_dir, fname))
        coco["images"].append({"id": image_id, "file_name": fname, "split": split,
                               "width": PAGE_SIZE[0], "height": PAGE_SIZE[1]})
        for cname, x1, y1, x2, y2 in annotations:
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id": ann_id, "image_id": image_id, "category_id": cat_id[cname],
                "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1
        if image_id % 100 == 0:
            print(f"  composited {image_id}/{len(plan)} pages")

    ann_path = os.path.join(OUTPUT_DIR, "annotations_coco.json")
    with open(ann_path, "w") as f:
        json.dump(coco, f)

    n_tr = sum(1 for im in coco["images"] if im["split"] == "train")
    print(f"\nWrote {n_tr} train + {len(coco['images']) - n_tr} val pages to {images_dir}")
    print(f"Wrote {len(coco['annotations'])} box annotations to {ann_path}")


if __name__ == "__main__":
    main()
