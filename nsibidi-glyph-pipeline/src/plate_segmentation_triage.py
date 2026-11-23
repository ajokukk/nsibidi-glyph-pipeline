"""
Nsibidi Plate Page: Ensemble Segmentation + Triage Pipeline
=============================================================

Segments a scanned plate page (a grid of small numbered glyphs, like the
"186-367" reference plate) into individual glyph crops, using two
independent classical-CV signals rather than one:

  1. GLYPH DETECTION (ink-based): connected components on the binarized
     page, with containment-merging so framed/boxed symbols (a border with
     disconnected marks inside it) are captured as ONE glyph rather than
     fragmenting into one box per internal mark.

  2. LABEL DETECTION (position + shape, no OCR/no deep learning): every
     digit caption on this kind of plate sits directly below its glyph and
     has a wider-than-tall aspect ratio (it's a short number string). This
     beats a generic text detector like CRAFT here because it encodes a
     fact about THIS page's layout that a pretrained model has no way to
     know -- see the conversation this script came out of for the direct
     empirical comparison (this approach found more correct labels with
     fewer false positives, and needs no GPU/model download).

  3. TRIAGE: every glyph candidate gets checked for two things a human
     should sanity-check: (a) is its size/aspect a statistical outlier
     versus the rest of the page (usually means several glyphs got merged
     into one blob), and (b) does a detected label overlap it in a way
     that suggests the "glyph" box is actually mostly a caption. Anything
     flagged goes to a review folder instead of being silently kept.

No fixed grid, no manual per-glyph tuning -- point it at a page and it
adapts to that page's own ink statistics.

Install:
    pip install pillow numpy opencv-python

Usage:
    Edit SOURCE_IMAGE / OUTPUT_DIR below, then:
        python plate_segmentation_triage.py
"""

import os
import csv
import numpy as np
import cv2
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------

SOURCE_IMAGE = r"<HOME>\Pictures\Screenshots\Screenshot 2026-09-05 072606.png"
OUTPUT_DIR = r"<ROOT>\nsibidi_training_data\new22"

# Binarization: pixels darker than this (0-255) count as ink.
INK_THRESHOLD = 200

# Glyph-blob detection: dilation kernel that bridges small gaps WITHIN one
# glyph's strokes without bridging gaps BETWEEN adjacent glyphs. If glyphs
# are getting merged together, lower this; if glyphs made of dashed/dotted
# lines are fragmenting into many small pieces, raise it.
GLYPH_DILATION_KERNEL = 3

# A component whose bbox covers more than this fraction of the page width
# or height is treated as a page border/frame artifact, not a glyph.
MAX_COMPONENT_PAGE_FRACTION = 0.3

# Components smaller than this many pixels are treated as scan noise.
MIN_COMPONENT_AREA = 4

# Containment merge: a smaller component whose area is at least this
# fraction covered by a larger component's bbox gets folded into it
# (handles framed symbols: border + disconnected marks inside).
CONTAINMENT_OVERLAP_THRESHOLD = 0.85

# Label detection: lighter dilation, since digit strokes are simple and
# don't need much bridging (over-bridging here would merge separate digits
# or merge a label into its glyph).
LABEL_DILATION_KERNEL = 2
# A label candidate must be shorter than the page's median component
# height by less than this multiple (i.e. genuinely small).
LABEL_MAX_HEIGHT_VS_MEDIAN = 1.3
# A label candidate's width/height ratio must fall in this range (a short
# digit string is noticeably wider than tall; a round dot or plain tick is
# not).
LABEL_ASPECT_RANGE = (1.3, 4.0)
# How close (pixels) a label candidate must sit below a glyph blob, and how
# closely it must be centered under that glyph, to count as its caption.
LABEL_MAX_VERTICAL_GAP = 14
LABEL_MAX_HORIZONTAL_OFFSET = 18

# Triage: how extreme a glyph's area/aspect must be (in robust z-score
# terms) versus the rest of the page before it's flagged for review.
OUTLIER_ZSCORE_THRESHOLD = 3.5
# If a glyph box overlaps a detected label by more than this IoU AND the
# glyph box itself is short, flag it (it's probably mostly a caption, not
# real glyph ink).
LABEL_CONFLICT_IOU_THRESHOLD = 0.4
LABEL_CONFLICT_MAX_HEIGHT_FRACTION = 0.06

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def load_binary_ink(image_path, threshold=INK_THRESHOLD):
    gray = np.array(Image.open(image_path).convert("L"))
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return gray, binary


def connected_component_boxes(binary, dilation_kernel, h_img, w_img,
                               max_page_frac=MAX_COMPONENT_PAGE_FRACTION,
                               min_area=MIN_COMPONENT_AREA):
    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if w > max_page_frac * w_img or h > max_page_frac * h_img:
            continue  # page border/frame artifact
        if a < min_area:
            continue  # scan noise speck
        boxes.append([x, y, x + w, y + h])
    return boxes


def containment_merge(boxes, thresh=CONTAINMENT_OVERLAP_THRESHOLD):
    """Fold a component into a larger one if most of its area sits inside
    that larger box (handles a frame/border containing separate marks)."""
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    used = [False] * len(boxes)
    merged = []
    for i, big in enumerate(boxes):
        if used[i]:
            continue
        cur = list(big)
        for j, small in enumerate(boxes):
            if used[j] or i == j:
                continue
            ix1, iy1 = max(cur[0], small[0]), max(cur[1], small[1])
            ix2, iy2 = min(cur[2], small[2]), min(cur[3], small[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_small = (small[2] - small[0]) * (small[3] - small[1])
            if area_small > 0 and inter / area_small > thresh:
                cur = [min(cur[0], small[0]), min(cur[1], small[1]),
                       max(cur[2], small[2]), max(cur[3], small[3])]
                used[j] = True
        used[i] = True
        merged.append(cur)
    return merged


def detect_labels(binary, h_img, w_img):
    """Classical geometric label detector: a caption is small, wider than
    tall, and sits just below a bigger glyph blob. See module docstring."""
    kernel = np.ones((LABEL_DILATION_KERNEL, LABEL_DILATION_KERNEL), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    small_boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if w > MAX_COMPONENT_PAGE_FRACTION * w_img or h > MAX_COMPONENT_PAGE_FRACTION * h_img:
            continue
        if a < MIN_COMPONENT_AREA - 1:
            continue
        small_boxes.append({"x": x, "y": y, "w": w, "h": h, "cx": x + w / 2, "cy": y + h / 2})

    if not small_boxes:
        return []

    median_h = np.median([b["h"] for b in small_boxes])
    big_blobs = [b for b in small_boxes if b["h"] >= median_h * LABEL_MAX_HEIGHT_VS_MEDIAN]

    def glyph_directly_above(cand, blobs):
        best, best_gap = None, 1e9
        for g in blobs:
            dx = abs(g["cx"] - cand["cx"])
            gap = cand["y"] - (g["y"] + g["h"])
            if dx < LABEL_MAX_HORIZONTAL_OFFSET and 0 <= gap < LABEL_MAX_VERTICAL_GAP and gap < best_gap:
                best, best_gap = g, gap
        return best

    labels = []
    lo, hi = LABEL_ASPECT_RANGE
    for b in small_boxes:
        if b["h"] >= median_h * LABEL_MAX_HEIGHT_VS_MEDIAN:
            continue
        aspect = b["w"] / max(b["h"], 1)
        if not (lo <= aspect <= hi):
            continue
        if glyph_directly_above(b, big_blobs) is not None:
            labels.append([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]])
    return labels


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def modified_zscore(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or 1e-6
    return 0.6745 * (x - med) / mad


def triage(glyph_boxes, label_boxes, h_img):
    areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in glyph_boxes])
    aspects = np.array([(b[2] - b[0]) / max(b[3] - b[1], 1e-6) for b in glyph_boxes])
    area_z = modified_zscore(areas)
    aspect_z = modified_zscore(aspects)

    results = []
    for idx, box in enumerate(glyph_boxes):
        reasons = []
        if abs(area_z[idx]) > OUTLIER_ZSCORE_THRESHOLD:
            reasons.append("area_outlier")
        if abs(aspect_z[idx]) > OUTLIER_ZSCORE_THRESHOLD:
            reasons.append("aspect_outlier")
        bh = box[3] - box[1]
        if bh < LABEL_CONFLICT_MAX_HEIGHT_FRACTION * h_img:
            if any(iou(box, lb) > LABEL_CONFLICT_IOU_THRESHOLD for lb in label_boxes):
                reasons.append("label_conflict")
        results.append({"box": box, "flagged": len(reasons) > 0, "reasons": reasons})
    return results


def run(source_image, output_dir):
    os.makedirs(os.path.join(output_dir, "auto_accepted"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "flagged_for_review"), exist_ok=True)

    gray, binary = load_binary_ink(source_image)
    h_img, w_img = gray.shape

    glyph_boxes = connected_component_boxes(binary, GLYPH_DILATION_KERNEL, h_img, w_img)
    glyph_boxes = containment_merge(glyph_boxes)
    print(f"Glyph candidates (after containment merge): {len(glyph_boxes)}")

    label_boxes = detect_labels(binary, h_img, w_img)
    print(f"Detected labels: {len(label_boxes)}")

    results = triage(glyph_boxes, label_boxes, h_img)
    n_flagged = sum(r["flagged"] for r in results)
    print(f"Auto-accepted: {len(results) - n_flagged}   Flagged for review: {n_flagged}")

    src_img = Image.open(source_image).convert("RGB")
    overlay = src_img.copy()
    draw = ImageDraw.Draw(overlay)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["box_id", "x1", "y1", "x2", "y2", "status", "reasons", "crop_path"])
        for i, r in enumerate(results):
            x1, y1, x2, y2 = [int(v) for v in r["box"]]
            crop = src_img.crop((x1, y1, x2, y2))
            status = "flagged" if r["flagged"] else "auto_accepted"
            reasons_str = "|".join(r["reasons"])
            subfolder = "flagged_for_review" if r["flagged"] else "auto_accepted"
            suffix = f"_{reasons_str}" if r["flagged"] else ""
            crop_path = os.path.join(output_dir, subfolder, f"box_{i:03d}{suffix}.png")
            crop.save(crop_path)
            draw.rectangle([x1, y1, x2, y2],
                            outline=(220, 30, 30) if r["flagged"] else (30, 160, 30), width=1)
            writer.writerow([i, x1, y1, x2, y2, status, reasons_str, crop_path])

    for lb in label_boxes:
        draw.rectangle([int(v) for v in lb], outline=(0, 100, 255), width=1)

    overlay_path = os.path.join(output_dir, "triage_overlay.png")
    overlay.save(overlay_path)
    print(f"\nSaved manifest: {manifest_path}")
    print(f"Saved review overlay: {overlay_path}")
    print(f"Crops saved under: {output_dir}/auto_accepted and {output_dir}/flagged_for_review")


if __name__ == "__main__":
    run(SOURCE_IMAGE, OUTPUT_DIR)
