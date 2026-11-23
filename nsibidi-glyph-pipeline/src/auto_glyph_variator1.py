"""
Automatic Nsibidi Glyph Variation Generator (v2 — folder input + collated
degradation channels)
============================================================================

Replaces hand-written procedural glyph scripts (crossbow.py, ironbar.py,
lover.py, and the ~140 others like them, which each manually encode one
glyph's topology as coordinates) with a single automatic pipeline:

    1. Trace EVERY image found in INPUT_DIR to a vector outline (potrace) —
       no per-glyph script, no manual coordinate math.
    2. Normalize each traced glyph to a canonical size/position on a canvas.
    3. For each variation: flatten curves to a dense polyline, jitter every
       point (hand-tremor wobble), apply a random rotate/scale/translate,
       and re-render to a raster PNG.
    4. Run a consolidated environmental-degradation pass, collated from the
       reusable (shape-agnostic) post-processing channels found across the
       procedural scripts:
         - crossbow.py-style paper grain / ink bleed / fading water-patches
         - lover.py-style clean / blur / scanner-bleed-threshold / salt-&-
           pepper sensor noise channel split
         - ironbar.py-style independent vignette / shadow / highlight /
           deep-fade lighting layers
         - a new ink-weight jitter (thicken/thin strokes) standing in for
           pen-pressure variation, since it no longer comes from per-point
           stroke width like the old procedural scripts

You only ever drop reference PNGs into INPUT_DIR — no manual coordinate
math, no per-glyph script. Every other variation follows automatically.

Install:
    pip install potracer cairosvg pillow numpy

Usage:
    Edit INPUT_DIR / OUTPUT_DIR / VARIATIONS_PER_GLYPH below, then:
        python auto_glyph_variator.py
"""

import io
import os
import glob
import math
import random

import numpy as np
import potrace
import cairosvg
from PIL import Image, ImageFilter, ImageEnhance

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------

# Folder containing one clean reference image per glyph. Every image file
# found here (recursively is NOT used — top-level only) becomes its own
# glyph; the output folder name is taken from the filename.
INPUT_DIR = r"<HOME>\PycharmProjects\NsibidiDataEfficientPipeline\dataset"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

OUTPUT_DIR = r"<HOME>\PycharmProjects\NsibidiDataEfficientPipeline\output_outline"
VARIATIONS_PER_GLYPH = 50
CANVAS_SIZE = 224          # output PNGs are CANVAS_SIZE x CANVAS_SIZE

# Shape/geometry augmentation ranges.
SCALE_RANGE = (0.45, 0.90)          # overall size variation
ROTATION_RANGE_DEG = (-20, 20)      # tilt
TRANSLATE_JITTER_FRAC = 0.02        # canvas-fraction jitter of glyph center
POINT_JITTER_FRAC = 0.010           # canvas-fraction per-point hand-tremor amplitude
BEZIER_SAMPLES = 6                  # polyline density per curve segment (higher = smoother)
ORGANIC_BOW_FRAC = 0.03             # canvas-fraction max "belly" drift per contour (breastmilk.py/
                                     # Forest_bush.py-style up-and-down stroke waviness, see _add_organic_bow)

# Ink tone ranges (base_building_complex.py-style: natural ink darkness variance,
# with a paler, pre-faded tone reserved for the heavy_patches channel below).
INK_TONE_NORMAL_RANGE = (0, 35)     # dark ink, used for most channels
INK_TONE_FADED_RANGE = (50, 110)    # pale ink, used only under heavy water-damage patches

# Environmental degradation ranges.
FADING_RANGE = (0.1, 0.55)          # ink fade intensity for the light paper-grain channel
HEAVY_FADING_RANGE = (0.40, 0.75)   # ink fade intensity for the heavy water-damage channel

# Primary degradation channel weights — must align with the branches in
# apply_channel_degradation(). "heavy_patches" is deliberately rare (~10%),
# matching base_building_complex.py's explicit 5-10% target rate for its
# heaviest distress treatment.
CHANNEL_WEIGHTS = {
    "clean": 0.29,
    "blur": 0.20,
    "scanner_bleed": 0.13,
    "salt_pepper": 0.09,
    "light_patches": 0.19,
    "heavy_patches": 0.10,
}

# If True, only render a quick [original | variation x N] preview grid for
# the FIRST image found in INPUT_DIR instead of generating the full batch —
# use this to sanity-check the look before committing to hundreds of runs.
PREVIEW_MODE = False
PREVIEW_COUNT = 4

# ---------------------------------------------------------------------------
# 1. Trace a reference image to vector curves (potrace)
# ---------------------------------------------------------------------------

def _get_raw_curves(image_path, threshold=128, edge_tol=1.0):
    """
    Trace image_path with potrace and return a list of curve dicts, each:
      {"start": (x, y), "segments": [ {"type": "corner", "c":(x,y), "end":(x,y)}
                                       | {"type": "curve", "c1":(x,y), "c2":(x,y), "end":(x,y)} ]}
    Drops the whole-canvas border artifact potrace always emits.
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    bitmap = potrace.Bitmap(arr < threshold)  # True = ink pixel
    path = bitmap.trace()
    w, h = img.size

    curves = []
    for curve in path:
        pts = np.array([[p.x, p.y] for p in curve.decomposition_points])
        touches_all_edges = (
            pts[:, 0].min() <= edge_tol and pts[:, 1].min() <= edge_tol and
            pts[:, 0].max() >= w - edge_tol and pts[:, 1].max() >= h - edge_tol
        )
        if touches_all_edges:
            continue  # background border artifact, not real glyph content

        segments = []
        for seg in curve:
            if seg.is_corner:
                segments.append({"type": "corner", "c": (seg.c.x, seg.c.y),
                                  "end": (seg.end_point.x, seg.end_point.y)})
            else:
                segments.append({"type": "curve", "c1": (seg.c1.x, seg.c1.y),
                                  "c2": (seg.c2.x, seg.c2.y),
                                  "end": (seg.end_point.x, seg.end_point.y)})
        curves.append({"start": (curve.start_point.x, curve.start_point.y), "segments": segments})

    return curves, (w, h)


def _all_points(curves):
    pts = []
    for c in curves:
        pts.append(c["start"])
        for s in c["segments"]:
            if s["type"] == "corner":
                pts.append(s["c"]); pts.append(s["end"])
            else:
                pts.append(s["c1"]); pts.append(s["c2"]); pts.append(s["end"])
    return np.array(pts)


def extract_canonical_curves(image_path, canvas_size=100, fill_fraction=1.0, threshold=128):
    """
    Trace image_path and normalize the resulting curves so the glyph's
    longest bounding-box dimension equals fill_fraction * canvas_size,
    centered on a canvas_size x canvas_size canvas. This is the "scale=1.0"
    reference that SCALE_RANGE variations then shrink/grow from.
    """
    curves, size = _get_raw_curves(image_path, threshold=threshold)
    pts = _all_points(curves)
    if len(pts) == 0:
        raise ValueError(
            f"No glyph strokes found in {image_path} (only background artifact "
            f"detected). Check the threshold or the image contents."
        )

    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    bw, bh = maxx - minx, maxy - miny
    scale = fill_fraction * canvas_size / max(bw, bh, 1e-6)
    bcx, bcy = (minx + maxx) / 2, (miny + maxy) / 2
    ccx, ccy = canvas_size / 2, canvas_size / 2

    def norm_pt(p):
        x, y = p
        return ((x - bcx) * scale + ccx, (y - bcy) * scale + ccy)

    normalized = []
    for c in curves:
        segs = []
        for s in c["segments"]:
            if s["type"] == "corner":
                segs.append({"type": "corner", "c": norm_pt(s["c"]), "end": norm_pt(s["end"])})
            else:
                segs.append({"type": "curve", "c1": norm_pt(s["c1"]), "c2": norm_pt(s["c2"]),
                             "end": norm_pt(s["end"])})
        normalized.append({"start": norm_pt(c["start"]), "segments": segs})
    return normalized


# ---------------------------------------------------------------------------
# 2. Flatten curves to dense polylines (so wobble can act per-point) —
#    this is our equivalent of the procedural scripts' get_bezier_points /
#    compute_shaky_line_points hand-tremor wobble, applied generically to
#    any traced shape instead of hand-specified stroke endpoints.
# ---------------------------------------------------------------------------

def _cubic_bezier_points(p0, c1, c2, p1, n=6):
    pts = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * c1[0] + 3 * mt * t**2 * c2[0] + t**3 * p1[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * c1[1] + 3 * mt * t**2 * c2[1] + t**3 * p1[1]
        pts.append((x, y))
    return pts


def flatten_curve(curve, bezier_samples=6):
    pts = [curve["start"]]
    cur = curve["start"]
    for s in curve["segments"]:
        if s["type"] == "corner":
            pts.append(s["c"])
            pts.append(s["end"])
            cur = s["end"]
        else:
            pts.extend(_cubic_bezier_points(cur, s["c1"], s["c2"], s["end"], n=bezier_samples))
            cur = s["end"]
    return pts


# ---------------------------------------------------------------------------
# 3. Build a wobbled/rotated/scaled variation and render it
# ---------------------------------------------------------------------------

def _add_organic_bow(poly, max_bow_amp):
    """
    breastmilk.py / Forest_bush.py-style handwriting drift: those scripts add
    a coherent sin(t*pi) "belly" deviation along a stroke (see e.g. ironbar.py's
    get_bezier_points bow_x/bow_y), which reads as natural up-and-down waviness
    rather than uniform noise. This generalizes that to any traced contour:
    walk the polyline by arc length, and displace each point perpendicular to
    the local direction by a smooth sine wave that starts/ends near zero, with
    a random amplitude, phase, and 1-2 humps (sometimes a single gentle bow,
    sometimes an up-then-down sway) — different every time, and independent
    per contour, so different strokes in the same glyph drift differently.
    """
    n = len(poly)
    if n < 3 or max_bow_amp <= 0:
        return poly

    dists = [0.0]
    for i in range(1, n):
        dx = poly[i][0] - poly[i - 1][0]
        dy = poly[i][1] - poly[i - 1][1]
        dists.append(dists[-1] + math.hypot(dx, dy))
    total = dists[-1] if dists[-1] > 1e-6 else 1.0

    bow_amp = random.uniform(0, max_bow_amp)
    n_humps = random.choice([1, 1, 2])   # mostly a single gentle bow, sometimes up-then-down
    phase = random.uniform(0, 2 * math.pi)

    new_poly = []
    for i, (x, y) in enumerate(poly):
        t = dists[i] / total
        i0, i1 = max(0, i - 1), min(n - 1, i + 1)
        tx = poly[i1][0] - poly[i0][0]
        ty = poly[i1][1] - poly[i0][1]
        length = math.hypot(tx, ty) or 1.0
        px, py = -ty / length, tx / length  # unit perpendicular to local direction
        disp = bow_amp * math.sin(t * n_humps * math.pi + phase)
        new_poly.append((x + px * disp, y + py * disp))
    return new_poly


def build_variation_svg(curves, canvas_size, scale, angle_deg, translate_jitter,
                         point_jitter_amp, bezier_samples=6, bow_amp=0.0, ink_tone=0):
    cx = cy = canvas_size / 2
    angle_rad = math.radians(angle_deg)
    s_ang, c_ang = math.sin(angle_rad), math.cos(angle_rad)
    tx = random.uniform(-translate_jitter, translate_jitter)
    ty = random.uniform(-translate_jitter, translate_jitter)

    d_parts = []
    for curve in curves:
        poly = flatten_curve(curve, bezier_samples=bezier_samples)
        poly = _add_organic_bow(poly, bow_amp)
        transformed = []
        for (x, y) in poly:
            if point_jitter_amp > 0:
                x += random.uniform(-point_jitter_amp, point_jitter_amp)
                y += random.uniform(-point_jitter_amp, point_jitter_amp)
            x -= cx; y -= cy
            x *= scale; y *= scale
            xr = x * c_ang - y * s_ang
            yr = x * s_ang + y * c_ang
            transformed.append((xr + cx + tx, yr + cy + ty))

        d = f"M{transformed[0][0]},{transformed[0][1]}"
        for (x, y) in transformed[1:]:
            d += f" L{x},{y}"
        d += " Z"
        d_parts.append(d)

    path_d = " ".join(d_parts)
    fill = f"rgb({ink_tone},{ink_tone},{ink_tone})"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" '
            f'viewBox="0 0 {canvas_size} {canvas_size}">'
            f'<path d="{path_d}" fill="{fill}" fill-rule="evenodd"/></svg>')


def render_svg(svg_string, canvas_size):
    png_bytes = cairosvg.svg2png(bytestring=svg_string.encode(),
                                  output_width=canvas_size, output_height=canvas_size)
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    bg.paste(im, mask=im.split()[-1])
    return bg.convert("RGB")


# ---------------------------------------------------------------------------
# 4. Collated environmental degradation channels
#    (pulled from crossbow.py / ironbar.py / lover.py-style scripts)
# ---------------------------------------------------------------------------

def apply_paper_grain_and_patches(img, fading_intensity):
    """crossbow.py-style: dry-brush fade + ink-bleed blur + paper grain +
    2-4 organic water-stain/fading patches. Generalized off a hardcoded
    100x100 canvas to work at any size."""
    img_arr = np.array(img).astype(float)
    h, w = img_arr.shape[:2]

    if fading_intensity > 0.1:
        ink_mask = img_arr < 220
        img_arr[ink_mask] += (255.0 - img_arr[ink_mask]) * (fading_intensity * 0.4)

    if random.random() < 0.5:
        img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))
        img_arr = np.array(img).astype(float)

    grain = np.random.normal(0, random.uniform(3.0, 6.0), img_arr.shape)
    img_arr = np.clip(img_arr + grain, 0, 255)

    min_dim = min(h, w)
    for _ in range(random.randint(2, 4)):
        cx = random.randint(int(0.05 * w), int(0.95 * w))
        cy = random.randint(int(0.05 * h), int(0.95 * h))
        radius = random.randint(max(2, int(0.05 * min_dim)), max(3, int(0.18 * min_dim)))
        y_grid, x_grid = np.ogrid[:h, :w]
        mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius ** 2
        img_arr[mask] += random.uniform(20, 65)

    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))


def apply_heavy_patched_textures(img, fading_intensity):
    """base_building_complex.py-style: a rarer, more distressed variant of the
    light patches above — 4-7 larger water-stain patches (radius 10-28 vs. the
    light channel's 5-18), stronger fade-through, and heavier grain. Meant to
    be used sparingly (see CHANNEL_WEIGHTS["heavy_patches"]) on ink that was
    already rendered with a paler pre-faded tone (see INK_TONE_FADED_RANGE)."""
    img_arr = np.array(img).astype(float)
    h, w = img_arr.shape[:2]

    if fading_intensity > 0.1:
        ink_mask = img_arr < 235
        img_arr[ink_mask] += (255.0 - img_arr[ink_mask]) * (fading_intensity * 0.65)

    if random.random() < 0.7:
        img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 0.8)))
        img_arr = np.array(img).astype(float)

    grain = np.random.normal(0, random.uniform(5.0, 9.0), img_arr.shape)
    img_arr = np.clip(img_arr + grain, 0, 255)

    min_dim = min(h, w)
    for _ in range(random.randint(4, 7)):
        cx = random.randint(int(0.05 * w), int(0.95 * w))
        cy = random.randint(int(0.05 * h), int(0.95 * h))
        radius = random.randint(max(3, int(0.10 * min_dim)), max(4, int(0.28 * min_dim)))
        y_grid, x_grid = np.ogrid[:h, :w]
        mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius ** 2
        img_arr[mask] += random.uniform(40, 95)

    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))


def apply_subtle_clean_grain(img):
    """base_building_complex.py-style: even the 'clean' channel gets a hint of
    organic paper grain rather than a perfectly flat render."""
    img_arr = np.array(img).astype(float)
    grain = np.random.normal(0, 2.0, img_arr.shape)
    return Image.fromarray(np.clip(img_arr + grain, 0, 255).astype(np.uint8))


def apply_scanner_bleed(img):
    """lover.py-style: soft blur + heavy contrast + hard threshold, mimicking
    a fast flatbed scanner pass that bleeds ink and clips gray levels."""
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(2.5, 3.5))
    return img.point(lambda p: 255 if p > 140 else 0)


def apply_salt_pepper_noise(img):
    """lover.py-style: high-frequency sensor/scan noise."""
    img_np = np.array(img.convert("L"))
    noise_density = random.uniform(0.015, 0.03)
    num_salt = int(noise_density * img_np.size * 0.5)

    coords = [np.random.randint(0, i - 1, num_salt) for i in img_np.shape]
    img_np[tuple(coords)] = 255
    coords = [np.random.randint(0, i - 1, num_salt) for i in img_np.shape]
    img_np[tuple(coords)] = 0

    img = Image.fromarray(img_np).convert("RGB")
    return img.filter(ImageFilter.GaussianBlur(0.3))


def apply_vignette(img):
    """ironbar.py-style: darkened corners."""
    w, h = img.size
    strength = random.uniform(0.25, 0.38)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(x ** 2 + y ** 2)
    vignette_mask = np.clip(1.0 - (radius * strength), 0.0, 1.0)
    img_np = np.array(img).astype(float)
    for c in range(3):
        img_np[:, :, c] *= vignette_mask
    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))


def apply_shadow(img):
    """ironbar.py-style: directional darkening gradient."""
    w, h = img.size
    img_np = np.array(img).astype(float)
    angle = random.uniform(0, 2 * math.pi)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    grad = (x * math.cos(angle) + y * math.sin(angle) + 1.0) / 2.0
    factor = 0.82 + (grad * 0.18)
    for c in range(3):
        img_np[:, :, c] *= factor
    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))


def apply_highlight(img):
    """ironbar.py-style: directional lightening gradient."""
    w, h = img.size
    img_np = np.array(img).astype(float)
    angle = random.uniform(0, 2 * math.pi)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    grad = (x * math.cos(angle) + y * math.sin(angle) + 1.0) / 2.0
    factor = 1.0 + (grad * 0.14)
    for c in range(3):
        img_np[:, :, c] *= factor
    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))


def apply_deep_fading(img):
    """ironbar.py-style: directional ink-fade gradient (heavier than the
    uniform fade in apply_paper_grain_and_patches)."""
    w, h = img.size
    img_np = np.array(img).astype(float)
    angle = random.uniform(0, 2 * math.pi)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    grad = (x * math.cos(angle) + y * math.sin(angle) + 1.0) / 2.0
    fade_mask = grad * random.uniform(0.22, 0.36)
    for c in range(3):
        img_np[:, :, c] += (255.0 - img_np[:, :, c]) * fade_mask
    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))


def apply_ink_weight_jitter(img):
    """New: thicken or thin strokes slightly, standing in for the pen-
    pressure variation the old procedural scripts got for free from
    per-point variable stroke width."""
    roll = random.random()
    if roll < 0.20:
        return img.filter(ImageFilter.MinFilter(3))   # thicken (grow ink)
    elif roll < 0.40:
        return img.filter(ImageFilter.MaxFilter(3))   # thin (shrink ink)
    return img


def choose_channel():
    """Pick one primary degradation channel by weight (see CHANNEL_WEIGHTS)."""
    names = list(CHANNEL_WEIGHTS.keys())
    weights = list(CHANNEL_WEIGHTS.values())
    return random.choices(names, weights=weights, k=1)[0]


def ink_tone_for_channel(channel):
    """base_building_complex.py ties ink darkness to whether a sample will be
    heavily distressed: normal samples get dark ink, samples headed for heavy
    water-damage get a paler, pre-faded tone so the patches look like they're
    degrading already-weak ink rather than being pasted on top of solid black."""
    if channel == "heavy_patches":
        return random.randint(*INK_TONE_FADED_RANGE)
    return random.randint(*INK_TONE_NORMAL_RANGE)


def apply_channel_degradation(img, channel, fading_intensity):
    if channel == "clean":
        return apply_subtle_clean_grain(img)
    elif channel == "blur":
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
    elif channel == "scanner_bleed":
        return apply_scanner_bleed(img)
    elif channel == "salt_pepper":
        return apply_salt_pepper_noise(img)
    elif channel == "light_patches":
        return apply_paper_grain_and_patches(img, fading_intensity)
    elif channel == "heavy_patches":
        return apply_heavy_patched_textures(img, fading_intensity)
    return img


def apply_environmental_degradation(img, channel, fading_intensity):
    """
    Master orchestrator collating all channels above:
      - one primary channel, mutually exclusive, chosen by CHANNEL_WEIGHTS
        (mirrors lover.py's roll thresholds / base_building_complex.py's
        should_fade split — a real "clean" fraction matters for training
        realism, and heavy distress should stay rare)
      - independent low-probability lighting layers on top (mirrors
        ironbar.py's stacked 1-5% channels)
    """
    img = apply_ink_weight_jitter(img)
    img = apply_channel_degradation(img, channel, fading_intensity)

    if random.random() < 0.05:
        img = apply_vignette(img)
    if random.random() < 0.05:
        img = apply_shadow(img)
    if random.random() < 0.05:
        img = apply_highlight(img)
    if random.random() < 0.05:
        img = apply_deep_fading(img)

    return img


# ---------------------------------------------------------------------------
# 5. Driver
# ---------------------------------------------------------------------------

def discover_glyph_images(input_dir, exts=IMAGE_EXTS):
    """Find every image directly inside input_dir (non-recursive)."""
    paths = []
    for f in sorted(glob.glob(os.path.join(input_dir, "*"))):
        if os.path.splitext(f)[1].lower() in exts:
            paths.append(f)
    if not paths:
        raise FileNotFoundError(f"No images with extensions {exts} found in {input_dir}")
    return paths


def generate_variation(canonical_curves, canvas_size):
    # Decide the degradation channel FIRST, since it determines the ink tone
    # the glyph gets rendered with (see ink_tone_for_channel).
    channel = choose_channel()
    ink_tone = ink_tone_for_channel(channel)
    fading_intensity = random.uniform(*(HEAVY_FADING_RANGE if channel == "heavy_patches" else FADING_RANGE))

    scale = random.uniform(*SCALE_RANGE)
    angle = random.uniform(*ROTATION_RANGE_DEG)
    translate_jitter = canvas_size * TRANSLATE_JITTER_FRAC
    point_jitter_amp = canvas_size * POINT_JITTER_FRAC * random.uniform(0.5, 1.5)
    bow_amp = canvas_size * ORGANIC_BOW_FRAC * random.uniform(0.3, 1.2)

    svg = build_variation_svg(canonical_curves, canvas_size, scale, angle,
                               translate_jitter, point_jitter_amp,
                               bezier_samples=BEZIER_SAMPLES, bow_amp=bow_amp, ink_tone=ink_tone)
    img = render_svg(svg, canvas_size)
    img = apply_environmental_degradation(img, channel, fading_intensity)
    return img


def generate_variations_for_glyph(image_path, output_dir, n_variations, canvas_size):
    canonical = extract_canonical_curves(image_path, canvas_size=canvas_size, fill_fraction=1.0)
    name = os.path.splitext(os.path.basename(image_path))[0].replace(" ", "_")
    glyph_dir = os.path.join(output_dir, name, "png")
    os.makedirs(glyph_dir, exist_ok=True)

    for i in range(n_variations):
        img = generate_variation(canonical, canvas_size)
        img.save(os.path.join(glyph_dir, f"{name}_{i:04d}.png"))

    print(f"[{name}] wrote {n_variations} variations to {glyph_dir}")
    return glyph_dir


def preview_grid(image_path, output_dir, canvas_size, n_variations=4):
    """Save [original | var1 | var2 | ...] side by side so you can sanity-check
    the look before generating hundreds of images."""
    canonical = extract_canonical_curves(image_path, canvas_size=canvas_size, fill_fraction=1.0)
    original = Image.open(image_path).convert("RGB").resize((canvas_size, canvas_size))

    tiles = [original]
    for _ in range(n_variations):
        tiles.append(generate_variation(canonical, canvas_size))

    gap = 6
    sheet = Image.new("RGB", (len(tiles) * canvas_size + (len(tiles) - 1) * gap, canvas_size), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * (canvas_size + gap), 0))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "preview.png")
    sheet.save(out_path)
    print(f"Preview saved: {out_path}  (leftmost tile = original, rest = generated variations)")
    return out_path


if __name__ == "__main__":
    glyph_paths = discover_glyph_images(INPUT_DIR)
    print(f"Found {len(glyph_paths)} glyph image(s) in {INPUT_DIR}")

    if PREVIEW_MODE:
        preview_grid(glyph_paths[0], OUTPUT_DIR, CANVAS_SIZE, n_variations=PREVIEW_COUNT)
    else:
        for glyph_path in glyph_paths:
            generate_variations_for_glyph(glyph_path, OUTPUT_DIR, VARIATIONS_PER_GLYPH, CANVAS_SIZE)
        print("\nDone.")
