"""
make_baseline_aug.py -- generate the GENERIC-AUGMENTATION baselines from the same one-prototype-per-sign
that LPGS starts from, so LPGS can be compared against them at an identical image budget.

Why: LPGS must be shown to beat plain augmentation of the same prototype.  The IEEE paper compared
against rotation/scale/translation only; LPGS also adds stroke irregularity, blur, noise and paper
texture, so that comparison mixes two effects.  Two baselines separate them:

  naive   rotation +-15 deg, scale 0.85-1.15, translation +-5 px          (as in the IEEE paper)
  strong  naive + independent x/y scale, shear, ELASTIC deformation, stroke-width change
          (erode/dilate), blur, Gaussian noise, brightness/contrast, paper-like background shading,
          JPEG artefacts                                                  (the fair generic baseline)

Input   --proto-dir  either  <proto-dir>/<Sign>/<image>   (one folder per sign, as the other scripts use)
                     or      <proto-dir>/<Sign>.png       (one flat file per sign)
Output  --out        <out>/<Sign>/<Sign>_<mode>_00000.png ...     (same layout as the 'synthetic' folders)

Example (PowerShell, one line):
  python make_baseline_aug.py --proto-dir "C:\\...\\prototypes" --out "C:\\...\\aug_naive"  --mode naive  --per-class 100 --seed 0
  python make_baseline_aug.py --proto-dir "C:\\...\\prototypes" --out "C:\\...\\aug_strong" --mode strong --per-class 100 --seed 0
Needs only numpy, Pillow and OpenCV (no torch).
"""
import os
import io
import argparse
import numpy as np
import cv2
from PIL import Image, ImageOps

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def load_gray_dark_on_light(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    g = ImageOps.autocontrast(im.convert("L"), cutoff=1)
    a = np.asarray(g)
    if np.median(a) < 128:                      # light strokes on dark ground -> invert
        a = 255 - a
    return a


def fit_to_canvas(a, size, margin=0.12):
    """Crop to the ink, scale to fit inside the canvas with a margin, centre on a white square."""
    ink = a < 160
    if ink.any():
        ys, xs = np.where(ink)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = size * (1 - 2 * margin) / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((size, size), 255, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = a
    return out


def affine(a, rng, rot, sx, sy, shear, tx, ty):
    size = a.shape[0]
    ang = np.deg2rad(rng.uniform(-rot, rot))
    scx, scy = rng.uniform(*sx), rng.uniform(*sy)
    sh = np.deg2rad(rng.uniform(-shear, shear)) if shear else 0.0
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    S = np.array([[scx, 0], [0, scy]])
    H = np.array([[1, np.tan(sh)], [0, 1]])
    M = R @ H @ S
    ctr = np.array([size / 2, size / 2])
    t = ctr - M @ ctr + np.array([rng.uniform(-tx, tx), rng.uniform(-ty, ty)])
    return cv2.warpAffine(a, np.hstack([M, t[:, None]]).astype(np.float32), (size, size),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def elastic(a, rng, alpha, sigma):
    size = a.shape[0]
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (size, size)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (size, size)).astype(np.float32), (0, 0), sigma) * alpha
    gx, gy = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    return cv2.remap(a, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def stroke_width(a, rng):
    k = rng.choice([-1, 0, 0, 1, 1, 2])          # >0 thicken dark strokes, <0 thin them
    if k == 0:
        return a
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(k) + 1, 2 * abs(k) + 1))
    return cv2.erode(a, ker) if k > 0 else cv2.dilate(a, ker)   # dark ink: erode the light image = thicken


def photometric(a, rng):
    f = a.astype(np.float32)
    if rng.random() < 0.7:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 1.4))
    contrast, bright = rng.uniform(0.8, 1.1), rng.uniform(-20, 20)
    f = (f - 255) * contrast + 255 + bright                       # keep white ground white-ish
    size = a.shape[0]                                              # paper-like shading
    gx, gy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    f = f - (rng.uniform(0, 25) * (gx * rng.uniform(-1, 1) + gy * rng.uniform(-1, 1) + 1)).astype(np.float32)
    f = f + rng.normal(0, rng.uniform(0, 8), f.shape)
    f = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:                                         # JPEG artefacts
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        f = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return f


def make_one(proto, mode, rng):
    size = proto.shape[0]
    if mode == "naive":
        s = rng.uniform(0.85, 1.15)
        return affine(proto, rng, rot=15, sx=(s, s), sy=(s, s), shear=0, tx=5 * size / 100, ty=5 * size / 100)
    a = affine(proto, rng, rot=15, sx=(0.85, 1.15), sy=(0.85, 1.15), shear=8, tx=5 * size / 100, ty=5 * size / 100)
    a = elastic(a, rng, alpha=rng.uniform(2, 7) * size / 100, sigma=rng.uniform(3, 6) * size / 100)
    a = stroke_width(a, rng)
    return photometric(a, rng)


def find_prototypes(root):
    out = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            fs = sorted(f for f in os.listdir(p) if f.lower().endswith(IMG_EXT))
            if fs:
                out[name] = [os.path.join(p, f) for f in fs]
        elif name.lower().endswith(IMG_EXT):
            out[os.path.splitext(name)[0]] = [p]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proto-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["naive", "strong"], required=True)
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--size", type=int, default=100, help="output size in px (the IEEE paper used 100)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    protos = find_prototypes(a.proto_dir)
    if not protos:
        raise SystemExit(f"No prototypes found in {a.proto_dir}")
    rng = np.random.default_rng(a.seed)
    for sign, files in protos.items():
        base = [fit_to_canvas(load_gray_dark_on_light(f), a.size) for f in files]
        od = os.path.join(a.out, sign)
        os.makedirs(od, exist_ok=True)
        for i in range(a.per_class):
            img = make_one(base[i % len(base)], a.mode, rng)
            Image.fromarray(img).save(os.path.join(od, f"{sign}_{a.mode}_{i:05d}.png"))
        print(f"{sign}: {a.per_class} images from {len(files)} prototype(s)")
    print(f"Done: {len(protos)} signs x {a.per_class} = {len(protos) * a.per_class} images in {a.out}")


if __name__ == "__main__":
    main()
