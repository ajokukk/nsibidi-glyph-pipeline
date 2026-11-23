"""
check_leakage.py -- are any TEST crops near-duplicates of TRAIN crops?   (no torch needed)

The crops were split at random.  If the same photograph, web image or writer sample appears on both
sides (crop-level duplicates, resized copies, mirrored-compression copies), accuracy is inflated.
This script compares every test crop with every train crop of the SAME sign and reports pairs whose
normalised cross-correlation (after autocontrast, polarity normalisation, 48x48 resize) is high, plus
SSIM and a 64-bit difference hash for the same pair. Each test crop is also compared in its 8 rotated/\nmirrored forms, so rotated or flipped augmented copies are caught.

    python check_leakage.py --train "C:\\...\\real_split\\train" --test "C:\\...\\real_split\\test" --out leakage_pairs.csv

Read the output:  pairs with NCC >= 0.95 are almost certainly the same image;  0.90-0.95 are worth a look.
Then open the listed files side by side.  Report the count in Chapter 4 either way.
"""
import os
import csv
import argparse
import numpy as np
from PIL import Image, ImageOps

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def view(path, n=48):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    g = ImageOps.autocontrast(im.convert("L"), cutoff=2)
    if np.median(np.asarray(g)) < 128:
        g = ImageOps.invert(g)
    return np.asarray(g.resize((n, n), Image.LANCZOS), dtype=np.float32)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def ssim(a, b):
    """Mean SSIM, 7x7 Gaussian window (sigma 1.5), on the 0-255 grey views."""
    import cv2
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (7, 7), 1.5)
    ma, mb = g(a), g(b)
    va, vb, cab = g(a * a) - ma * ma, g(b * b) - mb * mb, g(a * b) - ma * mb
    m = ((2 * ma * mb + C1) * (2 * cab + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))
    return float(m.mean())


def variants(a):
    """The 8 rotations/mirror images of a view (catches rotated or flipped augmented copies)."""
    out = []
    for f in (False, True):
        b = np.fliplr(a) if f else a
        for k in range(4):
            out.append(np.ascontiguousarray(np.rot90(b, k)))
    return out


def dhash(a):
    s = np.asarray(Image.fromarray(a.astype(np.uint8)).resize((9, 8), Image.LANCZOS), dtype=np.float32)
    return (s[:, 1:] > s[:, :-1]).flatten()


def classes(root):
    out = {}
    for c in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        fs = sorted(os.path.join(root, c, f) for f in os.listdir(os.path.join(root, c)) if f.lower().endswith(IMG_EXT))
        if fs:
            out[c] = fs
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="leakage_pairs.csv")
    ap.add_argument("--flag", type=float, default=0.90, help="report pairs with NCC >= this")
    a = ap.parse_args()
    tr, te = classes(a.train), classes(a.test)
    rows, n_test = [], 0
    for c, tfiles in te.items():
        trv = [(f, view(f)) for f in tr.get(c, [])]
        for tf in tfiles:
            n_test += 1
            tv = view(tf)
            vs = variants(tv)
            best = (-2.0, "", 0.0, 64, 0)          # ncc, file, ssim, hamming, transform index
            for f, v in trv:
                for i, tvv in enumerate(vs):
                    n = ncc(tvv, v)
                    if n > best[0]:
                        best = (n, f, ssim(tvv, v), int((dhash(tvv) != dhash(v)).sum()), i)
            rows.append([c, tf, best[1], f"{best[0]:.3f}", f"{best[2]:.3f}", best[3], "yes" if best[4] else "no"])
    rows.sort(key=lambda r: -float(r[3]))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["sign", "test_file", "closest_train_file", "ncc", "ssim", "dhash_hamming_of_64", "rotated_or_flipped"]); w.writerows(rows)
    vals = np.array([float(r[3]) for r in rows])
    print(f"{n_test} test crops compared with the train crops of the same sign")
    print(f"  NCC >= 0.95: {int((vals >= .95).sum())}   0.90-0.95: {int(((vals >= .90) & (vals < .95)).sum())}   "
          f"0.80-0.90: {int(((vals >= .80) & (vals < .90)).sum())}   median {np.median(vals):.2f}")
    print("Top 10 closest pairs (open these side by side):")
    for r in rows[:10]:
        print(f"  NCC {r[3]}  SSIM {r[4]}  hamming {r[5]:>2}  rot/flip:{r[6]:>3}  {os.path.basename(r[1])}  <->  {os.path.basename(r[2])}   [{r[0]}]")
    print(f"Full list in {a.out}")


if __name__ == "__main__":
    main()
