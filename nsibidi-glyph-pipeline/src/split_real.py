"""Split real glyph crops into disjoint train / test sets, per sign.

Usage (one line):
  python split_real.py --src "C:\\...\\real_glyphs" --out "C:\\...\\real_split" --test-frac 0.3

Input : src/<SignName>/*.png|jpg|...   (one folder per sign)
Output: out/train/<SignName>/...  and  out/test/<SignName>/...
Files are COPIED (originals untouched). Same seed -> same split every time.
Every sign keeps at least 1 train image; signs with only 1 image go to train only.
"""
import argparse, os, random, shutil

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if os.path.exists(os.path.join(a.out, "train")) or os.path.exists(os.path.join(a.out, "test")):
        raise SystemExit(f"{a.out} already has train/test folders. Delete it or choose another --out.")

    rng = random.Random(a.seed)
    tot_tr = tot_te = 0
    print(f"{'sign':28s} {'total':>5s} {'train':>5s} {'test':>5s}")
    for c in sorted(os.listdir(a.src)):
        p = os.path.join(a.src, c)
        if not os.path.isdir(p):
            continue
        files = sorted(f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in EXT)
        rng.shuffle(files)
        n = len(files)
        n_test = 0 if n < 2 else min(n - 1, max(1, round(n * a.test_frac)))
        for i, f in enumerate(files):
            split = "test" if i < n_test else "train"
            d = os.path.join(a.out, split, c)
            os.makedirs(d, exist_ok=True)
            shutil.copy2(os.path.join(p, f), os.path.join(d, f))
        tot_tr += n - n_test
        tot_te += n_test
        flag = "  <-- only 1 image, train only" if n < 2 else ""
        print(f"{c:28s} {n:5d} {n-n_test:5d} {n_test:5d}{flag}")
    print(f"\nTOTAL train={tot_tr} test={tot_te}\nWritten to {a.out}")

if __name__ == "__main__":
    main()
