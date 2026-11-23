"""
retrain_new_signs.py  --  OPTIONAL retraining after new signs were added to the dictionary

What it does (the recipe of experiment E6, "retrained" arm):
  1. finds the NEW signs: sign folders in the verified folder (and in --prototype-dir, if given) whose
     names are not among the signs of the existing synthetic corpora;
  2. takes ONE image for each new sign as its prototype (a clean drawing in --prototype-dir/<Sign>/ if you
     supply one, otherwise the first verified crop) and makes --per-class LPGS variations of it with
     run_lpgs_arm.py;
  3. fine-tunes the Adinkra ResNet-18 again on the old synthetic corpora + the new LPGS images with
     finetune_resnet.py (all old and new signs together);
  4. saves a NEW embedding network. Your existing .pth files are never changed or overwritten.

Experiment E6 found NO measurable gain from retraining (differences of +1.6, 0 and +0.9 points at 1, 2 and 5
shots on 27 signs; Table 4.15). Adding the prototype and a few verified crops to the dictionary is the cheaper
route and was just as good. Use this only if you want to check whether retraining helps for your signs.

Typical use (Windows, from the project folder):
  python retrain_new_signs.py --module auto_glyph_variator1.py ^
      --synth-dirs "<ROOT>\\dataset\\arm_C_lpgs" ^
      --verified-dir verified --resnet-ckpt adinkra_resnet18.pth --out e6_retrained_new.pth

Time: about 12 minutes per 2,000 images for 3 epochs on a laptop CPU; close other programs first.
Add --dry-run to see which signs would be treated, without generating or training anything.
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
HERE = os.path.dirname(os.path.abspath(__file__))


def norm(name):
    return re.sub(r"[\s_\-]+", "", name).lower()


def sign_folders(d):
    """{sign name: sorted image paths} for the sub-folders of d that hold at least one image."""
    out = {}
    if not d or not os.path.isdir(d):
        return out
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        if os.path.isdir(p):
            fs = sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(IMG_EXT))
            if fs:
                out[os.path.basename(p)] = fs
    return out


def run(cmd):
    print("\n>>", " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"Step failed with exit code {r.returncode}: {cmd[1] if len(cmd) > 1 else cmd[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", required=True, help="the LPGS variator .py (the same one used for the synthetic folders)")
    ap.add_argument("--synth-dirs", nargs="+", required=True,
                    help="existing synthetic corpora (one sub-folder per sign), e.g. the arm C folder")
    ap.add_argument("--verified-dir", default="verified", help="folder of verified crops written by the application")
    ap.add_argument("--prototype-dir", default=None,
                    help="optional: <Sign>/<clean drawing> for new signs; used instead of a verified crop as the LPGS seed")
    ap.add_argument("--resnet-ckpt", required=True, help="Adinkra ResNet-18 .pth to start training from")
    ap.add_argument("--out", required=True, help="NEW embedding network to write (must not already exist)")
    ap.add_argument("--work-dir", default="retrain_work")
    ap.add_argument("--per-class", type=int, default=100, help="LPGS images generated for each new sign")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-per-class", type=int, default=100)
    ap.add_argument("--with-real", default=None,
                    help="optional folder of real training crops; they are merged with the verified crops and "
                         "added to training (the E7 recipe) instead of synthetic only")
    ap.add_argument("--real-repeat", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} already exists. Choose a new name: this script never overwrites a network.")
    if not os.path.isfile(a.module):
        raise SystemExit(f"LPGS variator not found: {a.module}")

    old = {}
    for d in a.synth_dirs:
        s = sign_folders(d)
        if not s:
            raise SystemExit(f"No sign sub-folders with images in --synth-dirs entry: {d}")
        old.update({k: v for k, v in s.items()})
    old_keys = {norm(k) for k in old}
    print(f"Existing synthetic corpora: {len(old)} signs (e.g. {', '.join(sorted(old)[:5])} ...)")

    verified = sign_folders(a.verified_dir)
    protos = sign_folders(a.prototype_dir) if a.prototype_dir else {}
    candidates = {}
    for src, label in ((verified, "verified crops"), (protos, "prototype folder")):
        for k, fs in src.items():
            if norm(k) not in old_keys:
                candidates.setdefault(k, label)
    if not candidates:
        raise SystemExit("No new signs found: every sign in the verified folder already has a synthetic corpus. "
                         "Nothing to retrain.")

    seeds = {}
    for k in sorted(candidates):
        if k in protos:
            seeds[k] = (protos[k][0], "prototype folder")
        else:
            seeds[k] = (verified[k][0], "first verified crop")
    print(f"\nNew signs ({len(seeds)}):")
    for k, (p, how) in seeds.items():
        print(f"  {k:<28} seed: {os.path.basename(p)}  ({how};"
              f" {len(verified.get(k, []))} verified crop(s))")
    if any(how == "first verified crop" for _, how in seeds.values()):
        print("\nNOTE: LPGS traces its seed image. A crop cut from a page can be noisy; a clean drawing in "
              "--prototype-dir/<Sign>/ gives better variations (E6 used one clean drawing per new sign).")
    if a.dry_run:
        print("\nDry run: nothing generated or trained.")
        return

    work = os.path.abspath(a.work_dir)
    proto_tmp = os.path.join(work, "proto")
    new_synth = os.path.join(work, "new_synth")
    for d in (proto_tmp, new_synth):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    for k, (p, _) in seeds.items():
        os.makedirs(os.path.join(proto_tmp, k), exist_ok=True)
        shutil.copy2(p, os.path.join(proto_tmp, k, os.path.basename(p)))

    py = sys.executable
    run([py, os.path.join(HERE, "run_lpgs_arm.py"), "--module", a.module, "--proto-dir", proto_tmp,
         "--out", new_synth, "--per-class", str(a.per_class), "--size", str(a.size), "--seed", str(a.seed)])

    tmp_out = os.path.join(work, "candidate.pth")
    cmd = [py, os.path.join(HERE, "finetune_resnet.py"), "--synth-dirs", *a.synth_dirs, new_synth,
           "--resnet-ckpt", a.resnet_ckpt, "--out", tmp_out, "--max-per-class", str(a.max_per_class),
           "--epochs", str(a.epochs), "--seed", str(a.seed)]
    if a.with_real:
        merged = os.path.join(work, "real_merged")
        shutil.rmtree(merged, ignore_errors=True)
        for src in (a.with_real, a.verified_dir):
            for k, fs in sign_folders(src).items():
                dst = os.path.join(merged, k)
                os.makedirs(dst, exist_ok=True)
                for f in fs:
                    shutil.copy2(f, os.path.join(dst, f"{os.path.basename(src.rstrip(os.sep)) or 'src'}_{os.path.basename(f)}"))
        cmd += ["--real-train-dir", merged, "--real-repeat", str(a.real_repeat)]
    run(cmd)

    if not os.path.isfile(tmp_out):
        raise SystemExit("Training finished but no network was written; see the messages above.")
    os.replace(tmp_out, a.out)
    meta_old = os.path.splitext(tmp_out)[0] + "_meta.json"
    log = {"time": dt.datetime.now().isoformat(timespec="seconds"), "out": a.out, "start_checkpoint": a.resnet_ckpt,
           "new_signs": sorted(seeds), "per_class": a.per_class, "epochs": a.epochs, "seed": a.seed,
           "synth_dirs": a.synth_dirs, "with_real": a.with_real, "variator": a.module}
    with open(os.path.splitext(a.out)[0] + "_retrain_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    if os.path.isfile(meta_old):
        os.replace(meta_old, os.path.splitext(a.out)[0] + "_meta.json")
    print(f"\nDone. New embedding network: {a.out}\n"
          "Your earlier networks were not changed. To use it, put its name in the application's "
          "'Embedding network' box. Similarity thresholds were set for earlier networks, so check the lanes again.")


if __name__ == "__main__":
    main()
