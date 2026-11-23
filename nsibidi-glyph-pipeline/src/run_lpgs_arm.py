"""
Generate one augmentation-ablation arm from YOUR OWN LPGS variator script, without editing it.

It imports the variator file you point at (auto_glyph_variator1.py, ..._3_.py or ..._deformator2.py),
overrides its settings, and writes   <out>/<Sign>/<Sign>_<tag>_00000.png ...   (same layout as the
other arms, so finetune_resnet.py --synth-dirs <out> works unchanged).

  Arm C (LPGS, full):          add nothing
  Arm D (LPGS, degradation off): add --no-degrade
        -> keeps the geometry (retrace/skeleton deformation, bow, hand-tremor jitter, rotate/scale)
           but removes ink-weight jitter, grain, blur, scanner bleed, salt-and-pepper, patches, lighting.
           Ink is rendered dark, clean channel only.

Usage (Windows):
  python run_lpgs_arm.py --module "C:\\path\\auto_glyph_variator1.py" ^
      --proto-dir "<ROOT>\\dataset\\Prototypes for the dictionary" ^
      --out "<ROOT>\\dataset\\arm_C_lpgs" --per-class 100 --size 100 --seed 0
  (add --no-degrade and a different --out for arm D)
"""
import argparse, glob, importlib.util, os, random
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def load_module(path):
    spec = importlib.util.spec_from_file_location("variator", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def find_protos(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        if os.path.isdir(p):
            fs = sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(IMG_EXT))
            if fs:
                out[os.path.basename(p)] = fs[0]
        elif p.lower().endswith(IMG_EXT):
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="path to the variator .py you used for the synthetic folder")
    ap.add_argument("--proto-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-degrade", action="store_true")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    m = load_module(a.module)
    random.seed(a.seed)
    np.random.seed(a.seed)
    tag = a.tag or ("lpgsnodeg" if a.no_degrade else "lpgs")

    if a.no_degrade:
        m.choose_channel = lambda: "clean"                       # dark ink tone, no faded-ink branch
        m.apply_environmental_degradation = lambda img, *args, **kw: img

    if hasattr(m, "extract_master_trace_info"):                  # retrace version
        prep = lambda p: m.extract_master_trace_info(p, canvas_size=a.size, fill_fraction=1.0)
    elif hasattr(m, "extract_canonical_skeleton"):               # skeleton-deformation version
        prep = lambda p: m.extract_canonical_skeleton(p, canvas_size=a.size, fill_fraction=1.0)
    else:                                                        # original single-trace version
        prep = lambda p: m.extract_canonical_curves(p, canvas_size=a.size, fill_fraction=1.0)

    protos = find_protos(a.proto_dir)
    if not protos:
        raise SystemExit(f"No prototype images found in {a.proto_dir}")
    for sign, path in protos.items():
        info = prep(path)
        od = os.path.join(a.out, sign)
        os.makedirs(od, exist_ok=True)
        for i in range(a.per_class):
            m.generate_variation(info, a.size).save(os.path.join(od, f"{sign}_{tag}_{i:05d}.png"))
        print(f"{sign}: {a.per_class} images")
    print(f"Done: {len(protos)} signs x {a.per_class} in {a.out}  (degradation {'OFF' if a.no_degrade else 'ON'})")


if __name__ == "__main__":
    main()
