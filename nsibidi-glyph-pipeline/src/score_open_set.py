"""
Score every labelled real test crop the same way classify_detections.py scores a detected box
(best-class similarity, runner-up, margin), so we can compare 'known sign' confidence against
the confidence the same model gives to out-of-vocabulary boxes on a plate.

Put this file next to classify_detections.py.

  python score_open_set.py --support-dir "<ROOT>\\dataset\\real_split\\train" ^
      --query-dir "<ROOT>\\dataset\\real_split\\test" ^
      --resnet-ckpt e7_clean_s0.pth --out test_scores.csv
"""
import argparse, csv, os
import numpy as np
import classify_detections as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support-dir", nargs="+", required=True)
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--resnet-ckpt", required=True)
    ap.add_argument("--shots", type=int, default=20, help="same default as `classify`")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--out", default="test_scores.csv")
    a = ap.parse_args()

    emb = C.Embedder("resnet", C.DINO_MODEL, a.resnet_ckpt, gray_norm=True)
    class_files = C.sample_support(a.support_dir, a.shots, a.seed)
    names, protos = C.build_prototypes(emb, class_files)

    rows = []
    for sign, files in sorted(C.list_class_files(a.query_dir).items()):
        embs, ok = C.embed_files(emb, files)
        if len(ok) == 0:
            continue
        sims = embs @ protos.T
        for f, s in zip(ok, sims):
            o = np.argsort(-s)
            rows.append([os.path.basename(f), sign, names[o[0]], round(float(s[o[0]]), 4),
                         names[o[1]], round(float(s[o[1]]), 4), round(float(s[o[0]] - s[o[1]]), 4),
                         int(names[o[0]] == sign)])
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "pred", "sim", "second", "second_sim", "margin", "correct"])
        w.writerows(rows)
    sims = np.array([r[3] for r in rows]); acc = np.mean([r[7] for r in rows])
    print(f"{len(rows)} crops scored; top-1 accuracy {acc:.3f}")
    print(f"similarity: min {sims.min():.2f}  median {np.median(sims):.2f}  max {sims.max():.2f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
