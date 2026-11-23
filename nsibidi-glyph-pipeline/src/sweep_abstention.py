"""
sweep_abstention.py -- choose --min-sim / --min-margin with evidence (no model needed).

The classify CSV already stores, for every detected box, the top guess (best_label), its
similarity (sim) and the runner-up similarity (second_sim).  So any threshold pair can be
replayed offline:   named  <=>  sim >= min_sim  AND  (sim - second_sim) >= min_margin.

Step 1  make a truth file from the classify CSV (fill in the true sign for every box):
    python sweep_abstention.py template --csv page_00779_classified.csv --out truth_page_00779.csv

Step 2  edit truth_page_00779.csv, column true_label:
    - the sign's name exactly as in the support folders (e.g. Maiden, Unity)
    - other   if the sign is NOT one of the 20 classes (the system should abstain on it)
    - delete the row (or write skip) for boxes that are not a glyph at all (false detections)

Step 3  sweep:
    python sweep_abstention.py sweep --csv page_00779_classified.csv --truth truth_page_00779.csv ^
        --out-csv sweep_page_00779.csv
"""
import argparse
import csv
import sys
import numpy as np


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def cmd_template(a):
    rows = read_csv(a.csv)
    need = {"best_label", "sim", "second_sim"}
    if not rows or not need <= set(rows[0]):
        sys.exit(f"{a.csv} must have columns {sorted(need)} (re-run classify with the latest script).")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "x1", "y1", "x2", "y2", "model_label", "model_best_guess", "sim", "second_sim",
                    "true_label"])
        for i, r in enumerate(rows, 1):
            named = r["label"] != "unknown"
            # Named boxes are pre-filled with the model's label: CHECK them. Unknown boxes are left blank.
            w.writerow([i, r["x1"], r["y1"], r["x2"], r["y2"], r["label"], r["best_label"], r["sim"],
                        r["second_sim"], r["label"] if named else ""])
    print(f"wrote {a.out}  ({len(rows)} rows). Fill/correct the true_label column, then run 'sweep'.")


def cmd_sweep(a):
    rows = read_csv(a.csv)
    truth = {int(r["row"]): r["true_label"].strip() for r in read_csv(a.truth)}
    miss = [i for i, t in truth.items() if t == ""]
    if miss:
        sys.exit(f"true_label is empty for rows {miss}. Write the sign name, 'other', or 'skip'.")
    idx, sim, sec, guess, true = [], [], [], [], []
    for i, r in enumerate(rows, 1):
        t = truth.get(i)
        if t is None or t.lower() == "skip":
            continue
        idx.append(i); sim.append(float(r["sim"])); sec.append(float(r["second_sim"]))
        guess.append(r["best_label"]); true.append("other" if t.lower() == "other" else t)
    sim, sec = np.array(sim), np.array(sec)
    margin = sim - sec
    in_vocab = np.array([t != "other" for t in true])
    right = np.array([g == t for g, t in zip(guess, true)])          # top guess is correct
    n, nv = len(sim), int(in_vocab.sum())
    print(f"{n} boxes scored: {nv} in the 20 classes, {n - nv} outside them (should abstain).")
    print(f"top guess correct for {int(right.sum())}/{nv} in-class boxes (no abstention at all).\n")

    sims = np.arange(a.sim_lo, a.sim_hi + 1e-9, a.sim_step)
    margs = np.arange(0.0, a.margin_hi + 1e-9, a.margin_step)
    out = []
    for s in sims:
        for m in margs:
            named = (sim >= s) & (margin >= m)
            tp = int((named & right).sum())                              # named and correct
            fp = int((named & ~right).sum())                             # named and wrong (incl. outside classes)
            lost = int((~named & right).sum())                           # correct guess withheld
            saved = int((~named & ~right).sum())                         # wrong guess withheld
            out.append(dict(min_sim=round(float(s), 3), min_margin=round(float(m), 3), named=int(named.sum()),
                            coverage=named.mean(), precision=(tp / (tp + fp) if tp + fp else float("nan")),
                            tp=tp, fp=fp, lost_correct=lost, wrong_withheld=saved,
                            recall_in_class=tp / nv if nv else float("nan")))
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)

    # Operating points: fewest allowed wrong names, most correct names.
    print(f"{'allowed wrong':>13} {'best TP':>8} {'coverage':>9}  loosest / strictest / suggested setting with that result")
    chosen = None
    for budget in (0, 1, 2):
        ok = [o for o in out if o["fp"] <= budget]
        best = max(o["tp"] for o in ok)
        opt = [o for o in ok if o["tp"] == best]
        lo = min(opt, key=lambda o: (o["min_sim"] + o["min_margin"]))
        hi = max(opt, key=lambda o: (o["min_sim"] + o["min_margin"]))
        # Prefer the least restrictive similarity floor, then the middle margin among the tied settings.
        s0 = min(o["min_sim"] for o in opt)
        row_ = sorted([o for o in opt if o["min_sim"] == s0], key=lambda o: o["min_margin"])
        mid = row_[len(row_) // 2]
        print(f"{budget:>13} {best:>8} {mid['coverage']:>9.2f}  "
              f"sim>={lo['min_sim']:.2f},margin>={lo['min_margin']:.2f} / "
              f"sim>={hi['min_sim']:.2f},margin>={hi['min_margin']:.2f} / "
              f"sim>={mid['min_sim']:.2f},margin>={mid['min_margin']:.2f}   ({len(opt)} grid points tie)")
        if budget == 0:
            chosen = mid
    print("\nRisk-coverage with min_margin only (min_sim = 0):")
    print(f"{'min_margin':>10} {'named':>6} {'coverage':>9} {'precision':>10} {'TP':>4} {'FP':>4}")
    for o in out:
        if o["min_sim"] == round(float(sims[0]), 3) and round(o["min_margin"] / 0.04, 6) % 1 == 0:
            print(f"{o['min_margin']:>10.2f} {o['named']:>6} {o['coverage']:>9.2f} "
                  f"{o['precision'] if o['precision'] == o['precision'] else float('nan'):>10.2f} {o['tp']:>4} {o['fp']:>4}")
    print("\nRisk-coverage with min_sim only (min_margin = 0):")
    print(f"{'min_sim':>10} {'named':>6} {'coverage':>9} {'precision':>10} {'TP':>4} {'FP':>4}")
    for o in out:
        if o["min_margin"] == 0 and round(o["min_sim"] / 0.05, 6) % 1 == 0:
            print(f"{o['min_sim']:>10.2f} {o['named']:>6} {o['coverage']:>9.2f} {o['precision']:>10.2f} {o['tp']:>4} {o['fp']:>4}")
    print("\nWrong top guesses (these are what the thresholds must catch):")
    for i, g, t, s_, m_ in zip(idx, guess, true, sim, margin):
        if g != t:
            print(f"  row {i}: guessed {g} (sim {s_:.3f}, margin {m_:.3f}), truth {t}")
    print("\nSuggested starting point (middle of the 0-error region):"
          f" --min-sim {chosen['min_sim']:.2f} --min-margin {chosen['min_margin']:.2f}")
    print("Caution: the thresholds are tuned and scored on the same page. Treat the result as a calibration,"
          " and check it on a different plate (e.g. Dayrell) before you report it as generalising.")
    print(f"Full grid written to {a.out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template"); t.add_argument("--csv", required=True); t.add_argument("--out", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--csv", required=True); s.add_argument("--truth", required=True)
    s.add_argument("--out-csv", default="sweep.csv")
    s.add_argument("--sim-lo", type=float, default=0.0); s.add_argument("--sim-hi", type=float, default=0.95)
    s.add_argument("--sim-step", type=float, default=0.01)
    s.add_argument("--margin-hi", type=float, default=0.30); s.add_argument("--margin-step", type=float, default=0.01)
    a = ap.parse_args()
    {"template": cmd_template, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    main()
