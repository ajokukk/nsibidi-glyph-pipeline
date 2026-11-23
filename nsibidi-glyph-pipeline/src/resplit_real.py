import random, shutil, csv, math
from pathlib import Path

SRC = Path(r"<ROOT>\dataset\real_split_ids")
DST = Path(r"<ROOT>\dataset\real_split_ids_v2")
random.seed(0)

if DST.exists():
    raise SystemExit("real_split_ids_v2 already exists; delete or rename it first")

def files(d): return sorted(p for p in d.glob("*") if p.is_file())

log = []
for split in ("train", "test"):
    for sign in sorted((SRC / split).iterdir()):
        if sign.is_dir():
            (DST / split / sign.name).mkdir(parents=True, exist_ok=True)
            for f in files(sign):
                shutil.copy(f, DST / split / sign.name / f.name)

for sign in sorted((DST / "train").iterdir()):
    n_train = len(files(sign))
    n_test = len(files(DST / "test" / sign.name)) if (DST / "test" / sign.name).exists() else 0
    if n_test == 0 and n_train >= 2:
        k = max(1, math.floor(0.2 * n_train))
        moved = random.sample(files(sign), k)
        dst = DST / "test" / sign.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in moved:
            shutil.move(str(f), dst / f.name)
            log.append([sign.name, f.name, "moved train->test"])

with open(DST / "resplit_log.csv", "w", newline="", encoding="utf-8") as fh:
    csv.writer(fh).writerows([["sign", "file", "action"]] + log)

tr = [d for d in (DST / "train").iterdir() if d.is_dir() and files(d)]
te = [d for d in (DST / "test").iterdir() if d.is_dir() and files(d)]
both = {d.name for d in tr} & {d.name for d in te}
print("train signs:", len(tr), "test signs:", len(te), "evaluable (both):", len(both))