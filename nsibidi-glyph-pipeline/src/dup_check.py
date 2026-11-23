import itertools, csv
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(r"<ROOT>\dataset\real_split_ids_v2")
items = []
for split in ("train", "test"):
    for p in (ROOT / split).glob("*/*"):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            im = Image.open(p).convert("L").resize((16, 16))
            a = np.asarray(im, float)
            items.append((p, a > a.mean()))

rows = []
for (p1, h1), (p2, h2) in itertools.combinations(items, 2):
    if p1.parent.name == p2.parent.name:
        continue
    d = int((h1 != h2).sum())
    if d <= 12:
        rows.append([p1.parent.name, p1.name, p2.parent.name, p2.name, d])

rows.sort(key=lambda r: r[4])
with open("dup_check.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([["sign_a", "file_a", "sign_b", "file_b", "hamming"]] + rows)
print(len(rows), "near-duplicate pairs across different signs -> dup_check.csv")