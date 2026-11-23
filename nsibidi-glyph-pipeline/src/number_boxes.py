"""
Draw numbered boxes from a classify_detections CSV onto the ORIGINAL plate image and write a
truth-template CSV you fill in.  Use the original, un-annotated image.

  python number_boxes.py --image "Macgregor_001.png" --csv "page_e7clean_31st_Macgregor.csv" --out-prefix macgregor

Writes  macgregor_numbered.png  and  macgregor_truth.csv .
In the 'truth' column of the CSV write ONE of:
   <SignName>   one of the 20 vocabulary signs (Breast_milk, Chief, Crossbow, Eye, Female, Forest, Journey,
                Leopard, Love_affection, Lovers, Maiden, Male, MansQuarters, Penis, See, Staff, Trade,
                Trader, Unity, Vagina)
   oov          a real Nsibidi sign that is NOT one of the 20
   not_sign     a figure number, letter, stain, fragment or anything that is not a whole sign
"""
import argparse, csv
from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-prefix", default="plate")
    ap.add_argument("--scale", type=float, default=2.5, help="upscale so numbers are readable")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    im = Image.open(a.image).convert("RGB")
    s = a.scale
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, int(7 * s)))
    except Exception:
        font = ImageFont.load_default()
    for i, r in enumerate(rows, 1):
        x1, y1, x2, y2 = [float(r[k]) * s for k in ("x1", "y1", "x2", "y2")]
        d.rectangle([x1, y1, x2, y2], outline=(220, 0, 0), width=2)
        d.rectangle([x1, y1, x1 + 12 * len(str(i)) + 6, y1 + 18], fill=(220, 0, 0))
        d.text((x1 + 3, y1 + 1), str(i), fill=(255, 255, 255), font=font)
    im.save(f"{a.out_prefix}_numbered.png")

    with open(f"{a.out_prefix}_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "x1", "y1", "x2", "y2", "det_score", "pred", "sim", "margin", "truth"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["x1"], r["y1"], r["x2"], r["y2"], r["det_score"], r["best_label"],
                        r["sim"], round(float(r["sim"]) - float(r["second_sim"]), 4), ""])
    print(f"wrote {a.out_prefix}_numbered.png and {a.out_prefix}_truth.csv ({len(rows)} boxes)")


if __name__ == "__main__":
    main()
