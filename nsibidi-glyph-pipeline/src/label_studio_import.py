"""
Manifest -> Label Studio import tasks
========================================

Converts the manifest.csv (+ source page image) produced by
plate_segmentation_triage.py into a Label Studio import file: one task per
PAGE, with every candidate glyph box pre-drawn as a prediction so the
reviewer edits/moves/deletes/adds rectangles directly on the full page
instead of working crop-by-crop. This keeps full grid context available,
which matters for fixing merged multi-glyph clusters correctly.

Boxes are pre-labeled:
    "Glyph"        - auto-accepted, probably fine, but still editable
    "NeedsReview"  - flagged by triage, reviewer should look closely

Supports multiple pages in one project: just add more (image_path,
manifest_csv_path) pairs to PAGES below.

Setup in Label Studio:
    1. Create a project, skip the template, and paste LABEL_CONFIG (printed
       below, also saved to label_config.xml) into
       Settings -> Labeling Interface -> Code.
    2. Enable local file serving so Label Studio can read your images:
       set environment variables before starting Label Studio:
           LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
           LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<folder containing your page images>
       (see Label Studio docs: "Local storage" / local file serving)
    3. In the project, Import -> upload the generated tasks.json.

Install:
    pip install label-studio   (only needed to RUN Label Studio itself)
"""

import os
import csv
import json
from PIL import Image

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------

# One entry per page you've already run plate_segmentation_triage.py on.
PAGES = [
    {
        "image_path": r"<HOME>\Pictures\Screenshots\Screenshot 2026-09-05 072606.png",
        "manifest_csv": r"<ROOT>\nsibidi_training_data\new22\manifest.csv",
    },
    # {"image_path": r"...plate_page_2.png", "manifest_csv": r"...\manifest.csv"},
]

OUTPUT_TASKS_JSON = r"<ROOT>\nsibidi_training_data\new22\label_studio_tasks.json"

# How Label Studio should locate each image. With local file serving enabled
# (see docstring), this is "/data/local-files/?d=<filename-or-subpath>"
# relative to LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT. Change this prefix if
# your setup differs (e.g. serving via a plain HTTP server instead).
IMAGE_URL_PREFIX = "<ROOT>/nsibidi_training_data/new22"

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Glyph" background="#2ecc71"/>
    <Label value="NeedsReview" background="#e74c3c"/>
  </RectangleLabels>
</View>"""

# ---------------------------------------------------------------------------

def build_task(image_path, manifest_csv, image_url_prefix):
    with Image.open(image_path) as im:
        w_img, h_img = im.size

    results = []
    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            label = "NeedsReview" if row["status"] == "flagged" else "Glyph"
            # Label Studio wants percentages of image width/height, not pixels.
            x_pct = 100.0 * x1 / w_img
            y_pct = 100.0 * y1 / h_img
            w_pct = 100.0 * (x2 - x1) / w_img
            h_pct = 100.0 * (y2 - y1) / h_img
            results.append({
                "id": f"box_{row['box_id']}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": w_img,
                "original_height": h_img,
                "image_rotation": 0,
                "value": {
                    "x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct,
                    "rotation": 0,
                    "rectanglelabels": [label],
                },
            })

    filename = os.path.basename(image_path)
    task = {
        "data": {"image": image_url_prefix + filename},
        "predictions": [{"model_version": "triage_v1", "result": results}],
    }
    return task


def main():
    tasks = [build_task(p["image_path"], p["manifest_csv"], IMAGE_URL_PREFIX) for p in PAGES]

    os.makedirs(os.path.dirname(OUTPUT_TASKS_JSON) or ".", exist_ok=True)
    with open(OUTPUT_TASKS_JSON, "w") as f:
        json.dump(tasks, f, indent=2)

    config_path = os.path.join(os.path.dirname(OUTPUT_TASKS_JSON) or ".", "label_config.xml")
    with open(config_path, "w") as f:
        f.write(LABEL_CONFIG)

    total_boxes = sum(len(t["predictions"][0]["result"]) for t in tasks)
    print(f"Wrote {len(tasks)} task(s), {total_boxes} pre-annotated boxes total.")
    print(f"Tasks:  {OUTPUT_TASKS_JSON}")
    print(f"Label config (paste into Label Studio's Labeling Interface -> Code): {config_path}")


if __name__ == "__main__":
    main()
