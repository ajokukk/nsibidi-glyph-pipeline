"""
Label Studio export -> merged final annotations (the "seed set")
====================================================================

Takes the JSON you export from Label Studio after reviewing/correcting the
pre-annotated boxes (Project -> Export -> JSON) and converts every box back
to pixel coordinates, producing one clean CSV per page: this is the
corrected ground truth ready to train the actual detector.

Handles the review workflow naturally: whatever the reviewer left in
Label Studio IS the final answer for that page --
    - edited/resized "Glyph" or "NeedsReview" boxes -> kept as final boxes
    - deleted boxes -> gone (correctly excluded)
    - newly-drawn boxes (e.g. splitting a merged cluster into two glyphs)
      -> included automatically, no special-casing needed
The "Glyph" vs "NeedsReview" label itself doesn't need to be accurate any
more by this point -- once a human has looked at it, every remaining box is
just "a glyph", so the output collapses both into one class unless you
choose to keep the distinction (see KEEP_LABEL_CLASS below).

Usage:
    Edit EXPORT_JSON / OUTPUT_DIR below, then:
        python label_studio_export_merge.py
"""

import os
import json
import csv

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------

EXPORT_JSON = r"<ROOT>\nsibidi_training_data\new22\label_studio_tasks.json"
OUTPUT_DIR = r"<ROOT>\nsibidi_training_data\new22\final_seed_set"

# If True, the output CSV keeps whatever class label the reviewer left on
# each box (e.g. if you used extra label classes for glyph sub-types).
# If False, every surviving box is just recorded as "glyph".
KEEP_LABEL_CLASS = False

# ---------------------------------------------------------------------------

def pct_to_pixels(value, original_width, original_height):
    x1 = value["x"] / 100.0 * original_width
    y1 = value["y"] / 100.0 * original_height
    w = value["width"] / 100.0 * original_width
    h = value["height"] / 100.0 * original_height
    return x1, y1, x1 + w, y1 + h


def image_field_to_filename(image_field):
    # Handles both "/data/local-files/?d=filename.png" and plain URLs/paths.
    if "?d=" in image_field:
        return os.path.basename(image_field.split("?d=")[-1])
    return os.path.basename(image_field)


def main():
    with open(EXPORT_JSON) as f:
        tasks = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_boxes = 0

    for task in tasks:
        image_name = image_field_to_filename(task["data"]["image"])
        page_stem = os.path.splitext(image_name)[0]

        # Use the latest annotation if there are several (e.g. re-reviewed).
        annotations = task.get("annotations", [])
        if not annotations:
            continue
        annotation = annotations[-1]

        rows = []
        for i, result in enumerate(annotation.get("result", [])):
            if result.get("type") != "rectanglelabels":
                continue
            value = result["value"]
            ow, oh = result["original_width"], result["original_height"]
            x1, y1, x2, y2 = pct_to_pixels(value, ow, oh)
            label = value["rectanglelabels"][0] if value.get("rectanglelabels") else "glyph"
            if not KEEP_LABEL_CLASS:
                label = "glyph"
            rows.append([i, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), label])

        out_path = os.path.join(OUTPUT_DIR, f"{page_stem}_final.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["box_id", "x1", "y1", "x2", "y2", "label"])
            writer.writerows(rows)

        print(f"{image_name}: {len(rows)} final boxes -> {out_path}")
        total_boxes += len(rows)

    print(f"\nTotal final boxes across all pages: {total_boxes}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
