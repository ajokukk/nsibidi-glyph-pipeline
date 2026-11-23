"""
Trace a glyph to a vector outline and compare it side-by-side against the
original raster image.

Uses `potracer` (pure-Python port of potrace) and `cairosvg` (pure-Python
SVG rasterizer) — both plain `pip install`, no external binaries required,
so this runs the same on Windows/PyCharm as anywhere else:

    pip install potracer cairosvg pillow numpy

Usage (CLI overrides the defaults below):
    python trace_compare.py --images "path/to/glyph1.png" "path/to/glyph2.png" --out compare.png

Or just edit IMAGES / OUT_PATH below and hit Run — no CLI args needed.
"""

import argparse
import io
import os
import numpy as np
import potrace
import cairosvg
from PIL import Image

# ---------------------------------------------------------------------------
# EDIT THESE to point at your actual glyph files, then just hit Run.
# ---------------------------------------------------------------------------
IMAGES = [
    r"<HOME>\PycharmProjects\NsibidiDataEfficientPipeline\dataset\0355 pot.png",
    r"<HOME>\PycharmProjects\NsibidiDataEfficientPipeline\dataset\0838 bank.png",
    r"<HOME>\PycharmProjects\NsibidiDataEfficientPipeline\dataset\0832 season.png",
    r"<HOME>\PycharmProjects\NsibidiDataEfficientPipeline\dataset\0882 he-goat.png",
    r"<HOME>\PycharmProjects\NsibidiDataEfficientPipeline\dataset\0844 heat.png",
    r"<HOME>\PycharmProjects\NsibidiDataEfficientPipeline\dataset\0034 guard.png"
]
OUT_PATH = r"<HOME>\PycharmProjects\NsibidiDataEfficientPipeline\output\compare.png"
# ---------------------------------------------------------------------------


def image_to_svg_paths(image_path, threshold=128):
    """Binarize a glyph image and trace it to vector path data with potracer."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    bitmap = potrace.Bitmap(arr < threshold)  # True = ink pixel
    path = bitmap.trace()
    return path, img.size


def path_to_svg_string(path, size, edge_tol=1.0):
    """
    potracer's Bitmap.trace() always includes one extra curve that traces
    the full canvas border (a background artifact, not glyph content) —
    confirmed by checking its bounding box against the image size. Left in,
    it cancels out the real glyph under the evenodd fill rule and inverts
    the result. Drop it before building the SVG path.
    """
    w, h = size
    d_parts = []

    for curve in path:
        pts = np.array([[p.x, p.y] for p in curve.decomposition_points])
        touches_all_edges = (
            pts[:, 0].min() <= edge_tol and pts[:, 1].min() <= edge_tol and
            pts[:, 0].max() >= w - edge_tol and pts[:, 1].max() >= h - edge_tol
        )
        if touches_all_edges:
            continue  # background artifact, not real glyph content

        d = f"M{curve.start_point.x},{curve.start_point.y}"
        for seg in curve:
            if seg.is_corner:
                d += f" L{seg.c.x},{seg.c.y} L{seg.end_point.x},{seg.end_point.y}"
            else:
                d += (f" C{seg.c1.x},{seg.c1.y} {seg.c2.x},{seg.c2.y} "
                      f"{seg.end_point.x},{seg.end_point.y}")
        d += " Z"
        d_parts.append(d)

    path_d = " ".join(d_parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">'
        f'<path d="{path_d}" fill="black" fill-rule="evenodd"/></svg>'
    )


def trace_glyph(image_path, out_svg=None, out_png=None, render_size=300):
    """
    Trace one glyph image to an SVG (exact vector outline, holes handled via
    the evenodd fill rule) and optionally rasterize it back to PNG for
    visual comparison.

    Returns (svg_string, rendered_png_bytes_or_None)
    """
    path, size = image_to_svg_paths(image_path)
    svg_string = path_to_svg_string(path, size)

    if out_svg:
        with open(out_svg, "w") as f:
            f.write(svg_string)

    png_bytes = None
    if out_png or render_size:
        png_bytes = cairosvg.svg2png(
            bytestring=svg_string.encode(), output_width=render_size, output_height=render_size
        )
        if out_png:
            with open(out_png, "wb") as f:
                f.write(png_bytes)

    return svg_string, png_bytes


def _flatten_on_white(im, size):
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    bg.paste(im, mask=im.split()[-1])
    return bg.convert("RGB").resize(size)


def compare_grid(image_paths, out_path, cell_size=(300, 300), gap=10):
    """
    Build a side-by-side [original | traced] comparison grid, one row per
    input glyph — reproduces the layout used to sanity-check the extractor.
    """
    rows = []
    for p in image_paths:
        _, png_bytes = trace_glyph(p, render_size=cell_size[0])
        original = Image.open(p).convert("RGB").resize(cell_size)
        traced = _flatten_on_white(Image.open(io.BytesIO(png_bytes)), cell_size)
        rows.append((original, traced))

    w, h = cell_size
    sheet = Image.new("RGB", (2 * w + gap, len(rows) * (h + gap) - gap), "white")
    for i, (orig, traced) in enumerate(rows):
        y = i * (h + gap)
        sheet.paste(orig, (0, y))
        sheet.paste(traced, (w + gap, y))
    sheet.save(out_path)
    return out_path


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", default=IMAGES, help="One or more glyph image paths")
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    missing = [p for p in args.images if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Image(s) not found: {missing}\n"
            f"Edit IMAGES at the top of trace_compare.py, or pass --images."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    compare_grid(args.images, args.out)
    print(f"Saved comparison grid: {args.out}")


if __name__ == "__main__":
    _main()
