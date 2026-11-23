"""
Automatic Nsibidi Glyph Variation Generator (v3 — skeleton-graph structural
deformation + folder input + collated degradation channels)
============================================================================

Replaces hand-written procedural glyph scripts (crossbow.py, ironbar.py,
lover.py, base_building_complex.py, Forest_bush.py, and the ~140 others
like them, which each manually encode one glyph's topology as coordinates)
with a single automatic pipeline:

    1. For every image in INPUT_DIR: binarize it, skeletonize it, and parse
       the skeleton into a graph — nodes at junctions/endpoints, edges as
       the stroke paths between them (with per-edge thickness recovered
       from the distance transform).
    2. Normalize that graph to a canonical size/position on a canvas.
    3. For each variation:
         - STRUCTURAL deformation: treat each connected part of the graph
           as a bone hierarchy rooted at its main junction. Every edge
           (bone) gets its own small random rotation and length scale,
           cascading down the tree — so junction positions shift and
           branch lengths vary while the topology (what connects to what)
           stays identical. This is what reproduces the effect you pointed
           out in the Forest_bush.py samples: same skeleton, different
           joining heights and branch lengths.
         - Pure closed loops (rings with no junctions, e.g. a circle) get
           an organic sine-wave "bow" + fine per-point jitter instead,
           since there's no joint to move.
         - A global rotate/scale/translate + fine hand-tremor jitter is
           then applied to the whole assembled glyph.
    4. Render every edge/loop as an actual stroke (using its own recovered
       thickness) and rasterize to PNG.
    5. Run the same collated environmental-degradation pass as before
       (paper grain/patches, scanner bleed, salt-and-pepper noise, heavy
       water-damage patches, vignette/shadow/highlight/deep-fade).

You only ever drop reference PNGs into INPUT_DIR — no manual coordinate
math, no per-glyph script, no per-glyph parameter tuning.

Install (no potrace / poppler / other system binaries needed):
    pip install cairosvg pillow numpy scikit-image scipy

Usage:
    Edit INPUT_DIR / OUTPUT_DIR / VARIATIONS_PER_GLYPH below, then:
        python auto_glyph_variator.py
"""

import io
import os
import glob
import math
import random

import numpy as np
import cairosvg
from PIL import Image, ImageFilter, ImageEnhance
from skimage.morphology import skeletonize
from scipy import ndimage

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------

INPUT_DIR = r"<HOME>\PycharmProjects\NsibidiDataEfficientPipeline\dataset"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

OUTPUT_DIR = r"<ROOT>\nsibidi_training_data\auto_variants_deformation2"
VARIATIONS_PER_GLYPH = 30
CANVAS_SIZE = 100

# Shape/geometry augmentation ranges.
SCALE_RANGE = (0.45, 0.90)
ROTATION_RANGE_DEG = (-20, 20)
TRANSLATE_JITTER_FRAC = 0.02
POINT_JITTER_FRAC = 0.008           # fine hand-tremor amplitude, canvas-fraction
ORGANIC_BOW_FRAC = 0.03             # loop-component sine "belly" drift, canvas-fraction

# Structural (skeleton bone) deformation ranges — this is what varies where
# joints sit and how long each branch is, per the Forest_bush.py observation.
BONE_ANGLE_JITTER_DEG = 4.0         # per-edge rotation, cascades down the tree
BONE_LENGTH_SCALE_RANGE = (0.85, 1.18)
MIN_SPUR_LENGTH_FRAC = 0.05         # skeleton pruning threshold, canvas-fraction

# Ink tone ranges.
INK_TONE_NORMAL_RANGE = (0, 35)
INK_TONE_FADED_RANGE = (50, 110)

# Environmental degradation ranges.
FADING_RANGE = (0.1, 0.55)
HEAVY_FADING_RANGE = (0.40, 0.75)

CHANNEL_WEIGHTS = {
    "clean": 0.29,
    "blur": 0.20,
    "scanner_bleed": 0.13,
    "salt_pepper": 0.13,
    "light_patches": 0.15,
    "heavy_patches": 0.10,
}

PREVIEW_MODE = False
PREVIEW_COUNT = 6

# ---------------------------------------------------------------------------
# 1. Skeleton graph extraction
# ---------------------------------------------------------------------------

NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def load_binary_mask(path, threshold=128):
    arr = np.array(Image.open(path).convert("L"))
    return arr < threshold


def get_skeleton_and_dist(mask):
    return skeletonize(mask), ndimage.distance_transform_edt(mask)


def _neighbors(p, skel_set):
    y, x = p
    return [(y + dy, x + dx) for dy, dx in NEIGH8 if (y + dy, x + dx) in skel_set]


def build_graph(skel):
    """
    Parse a skeletonized binary image into: nodes {id: (x,y)}, edges [{start,
    end, path:[(y,x),...]}], loop_paths [ [(y,x),...], ... ] for any closed
    ring with no junctions anywhere along it (handled separately since it
    has no joint to structurally deform).
    """
    ys, xs = np.nonzero(skel)
    skel_set = set(zip(ys.tolist(), xs.tolist()))
    degree = {p: len(_neighbors(p, skel_set)) for p in skel_set}
    node_pixels = {p for p, d in degree.items() if d != 2}

    visited = set()
    clusters = []
    for p in node_pixels:
        if p in visited:
            continue
        stack, comp = [p], []
        visited.add(p)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for q in _neighbors(cur, skel_set):
                if q in node_pixels and q not in visited:
                    visited.add(q); stack.append(q)
        clusters.append(comp)

    pixel_to_node, node_pos = {}, {}
    for i, comp in enumerate(clusters):
        for p in comp:
            pixel_to_node[p] = i
        ys_c = [p[0] for p in comp]; xs_c = [p[1] for p in comp]
        node_pos[i] = (sum(xs_c) / len(xs_c), sum(ys_c) / len(ys_c))

    edge_pixels_visited = set()
    edges = []
    for i, comp in enumerate(clusters):
        for p in comp:
            for q in _neighbors(p, skel_set):
                if q in node_pixels or q in edge_pixels_visited:
                    continue
                path = [p, q]
                edge_pixels_visited.add(q)
                prev, cur = p, q
                while True:
                    nbrs = [n for n in _neighbors(cur, skel_set) if n != prev]
                    node_hits = [n for n in nbrs if n in node_pixels]
                    if node_hits:
                        path.append(node_hits[0])
                        edges.append({"start": i, "end": pixel_to_node[node_hits[0]], "path": path})
                        break
                    plain = [n for n in nbrs if n not in edge_pixels_visited]
                    if not plain:
                        edges.append({"start": i, "end": i, "path": path})
                        break
                    nxt = plain[0]
                    edge_pixels_visited.add(nxt); path.append(nxt)
                    prev, cur = cur, nxt

    # Any skeleton pixel not part of a node cluster or a walked edge belongs
    # to a pure closed loop with no junctions anywhere (e.g. a plain ring)
    # floating separately from any branching structure in the same image.
    used = set(node_pixels) | edge_pixels_visited
    remaining = skel_set - used
    loop_paths = []
    while remaining:
        start = next(iter(remaining))
        stack, comp, seen = [start], [], {start}
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for q in _neighbors(cur, skel_set):
                if q not in seen:
                    seen.add(q); stack.append(q)
        remaining -= seen
        loop_paths.append(comp)

    return node_pos, edges, loop_paths


def edge_length(edge):
    pts = edge["path"]; total = 0.0
    for i in range(1, len(pts)):
        y0, x0 = pts[i - 1]; y1, x1 = pts[i]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def node_degree(nid, edges):
    return sum(1 for e in edges if e["start"] == nid or e["end"] == nid)


def prune_short_edges(nodes, edges, min_len):
    """Collapse skeletonization artifacts: tiny spurs at junctions AND
    redundant double-junctions from thick corners, both showing up as very
    short edges. Merges each short edge's two endpoints into one node."""
    nodes = dict(nodes); edges = [dict(e) for e in edges]
    changed = True
    while changed:
        changed = False
        cands = sorted(
            [(edge_length(e), i) for i, e in enumerate(edges) if e["start"] != e["end"]],
            key=lambda t: t[0]
        )
        cands = [c for c in cands if c[0] < min_len]
        if not cands:
            break
        _, idx = cands[0]
        e = edges.pop(idx)
        a, b = e["start"], e["end"]
        pa, pb = nodes[a], nodes[b]
        nodes[a] = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
        del nodes[b]
        for other in edges:
            if other["start"] == b: other["start"] = a
            if other["end"] == b: other["end"] = a
        edges = [oe for oe in edges if not (oe["start"] == oe["end"] and edge_length(oe) < 1.0)]
        changed = True
    return nodes, edges


def connected_components(nodes, edges):
    adj = {n: [] for n in nodes}
    for idx, e in enumerate(edges):
        adj[e["start"]].append((e["end"], idx))
        adj[e["end"]].append((e["start"], idx))
    seen, comps = set(), []
    for n in nodes:
        if n in seen:
            continue
        stack, comp_nodes = [n], {n}
        seen.add(n)
        while stack:
            cur = stack.pop()
            for nb, _ in adj[cur]:
                if nb not in comp_nodes:
                    comp_nodes.add(nb); seen.add(nb); stack.append(nb)
        comp_edges = [i for i, e in enumerate(edges) if e["start"] in comp_nodes and e["end"] in comp_nodes]
        comps.append((comp_nodes, comp_edges))
    return comps


# ---------------------------------------------------------------------------
# 2. Normalize to a canonical canvas
# ---------------------------------------------------------------------------

def extract_canonical_skeleton(image_path, canvas_size=100, fill_fraction=1.0, threshold=128):
    """
    Returns a dict: {"nodes": {id:(x,y)}, "edges":[{start,end,path:[(x,y)...],width}],
    "loops": [{"path":[(x,y)...],"width":w}], "dot": {"pos":(x,y),"radius":r} or None}
    all normalized so the glyph's longest bounding-box dimension equals
    fill_fraction * canvas_size, centered on the canvas.
    """
    mask = load_binary_mask(image_path, threshold=threshold)
    skel, dist = get_skeleton_and_dist(mask)

    if skel.sum() == 0:
        # No skeleton at all (skeletonize can collapse a small blob to
        # nothing) -- treat the whole ink mask as a single dot.
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            raise ValueError(f"No ink found in {image_path}.")
        cx, cy = xs.mean(), ys.mean()
        r = max((xs.max() - xs.min()), (ys.max() - ys.min())) / 2 or 3.0
        all_pts = [(cx, cy)]
    else:
        nodes, edges, loop_paths = build_graph(skel)
        min_spur = canvas_size * MIN_SPUR_LENGTH_FRAC  # in RAW pixel units pre-normalization is fine
        # use a resolution-appropriate spur threshold based on the actual image size
        h, w = mask.shape
        min_spur_px = max(h, w) * MIN_SPUR_LENGTH_FRAC
        nodes, edges = prune_short_edges(nodes, edges, min_len=min_spur_px)

        all_pts = list(nodes.values())
        for lp in loop_paths:
            all_pts.extend([(x, y) for (y, x) in lp])
        if not all_pts:
            ys, xs = np.nonzero(mask)
            cx, cy = xs.mean(), ys.mean()
            all_pts = [(cx, cy)]

    pts_arr = np.array(all_pts)
    minx, miny = pts_arr.min(axis=0); maxx, maxy = pts_arr.max(axis=0)
    bw, bh = maxx - minx, maxy - miny
    scale = fill_fraction * canvas_size / max(bw, bh, 1e-6)
    bcx, bcy = (minx + maxx) / 2, (miny + maxy) / 2
    ccx, ccy = canvas_size / 2, canvas_size / 2

    def norm_pt(x, y):
        return ((x - bcx) * scale + ccx, (y - bcy) * scale + ccy)

    if skel.sum() == 0:
        return {"nodes": {}, "edges": [], "loops": [], "dot": {"pos": norm_pt(cx, cy), "radius": r * scale}}

    def edge_width(path):
        vals = [dist[y, x] for (y, x) in path]
        return 2.0 * sorted(vals)[len(vals) // 2] * scale

    norm_nodes = {nid: norm_pt(*p) for nid, p in nodes.items()}
    norm_edges = []
    for e in edges:
        path_xy = [norm_pt(x, y) for (y, x) in e["path"]]
        norm_edges.append({"start": e["start"], "end": e["end"], "path": path_xy, "width": edge_width(e["path"])})

    norm_loops = []
    for lp in loop_paths:
        path_xy = [norm_pt(x, y) for (y, x) in lp]
        norm_loops.append({"path": path_xy, "width": edge_width(lp)})

    all_widths = [e["width"] for e in norm_edges] + [lp["width"] for lp in norm_loops]
    avg_width_frac = (sum(all_widths) / len(all_widths) / canvas_size) if all_widths else 0.05

    return {"nodes": norm_nodes, "edges": norm_edges, "loops": norm_loops, "dot": None,
            "avg_width_frac": avg_width_frac}


# ---------------------------------------------------------------------------
# 3. Structural (bone-hierarchy) deformation for branching parts
# ---------------------------------------------------------------------------

def _build_spanning_tree(comp_nodes, comp_edge_idxs, edges, root):
    adj = {n: [] for n in comp_nodes}
    for idx in comp_edge_idxs:
        e = edges[idx]
        adj[e["start"]].append((e["end"], idx))
        adj[e["end"]].append((e["start"], idx))
    parent = {root: None}
    tree_edge_idx = {}
    order = [root]
    visited = {root}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nb, idx in adj[cur]:
            if nb not in visited:
                visited.add(nb); parent[nb] = cur; tree_edge_idx[nb] = idx
                order.append(nb); stack.append(nb)
    return parent, tree_edge_idx, order


def _transform_edge_path(path_pts_xy, p_orig, c_orig, p_new, c_new):
    orig_vec = np.array(c_orig) - np.array(p_orig)
    orig_len = np.linalg.norm(orig_vec)
    if orig_len < 1e-6:
        return [p_new for _ in path_pts_xy]
    u = orig_vec / orig_len
    perp = np.array([-u[1], u[0]])
    new_vec = np.array(c_new) - np.array(p_new)
    new_len = np.linalg.norm(new_vec)
    u2 = (new_vec / new_len) if new_len > 1e-6 else u
    perp2 = np.array([-u2[1], u2[0]])
    scale = new_len / orig_len
    p_orig_a = np.array(p_orig)
    out = []
    for (x, y) in path_pts_xy:
        rel = np.array([x, y]) - p_orig_a
        along, pperp = np.dot(rel, u), np.dot(rel, perp)
        newp = np.array(p_new) + along * scale * u2 + pperp * scale * perp2
        out.append((float(newp[0]), float(newp[1])))
    return out


def deform_component(nodes, edges, comp_nodes, comp_edge_idxs,
                      angle_jitter_deg=BONE_ANGLE_JITTER_DEG,
                      length_scale_range=BONE_LENGTH_SCALE_RANGE):
    """
    Treat this connected branching part as a bone hierarchy rooted at its
    highest-degree junction. Every edge gets its own random rotation
    (cascading down the tree) and length scale, so joint positions and
    branch lengths vary while connectivity (topology) never changes.
    """
    root = max(comp_nodes, key=lambda n: node_degree(n, [edges[i] for i in comp_edge_idxs]))
    parent, tree_edge_idx, order = _build_spanning_tree(comp_nodes, comp_edge_idxs, edges, root)

    new_pos = {root: nodes[root]}
    accum_rot = {root: 0.0}
    for node in order[1:]:
        p = parent[node]
        p_orig, c_orig = nodes[p], nodes[node]
        orig_vec = np.array(c_orig) - np.array(p_orig)
        orig_len = np.linalg.norm(orig_vec)
        orig_angle = math.atan2(orig_vec[1], orig_vec[0])
        local_delta = math.radians(random.uniform(-angle_jitter_deg, angle_jitter_deg))
        scale = random.uniform(*length_scale_range)
        new_angle = orig_angle + accum_rot[p] + local_delta
        new_len = orig_len * scale
        p_new = np.array(new_pos[p])
        new_pos[node] = (p_new[0] + new_len * math.cos(new_angle), p_new[1] + new_len * math.sin(new_angle))
        accum_rot[node] = accum_rot[p] + local_delta

    deformed_edges = []
    for idx in comp_edge_idxs:
        e = edges[idx]
        a, b = e["start"], e["end"]
        new_path = _transform_edge_path(e["path"], nodes[a], nodes[b], new_pos[a], new_pos[b])
        deformed_edges.append({"path": new_path, "width": e["width"]})
    return deformed_edges


# ---------------------------------------------------------------------------
# 4. Cosmetic finishing: organic bow (loops) + fine hand-tremor jitter (all)
# ---------------------------------------------------------------------------

def _add_organic_bow(poly, max_bow_amp):
    n = len(poly)
    if n < 3 or max_bow_amp <= 0:
        return poly
    dists = [0.0]
    for i in range(1, n):
        dx = poly[i][0] - poly[i - 1][0]; dy = poly[i][1] - poly[i - 1][1]
        dists.append(dists[-1] + math.hypot(dx, dy))
    total = dists[-1] if dists[-1] > 1e-6 else 1.0
    bow_amp = random.uniform(0, max_bow_amp)
    n_humps = random.choice([1, 1, 2])
    phase = random.uniform(0, 2 * math.pi)
    out = []
    for i, (x, y) in enumerate(poly):
        t = dists[i] / total
        i0, i1 = max(0, i - 1), min(n - 1, i + 1)
        tx = poly[i1][0] - poly[i0][0]; ty = poly[i1][1] - poly[i0][1]
        length = math.hypot(tx, ty) or 1.0
        px, py = -ty / length, tx / length
        disp = bow_amp * math.sin(t * n_humps * math.pi + phase)
        out.append((x + px * disp, y + py * disp))
    return out


def _jitter_points(poly, amp):
    if amp <= 0:
        return poly
    return [(x + random.uniform(-amp, amp), y + random.uniform(-amp, amp)) for (x, y) in poly]


def _apply_global_transform(poly, canvas_size, scale, angle_rad, tx, ty):
    cx = cy = canvas_size / 2
    s_ang, c_ang = math.sin(angle_rad), math.cos(angle_rad)
    out = []
    for (x, y) in poly:
        x -= cx; y -= cy
        x *= scale; y *= scale
        xr = x * c_ang - y * s_ang
        yr = x * s_ang + y * c_ang
        out.append((xr + cx + tx, yr + cy + ty))
    return out


# ---------------------------------------------------------------------------
# 5. Render strokes to raster
# ---------------------------------------------------------------------------

def render_strokes(stroke_list, canvas_size, ink_tone, closed_flags=None):
    """stroke_list: list of (polyline_points, width). closed_flags: matching
    list of bool, True to close the path with Z (for loops)."""
    if closed_flags is None:
        closed_flags = [False] * len(stroke_list)
    fill = f"rgb({ink_tone},{ink_tone},{ink_tone})"
    parts = []
    for (path_xy, width), closed in zip(stroke_list, closed_flags):
        if len(path_xy) < 2:
            continue
        d = f"M{path_xy[0][0]},{path_xy[0][1]}"
        for (x, y) in path_xy[1:]:
            d += f" L{x},{y}"
        if closed:
            d += " Z"
        parts.append(f'<path d="{d}" stroke="{fill}" stroke-width="{max(width, 1.2):.2f}" '
                     f'stroke-linecap="round" stroke-linejoin="round" fill="none"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" '
           f'viewBox="0 0 {canvas_size} {canvas_size}">' + "".join(parts) + "</svg>")
    png_bytes = cairosvg.svg2png(bytestring=svg.encode(), output_width=canvas_size, output_height=canvas_size)
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    bg.paste(im, mask=im.split()[-1])
    return bg.convert("RGB")


def render_dot(pos, radius, canvas_size, ink_tone):
    fill = f"rgb({ink_tone},{ink_tone},{ink_tone})"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" '
           f'viewBox="0 0 {canvas_size} {canvas_size}">'
           f'<circle cx="{pos[0]}" cy="{pos[1]}" r="{max(radius,1.5)}" fill="{fill}"/></svg>')
    png_bytes = cairosvg.svg2png(bytestring=svg.encode(), output_width=canvas_size, output_height=canvas_size)
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    bg.paste(im, mask=im.split()[-1])
    return bg.convert("RGB")


# ---------------------------------------------------------------------------
# 6. Collated environmental degradation channels
#    (crossbow.py / ironbar.py / lover.py / base_building_complex.py)
# ---------------------------------------------------------------------------

def apply_paper_grain_and_patches(img, fading_intensity):
    img_arr = np.array(img).astype(float)
    h, w = img_arr.shape[:2]
    if fading_intensity > 0.1:
        ink_mask = img_arr < 220
        img_arr[ink_mask] += (255.0 - img_arr[ink_mask]) * (fading_intensity * 0.4)
    if random.random() < 0.5:
        img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))
        img_arr = np.array(img).astype(float)
    grain = np.random.normal(0, random.uniform(3.0, 6.0), img_arr.shape)
    img_arr = np.clip(img_arr + grain, 0, 255)
    min_dim = min(h, w)
    for _ in range(random.randint(2, 4)):
        cx = random.randint(int(0.05 * w), int(0.95 * w))
        cy = random.randint(int(0.05 * h), int(0.95 * h))
        radius = random.randint(max(2, int(0.05 * min_dim)), max(3, int(0.18 * min_dim)))
        y_grid, x_grid = np.ogrid[:h, :w]
        mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius ** 2
        img_arr[mask] += random.uniform(20, 65)
    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))


def apply_heavy_patched_textures(img, fading_intensity):
    img_arr = np.array(img).astype(float)
    h, w = img_arr.shape[:2]
    if fading_intensity > 0.1:
        ink_mask = img_arr < 235
        img_arr[ink_mask] += (255.0 - img_arr[ink_mask]) * (fading_intensity * 0.65)
    if random.random() < 0.7:
        img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 0.8)))
        img_arr = np.array(img).astype(float)
    grain = np.random.normal(0, random.uniform(5.0, 9.0), img_arr.shape)
    img_arr = np.clip(img_arr + grain, 0, 255)
    min_dim = min(h, w)
    for _ in range(random.randint(4, 7)):
        cx = random.randint(int(0.05 * w), int(0.95 * w))
        cy = random.randint(int(0.05 * h), int(0.95 * h))
        radius = random.randint(max(3, int(0.10 * min_dim)), max(4, int(0.28 * min_dim)))
        y_grid, x_grid = np.ogrid[:h, :w]
        mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius ** 2
        img_arr[mask] += random.uniform(40, 95)
    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))


def apply_subtle_clean_grain(img):
    img_arr = np.array(img).astype(float)
    grain = np.random.normal(0, 2.0, img_arr.shape)
    return Image.fromarray(np.clip(img_arr + grain, 0, 255).astype(np.uint8))


def apply_scanner_bleed(img):
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(2.5, 3.5))
    return img.point(lambda p: 255 if p > 140 else 0)


def apply_salt_pepper_noise(img):
    img_np = np.array(img.convert("L"))
    noise_density = random.uniform(0.015, 0.03)
    num_salt = int(noise_density * img_np.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img_np.shape]
    img_np[tuple(coords)] = 255
    coords = [np.random.randint(0, i - 1, num_salt) for i in img_np.shape]
    img_np[tuple(coords)] = 0
    img = Image.fromarray(img_np).convert("RGB")
    return img.filter(ImageFilter.GaussianBlur(0.3))


def apply_vignette(img):
    w, h = img.size
    strength = random.uniform(0.25, 0.38)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    vmask = np.clip(1.0 - (np.sqrt(x ** 2 + y ** 2) * strength), 0.0, 1.0)
    arr = np.array(img).astype(float)
    for c in range(3):
        arr[:, :, c] *= vmask
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_shadow(img):
    w, h = img.size
    arr = np.array(img).astype(float)
    angle = random.uniform(0, 2 * math.pi)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    grad = (x * math.cos(angle) + y * math.sin(angle) + 1.0) / 2.0
    factor = 0.82 + (grad * 0.18)
    for c in range(3):
        arr[:, :, c] *= factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_highlight(img):
    w, h = img.size
    arr = np.array(img).astype(float)
    angle = random.uniform(0, 2 * math.pi)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    grad = (x * math.cos(angle) + y * math.sin(angle) + 1.0) / 2.0
    factor = 1.0 + (grad * 0.14)
    for c in range(3):
        arr[:, :, c] *= factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_deep_fading(img):
    w, h = img.size
    arr = np.array(img).astype(float)
    angle = random.uniform(0, 2 * math.pi)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    grad = (x * math.cos(angle) + y * math.sin(angle) + 1.0) / 2.0
    fade_mask = grad * random.uniform(0.22, 0.36)
    for c in range(3):
        arr[:, :, c] += (255.0 - arr[:, :, c]) * fade_mask
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _clamp01(v):
    return max(0.0, min(1.0, v))


def _lerp(a, b, t):
    return a + (b - a) * t


def get_adaptive_params(avg_width_frac):
    """
    Bold, densely-packed glyphs (thick strokes relative to canvas, features
    close together -- e.g. a glyph with two closely-spaced internal rungs)
    are much more prone to internal features touching/merging when strokes
    get thickened or bones shrink, since reconstruction approximates each
    stroke as a fixed-width tube rather than tracing the real ink boundary.
    Thinner, more open structures (e.g. a branching tree with generous gaps)
    can take much more aggressive deformation without any merging risk.
    This scales deformation/thickening aggressiveness down automatically as
    average stroke width (relative to canvas) increases, rather than
    needing a manual per-glyph setting.
    """
    bold_factor = _clamp01((avg_width_frac - 0.04) / 0.06)  # 0 = thin, 1 = very bold
    return {
        "length_scale_range": (
            _lerp(BONE_LENGTH_SCALE_RANGE[0], 0.94, bold_factor),
            _lerp(BONE_LENGTH_SCALE_RANGE[1], 1.08, bold_factor),
        ),
        "angle_jitter_deg": _lerp(BONE_ANGLE_JITTER_DEG, 2.0, bold_factor),
        "thicken_prob": _lerp(0.20, 0.03, bold_factor),
    }


def apply_ink_weight_jitter(img, thicken_prob=0.20, thin_prob=0.20):
    roll = random.random()
    if roll < thicken_prob:
        return img.filter(ImageFilter.MinFilter(3))
    elif roll < thicken_prob + thin_prob:
        return img.filter(ImageFilter.MaxFilter(3))
    return img


def choose_channel():
    names = list(CHANNEL_WEIGHTS.keys())
    weights = list(CHANNEL_WEIGHTS.values())
    return random.choices(names, weights=weights, k=1)[0]


def ink_tone_for_channel(channel):
    if channel == "heavy_patches":
        return random.randint(*INK_TONE_FADED_RANGE)
    return random.randint(*INK_TONE_NORMAL_RANGE)


def apply_channel_degradation(img, channel, fading_intensity):
    if channel == "clean":
        return apply_subtle_clean_grain(img)
    elif channel == "blur":
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
    elif channel == "scanner_bleed":
        return apply_scanner_bleed(img)
    elif channel == "salt_pepper":
        return apply_salt_pepper_noise(img)
    elif channel == "light_patches":
        return apply_paper_grain_and_patches(img, fading_intensity)
    elif channel == "heavy_patches":
        return apply_heavy_patched_textures(img, fading_intensity)
    return img


def apply_environmental_degradation(img, channel, fading_intensity, thicken_prob=0.20):
    img = apply_ink_weight_jitter(img, thicken_prob=thicken_prob, thin_prob=0.20)
    img = apply_channel_degradation(img, channel, fading_intensity)
    if random.random() < 0.05: img = apply_vignette(img)
    if random.random() < 0.05: img = apply_shadow(img)
    if random.random() < 0.05: img = apply_highlight(img)
    if random.random() < 0.05: img = apply_deep_fading(img)
    return img


# ---------------------------------------------------------------------------
# 7. Driver
# ---------------------------------------------------------------------------

def discover_glyph_images(input_dir, exts=IMAGE_EXTS):
    paths = [f for f in sorted(glob.glob(os.path.join(input_dir, "*")))
             if os.path.splitext(f)[1].lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images with extensions {exts} found in {input_dir}")
    return paths


def generate_variation(canonical, canvas_size):
    channel = choose_channel()
    ink_tone = ink_tone_for_channel(channel)
    fading_intensity = random.uniform(*(HEAVY_FADING_RANGE if channel == "heavy_patches" else FADING_RANGE))
    adaptive = get_adaptive_params(canonical.get("avg_width_frac", 0.05))

    if canonical.get("dot"):
        scale = random.uniform(*SCALE_RANGE)
        pos = canonical["dot"]["pos"]
        radius = canonical["dot"]["radius"] * scale
        img = render_dot(pos, radius, canvas_size, ink_tone)
        return apply_environmental_degradation(img, channel, fading_intensity, thicken_prob=adaptive["thicken_prob"])

    nodes, edges, loops = canonical["nodes"], canonical["edges"], canonical["loops"]

    stroke_list, closed_flags = [], []

    if nodes:
        comps = connected_components(nodes, edges)
        for comp_nodes, comp_edge_idxs in comps:
            if len(comp_nodes) < 2 or not comp_edge_idxs:
                continue
            deformed = deform_component(nodes, edges, comp_nodes, comp_edge_idxs,
                                         angle_jitter_deg=adaptive["angle_jitter_deg"],
                                         length_scale_range=adaptive["length_scale_range"])
            for d in deformed:
                jitter_amp = canvas_size * POINT_JITTER_FRAC * random.uniform(0.5, 1.5)
                poly = _jitter_points(d["path"], jitter_amp)
                stroke_list.append((poly, d["width"]))
                closed_flags.append(False)

    for lp in loops:
        bow_amp = canvas_size * ORGANIC_BOW_FRAC * random.uniform(0.3, 1.2)
        jitter_amp = canvas_size * POINT_JITTER_FRAC * random.uniform(0.5, 1.5)
        poly = _add_organic_bow(lp["path"], bow_amp)
        poly = _jitter_points(poly, jitter_amp)
        stroke_list.append((poly, lp["width"]))
        closed_flags.append(True)

    if not stroke_list:
        # fell through with nothing renderable -- fall back to a dot at canvas center
        img = render_dot((canvas_size / 2, canvas_size / 2), canvas_size * 0.05, canvas_size, ink_tone)
        return apply_environmental_degradation(img, channel, fading_intensity, thicken_prob=adaptive["thicken_prob"])

    scale = random.uniform(*SCALE_RANGE)
    angle_rad = math.radians(random.uniform(*ROTATION_RANGE_DEG))
    tx = random.uniform(-canvas_size * TRANSLATE_JITTER_FRAC, canvas_size * TRANSLATE_JITTER_FRAC)
    ty = random.uniform(-canvas_size * TRANSLATE_JITTER_FRAC, canvas_size * TRANSLATE_JITTER_FRAC)

    final_strokes = []
    for poly, width in stroke_list:
        new_poly = _apply_global_transform(poly, canvas_size, scale, angle_rad, tx, ty)
        final_strokes.append((new_poly, width * scale))

    img = render_strokes(final_strokes, canvas_size, ink_tone, closed_flags=closed_flags)
    img = apply_environmental_degradation(img, channel, fading_intensity, thicken_prob=adaptive["thicken_prob"])
    return img


def generate_variations_for_glyph(image_path, output_dir, n_variations, canvas_size):
    canonical = extract_canonical_skeleton(image_path, canvas_size=canvas_size, fill_fraction=1.0)
    name = os.path.splitext(os.path.basename(image_path))[0].replace(" ", "_")
    glyph_dir = os.path.join(output_dir, name, "png")
    os.makedirs(glyph_dir, exist_ok=True)
    for i in range(n_variations):
        img = generate_variation(canonical, canvas_size)
        img.save(os.path.join(glyph_dir, f"{name}_{i:04d}.png"))
    print(f"[{name}] wrote {n_variations} variations to {glyph_dir}")
    return glyph_dir


def preview_grid(image_path, output_dir, canvas_size, n_variations=6):
    canonical = extract_canonical_skeleton(image_path, canvas_size=canvas_size, fill_fraction=1.0)
    original = Image.open(image_path).convert("RGB").resize((canvas_size, canvas_size))
    tiles = [original] + [generate_variation(canonical, canvas_size) for _ in range(n_variations)]
    gap = 6
    sheet = Image.new("RGB", (len(tiles) * canvas_size + (len(tiles) - 1) * gap, canvas_size), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * (canvas_size + gap), 0))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "preview.png")
    sheet.save(out_path)
    print(f"Preview saved: {out_path}  (leftmost tile = original, rest = generated variations)")
    return out_path


if __name__ == "__main__":
    glyph_paths = discover_glyph_images(INPUT_DIR)
    print(f"Found {len(glyph_paths)} glyph image(s) in {INPUT_DIR}")

    if PREVIEW_MODE:
        preview_grid(glyph_paths[0], OUTPUT_DIR, CANVAS_SIZE, n_variations=PREVIEW_COUNT)
    else:
        for glyph_path in glyph_paths:
            generate_variations_for_glyph(glyph_path, OUTPUT_DIR, VARIATIONS_PER_GLYPH, CANVAS_SIZE)
        print("\nDone.")
