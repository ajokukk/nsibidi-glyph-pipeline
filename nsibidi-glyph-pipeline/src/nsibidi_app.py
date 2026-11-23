"""
Reading assistant for endangered scripts (Streamlit, runs locally on a CPU).

    pip install streamlit pillow numpy opencv-python
    streamlit run nsibidi_app.py

Keep these files in one folder:  nsibidi_app.py, box_editor.py, classify_detections.py,
plate_segmentation_triage.py (segmentation route), auto_glyph_variator1.py (LPGS), labels_en.csv (glossary).

The app asks three things, in this order:

  1. Which writing system?          (a drop-down; Nsibidi is the only one for now - see WRITING_SYSTEMS below)
  2. What do you want to do?
        A. Segment a plate, label the glyphs, and make LPGS variations of them (the Label Studio route), or
        B. Detect and recognise glyphs straight away with the 1,400-sign network.
  3. The work itself. In both routes every box can be moved, resized, drawn or deleted on the plate before it is
     confirmed. Sign numbers (S0001 ... S1400) are shown with their dictionary meaning, and the "Dictionary legend"
     tab maps every number to its glossary entry.

Everything the app writes goes under app_output/<writing system>/ : verified/, seeds/, lpgs_out/, labelstudio/, logs/.
"""
import csv
import datetime as dt
import importlib.util
import io
import json
import os
import random
import statistics
import subprocess
import sys
import time
import uuid
import zipfile

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import classify_detections as cd                      # noqa: E402
from box_editor import box_editor                      # noqa: E402

TOPK = 5
NOT_GLYPH = "(not a glyph)"
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
try:                                  # newer Streamlit replaced use_container_width by width="stretch"
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _v = (0, 0)
STRETCH = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

# ------------------------------------------------------------------------------------------ writing systems
# To add a script later, add an entry here (or put the same structure in writing_systems.json next to this file).
WRITING_SYSTEMS = {
    "Nsibidi (south-eastern Nigeria)": {
        "key": "nsibidi",
        "blurb": "Ideographic signs of the Ejagham, Ekoi, Efik, Ibibio and Igbo peoples. The dictionary used here is the "
                 "Neo-Nsibidi dictionary of 1,400 signs (sign numbers S0001 to S1400).",
        "detector_ckpt": "best.pth",
        "embedding_ckpt": "e13b_1400_mixed.pth",
        "dictionary_dirs": [r"<ROOT>\dataset\dict\proto_folders"],
        "labels_csv": "labels_en.csv",
        "segmenter": "plate_segmentation_triage.py",
        "variator": "auto_glyph_variator1.py",
        "confusable": [["S1009", "S0201"], ["S0905", "S0020"]],
        "file_as": {"S1009": "S0201"},
    },
}


def load_systems():
    systems = {k: dict(v) for k, v in WRITING_SYSTEMS.items()}
    p = os.path.join(HERE, "writing_systems.json")
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                for name, conf in json.load(f).items():
                    systems[name] = {**systems.get(name, {}), **conf}
        except Exception as ex:                                    # a broken file must not stop the app
            st.warning(f"writing_systems.json could not be read ({ex}).")
    return systems


def resolve(path):
    """A path as typed by the user, or relative to this folder."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(HERE, path)


# ------------------------------------------------------------------------------------------ dictionary and legend
@st.cache_data(show_spinner=False)
def load_labels(path, _mtime=0):
    """{S-ID: {'en': first meaning, 'cls': class, 'all': all meanings}} from labels_en.csv; empty if not found."""
    for p in (path, os.path.join(HERE, path or ""), os.path.join(HERE, "labels_en.csv")):
        if p and os.path.isfile(p):
            with open(p, newline="", encoding="utf-8") as f:
                out = {}
                for r in csv.DictReader(f):
                    sid = (r.get("file") or r.get("id") or "").strip()
                    if sid and not sid.startswith("S") and sid.isdigit():
                        sid = "S" + sid
                    if sid:
                        out[sid] = {"en": (r.get("en_label") or "").strip(), "cls": (r.get("class") or "").strip(),
                                    "all": (r.get("en_all") or "").strip()}
                return out
    return {}


def gloss(name, labels):
    en = labels.get(name, {}).get("en", "")
    return f"{name} {en}".strip()


@st.cache_data(show_spinner=False)
def proto_index(dirs):
    """{sign: path of its first prototype image} over the dictionary folders."""
    out = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for sign in sorted(os.listdir(d)):
            p = os.path.join(d, sign)
            if os.path.isdir(p) and sign not in out:
                fs = sorted(f for f in os.listdir(p) if f.lower().endswith(IMG_EXT))
                if fs:
                    out[sign] = os.path.join(p, fs[0])
    return out


@st.cache_data(show_spinner=False)
def thumb_uri(path, size=56):
    import base64
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def legend_rows(ids, labels, protos):
    import pandas as pd
    rows = []
    for i in ids:
        info = labels.get(i, {})
        rows.append({"Sign": thumb_uri(protos[i]) if i in protos else None, "Number": i, "Class": info.get("cls", ""),
                     "Meaning": info.get("en", ""), "All senses": info.get("all", "")})
    return pd.DataFrame(rows, columns=["Sign", "Number", "Class", "Meaning", "All senses"])


def show_legend_table(df, height=420):
    cfg = {"Sign": st.column_config.ImageColumn("Sign", width="small")}
    st.dataframe(df, column_config=cfg, hide_index=True, height=height, **STRETCH)


def legend_tab(cfg):
    labels = load_labels(cfg["labels_csv"])
    protos = proto_index(tuple(cfg["dictionary_dirs"]))
    st.subheader("Dictionary legend")
    st.caption("Every sign number with its dictionary meaning. The class letter is the dictionary's own: O = original "
               "(pre-1900), MM = original sign with a modified meaning; the others are later or constructed signs.")
    if not labels:
        st.info("labels_en.csv was not found next to the app, so only the sign numbers are known. Put it in this "
                "folder or give its path under Settings.")
        return
    ids = sorted(set(labels) | set(protos))
    c1, c2, c3 = st.columns([3, 2, 1])
    q = c1.text_input("Search a number or a meaning (for example S0109 or love)", "", key="leg_q").strip().lower()
    classes = sorted({labels[i]["cls"] for i in labels if labels[i]["cls"]})
    pick = c2.multiselect("Class", classes, key="leg_cls")
    limit = c3.number_input("Rows", 20, 1500, 150, 10, key="leg_n")
    sel = [i for i in ids if (not q or q in i.lower() or q in labels.get(i, {}).get("all", "").lower()
                              or q in labels.get(i, {}).get("en", "").lower())
           and (not pick or labels.get(i, {}).get("cls", "") in pick)]
    st.write(f"{len(sel)} of {len(ids)} signs" + (f" (showing the first {int(limit)})" if len(sel) > limit else ""))
    show_legend_table(legend_rows(sel[:int(limit)], labels, protos), height=520)
    if os.path.isfile(resolve(cfg["labels_csv"])):
        with open(resolve(cfg["labels_csv"]), "rb") as f:
            st.download_button("Download the glossary (CSV)", f.read(), file_name="glossary.csv", mime="text/csv")


# ------------------------------------------------------------------------------------------ small image helpers
def crop_xyxy(img, x1, y1, x2, y2, pad=0.04):
    w, h = img.size
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return img.crop((max(0, int(x1 - px)), max(0, int(y1 - py)), min(w, int(x2 + px)), min(h, int(y2 + py))))


def box_crop(img, b, pad=0.04):
    return crop_xyxy(img, b["x1"], b["y1"], b["x2"], b["y2"], pad)


def geom(b):
    return (round(b["x1"], 1), round(b["y1"], 1), round(b["x2"], 1), round(b["y2"], 1))


def moved(a, b, tol=1.5):
    return a is None or b is None or any(abs(p - q) > tol for p, q in zip(a, b))


# ------------------------------------------------------------------------------------------ models (cached)
@st.cache_resource(show_spinner="Loading the embedding network ...")
def get_embedder(resnet_ckpt, gray_norm):
    return cd.Embedder("resnet", resnet_ckpt=resnet_ckpt, gray_norm=gray_norm)


def _class_means(emb, class_files, label):
    pairs = [(c, f) for c, fs in sorted(class_files.items()) for f in fs]
    bar = st.progress(0.0, text=f"{label}: embedding {len(pairs)} images ...")
    embs, owners, chunk = [], [], 64
    for s in range(0, len(pairs), chunk):
        part = pairs[s:s + chunk]
        e, ok = cd.embed_files(emb, [f for _, f in part])
        okset = set(ok)
        owners += [c for c, f in part if f in okset]
        if len(ok):
            embs.append(e)
        bar.progress(min(1.0, (s + chunk) / max(1, len(pairs))))
    bar.empty()
    if not embs:
        raise ValueError(f"No usable images found for {label}.")
    embs = np.concatenate(embs, 0)
    out = {}
    for c in sorted(set(owners)):
        sel = np.array([o == c for o in owners])
        out[c] = (embs[sel].mean(axis=0), int(sel.sum()))
    return out


@st.cache_resource(show_spinner=False)
def get_base(resnet_ckpt, gray_norm, dirs, shots, cache_path, version):
    """Class means of the dictionary folders, read from a cache file when it still matches."""
    stamp = os.path.getmtime(resnet_ckpt) if os.path.exists(resnet_ckpt) else 0
    tag = f"{os.path.abspath(resnet_ckpt)}|{stamp}|{int(gray_norm)}|{shots}|{'|'.join(dirs)}"
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if str(z["tag"]) == tag:
            return [str(n) for n in z["names"]], z["means"], z["counts"]
    good = [d for d in dirs if os.path.isdir(d) and cd.list_class_files(d)]
    if not good:
        raise ValueError("No usable dictionary folder. Each folder must exist and contain one sub-folder per sign "
                         "with images in it. Check the path under Settings, for example "
                         "<ROOT>\\dataset\\dict\\proto_folders")
    emb = get_embedder(resnet_ckpt, gray_norm)
    cm = _class_means(emb, cd.sample_support(good, shots, seed=42), "Dictionary")
    names = sorted(cm)
    means = np.stack([cm[n][0] for n in names]).astype(np.float32)
    counts = np.array([cm[n][1] for n in names], dtype=np.int64)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, names=np.array(names), means=means, counts=counts, tag=np.array(tag))
    return names, means, counts


def combine_prototypes(names, means, counts, extra):
    """Add verified crops to the cached class means without recomputing them."""
    idx = {n: i for i, n in enumerate(names)}
    sums = [m * c for m, c in zip(means, counts)]
    cnt = list(counts)
    names = list(names)
    for c, (m, n) in extra.items():
        if c in idx:
            sums[idx[c]] = sums[idx[c]] + m * n
            cnt[idx[c]] += n
        else:
            idx[c] = len(names)
            names.append(c)
            sums.append(m * n)
            cnt.append(n)
    protos = np.stack([cd.l2_normalise(s / c) for s, c in zip(sums, cnt)])
    return names, protos


@st.cache_resource(show_spinner=False)
def get_dictionary(resnet_ckpt, gray_norm, dirs, shots, cache_path, verified_dir, version):
    names, means, counts = get_base(resnet_ckpt, gray_norm, dirs, shots, cache_path, 0)
    extra = {}
    if os.path.isdir(verified_dir) and cd.list_class_files(verified_dir):
        emb = get_embedder(resnet_ckpt, gray_norm)
        extra = _class_means(emb, cd.list_class_files(verified_dir), "Verified crops")
    return combine_prototypes(names, means, counts, extra)


def rank_crops(crops, embedder, names, protos, k=TOPK):
    """For each crop: the k nearest prototypes [(name, similarity)], best similarity and margin."""
    embs = embedder(crops)
    sims = embs @ protos.T
    order = np.argsort(-sims, axis=1)
    out = []
    for row, srow in zip(order, sims):
        top = [(names[j], float(srow[j])) for j in row[:k]]
        s2 = top[1][1] if len(top) > 1 else float("nan")
        out.append({"top": top, "sim": top[0][1], "margin": top[0][1] - s2})
    return out


# ------------------------------------------------------------------------------------------ segmentation
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def segment_plate(img, cfg, ink_threshold=200, dilation=3, max_frac=0.30, min_area=4):
    """Automatic segmentation of a plate into glyph candidates.
    Uses plate_segmentation_triage.py when it is next to the app (ink components, framed-glyph merging, caption
    removal and triage); otherwise a simple version of the same idea. Returns [{x1,y1,x2,y2,flag,reasons}]."""
    import cv2
    gray = np.array(img.convert("L"))
    _, binary = cv2.threshold(gray, ink_threshold, 255, cv2.THRESH_BINARY_INV)
    h, w = gray.shape
    path = resolve(cfg.get("segmenter", ""))
    if path and os.path.isfile(path):
        m = _load_module(path, "plate_segmentation_triage_app")
        boxes = m.connected_component_boxes(binary, dilation, h, w, max_page_frac=max_frac, min_area=min_area)
        boxes = m.containment_merge(boxes)
        labels = m.detect_labels(binary, h, w)
        boxes = m.exclude_labels_from_glyphs(boxes, labels)
        res = m.triage(boxes, labels, h) if boxes else []
        return [{"x1": float(r["box"][0]), "y1": float(r["box"][1]), "x2": float(r["box"][2]), "y2": float(r["box"][3]),
                 "flag": bool(r["flagged"]), "reasons": list(r["reasons"])} for r in res], "plate_segmentation_triage.py"
    kernel = np.ones((dilation, dilation), np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(binary, kernel), connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        if bw > max_frac * w or bh > max_frac * h or a < min_area:
            continue
        boxes.append({"x1": float(x), "y1": float(y), "x2": float(x + bw), "y2": float(y + bh), "flag": False, "reasons": []})
    return boxes, "built-in simple segmentation (plate_segmentation_triage.py was not found)"


# ------------------------------------------------------------------------------------------ LPGS
def load_variator(path):
    path = resolve(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LPGS variator not found: {path}. Put auto_glyph_variator1.py next to the app or set its "
                                "path under Settings.")
    return _load_module(path, "lpgs_variator_app")


def lpgs_prepare(m, size):
    if hasattr(m, "extract_master_trace_info"):
        return lambda p: m.extract_master_trace_info(p, canvas_size=size, fill_fraction=1.0)
    if hasattr(m, "extract_canonical_skeleton"):
        return lambda p: m.extract_canonical_skeleton(p, canvas_size=size, fill_fraction=1.0)
    return lambda p: m.extract_canonical_curves(p, canvas_size=size, fill_fraction=1.0)


def list_seeds(seeds_dir):
    out = {}
    if os.path.isdir(seeds_dir):
        for sign in sorted(os.listdir(seeds_dir)):
            p = os.path.join(seeds_dir, sign)
            if os.path.isdir(p):
                fs = sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(IMG_EXT))
                if fs:
                    out[sign] = fs
    return out


def lpgs_generate(cfg, seeds, signs, n_per_sign, size, seed, degrade, out_dir, progress=None):
    """Make n_per_sign variations for every chosen sign, split over that sign's seed crops. Returns {sign: [paths]}."""
    m = load_variator(cfg["variator"])
    random.seed(seed)
    np.random.seed(seed)
    if not degrade:
        m.choose_channel = lambda: "clean"
        m.apply_environmental_degradation = lambda im, *a, **k: im
    prep = lpgs_prepare(m, size)
    made = {}
    for k, sign in enumerate(signs):
        files = seeds[sign]
        per = [n_per_sign // len(files) + (1 if i < n_per_sign % len(files) else 0) for i in range(len(files))]
        od = os.path.join(out_dir, sign)
        os.makedirs(od, exist_ok=True)
        made[sign] = []
        j = 0
        for f, cnt in zip(files, per):
            info = prep(f)
            for _ in range(cnt):
                p = os.path.join(od, f"{sign}_lpgs_{j:05d}.png")
                m.generate_variation(info, size).save(p)
                made[sign].append(p)
                j += 1
        if progress:
            progress((k + 1) / len(signs), f"{sign}: {len(made[sign])} images")
    return made


def contact_sheet(made, per_row=8, rows=6, cell=64):
    signs = list(made)[:rows]
    if not signs:
        return None
    sheet = Image.new("RGB", (per_row * cell + 70, len(signs) * cell), "white")
    d = ImageDraw.Draw(sheet)
    for r, s in enumerate(signs):
        d.text((4, r * cell + cell // 2 - 5), s, fill="black")
        for c, p in enumerate(made[s][:per_row]):
            im = Image.open(p).convert("RGB").resize((cell - 2, cell - 2))
            sheet.paste(im, (70 + c * cell, r * cell + 1))
    return sheet


# ------------------------------------------------------------------------------------------ Label Studio
LS_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Glyph" background="#2ecc71"/>
    <Label value="NeedsReview" background="#e74c3c"/>
  </RectangleLabels>
</View>"""
LS_PREFIX = "/data/local-files/?d="


def ls_task(name, size, boxes):
    w, h = size
    res = []
    for b in boxes:
        if b["status"] == "skip":
            continue
        res.append({"id": f"box_{b['id']}", "type": "rectanglelabels", "from_name": "label", "to_name": "image",
                    "original_width": w, "original_height": h, "image_rotation": 0,
                    "value": {"x": 100.0 * b["x1"] / w, "y": 100.0 * b["y1"] / h, "width": 100.0 * (b["x2"] - b["x1"]) / w,
                              "height": 100.0 * (b["y2"] - b["y1"]) / h, "rotation": 0,
                              "rectanglelabels": ["NeedsReview" if b.get("flag") else "Glyph"]}})
    return {"data": {"image": LS_PREFIX + name}, "predictions": [{"model_version": "reading_assistant", "result": res}]}


def ls_write(P, out_dir):
    d = os.path.join(out_dir, "labelstudio")
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    img_path = os.path.join(d, "images", P["name"])
    P["img"].save(img_path)
    task = ls_task(P["name"], P["img"].size, P["boxes"])
    stem = os.path.splitext(P["name"])[0]
    tasks_path = os.path.join(d, f"tasks_{stem}.json")
    with open(tasks_path, "w", encoding="utf-8") as f:
        json.dump([task], f, indent=2)
    cfg_path = os.path.join(d, "label_config.xml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(LS_CONFIG)
    return d, tasks_path, cfg_path, len(task["predictions"][0]["result"])


def ls_parse_export(raw, plate_name):
    """Final boxes of the task for this plate from a Label Studio JSON export: [(x1, y1, x2, y2)]."""
    tasks = json.loads(raw)
    if isinstance(tasks, dict):
        tasks = [tasks]
    stem = os.path.splitext(plate_name)[0]
    pick = None
    for t in tasks:
        img = str(t.get("data", {}).get("image", ""))
        fn = os.path.basename(img.split("?d=")[-1])
        if os.path.splitext(fn)[0] == stem and t.get("annotations"):
            pick = t
    if pick is None:
        cands = [t for t in tasks if t.get("annotations")]
        pick = cands[-1] if cands else None
    if pick is None:
        raise ValueError("The export has no annotated task. Submit the annotations in Label Studio and export again.")
    ann = pick["annotations"][-1]
    out = []
    for r in ann.get("result", []):
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        ow, oh = r["original_width"], r["original_height"]
        x1, y1 = v["x"] / 100.0 * ow, v["y"] / 100.0 * oh
        out.append((x1, y1, x1 + v["width"] / 100.0 * ow, y1 + v["height"] / 100.0 * oh))
    return out


# ------------------------------------------------------------------------------------------ plate state
S = st.session_state


def new_plate(name, img, task, raw_boxes):
    """raw_boxes: dicts with x1..y2 and optional det_score, top, sim, margin, flag, reasons."""
    boxes = []
    for i, r in enumerate(raw_boxes, 1):
        boxes.append({"id": f"b{i}", "num": str(i), "x1": float(r["x1"]), "y1": float(r["y1"]), "x2": float(r["x2"]),
                      "y2": float(r["y2"]), "status": "flag" if r.get("flag") else "pending", "flag": bool(r.get("flag")),
                      "reasons": r.get("reasons", []), "det_score": r.get("det_score"), "top": r.get("top", []),
                      "sim": r.get("sim"), "margin": r.get("margin"), "decision": None, "saved": False, "new": False,
                      "orig": (round(float(r["x1"]), 1), round(float(r["y1"]), 1), round(float(r["x2"]), 1), round(float(r["y2"]), 1)),
                      "read_geom": (round(float(r["x1"]), 1), round(float(r["y1"]), 1), round(float(r["x2"]), 1), round(float(r["y2"]), 1))
                      if r.get("top") else None})
    return {"name": name, "img": img, "task": task, "token": uuid.uuid4().hex[:10], "boxes": boxes, "applied_rev": 0,
            "sel": None, "sel_n": 0, "select_id": None, "t_sel": {}, "next_new": 1, "created": time.time(), "removed": 0}


def sync_from_editor(P, out):
    """Take the boxes and selection the user left in the editor. Only fresh edits (higher revision) are applied."""
    if not out or out.get("token") != P["token"] or out.get("rev", 0) <= P["applied_rev"]:
        return False
    P["applied_rev"] = out["rev"]
    by = {b["id"]: b for b in P["boxes"]}
    keep = []
    for ob in out["boxes"]:
        b = by.get(ob["id"])
        if b is None:
            b = {"id": ob["id"], "num": "+", "status": "pending", "flag": False, "reasons": [], "det_score": None, "top": [],
                 "sim": None, "margin": None, "decision": None, "saved": False, "new": True,
                 "orig": None, "read_geom": None}
            P["next_new"] = max(P["next_new"], int(str(ob["id"])[1:] or 0) + 1) if str(ob["id"])[1:].isdigit() else P["next_new"]
        b.update(x1=float(ob["x1"]), y1=float(ob["y1"]), x2=float(ob["x2"]), y2=float(ob["y2"]))
        keep.append(b)
    P["removed"] += len(by) - len({b["id"] for b in keep} & set(by))
    P["boxes"] = keep
    sel = out.get("selected")
    if sel != P["sel"]:
        P["sel"] = sel
        if sel:
            P["t_sel"][sel] = time.time()
    return True


def box_by_id(P, bid):
    for b in P["boxes"]:
        if b["id"] == bid:
            return b
    return None


def next_pending(P, after_id=None, direction=1):
    ids = [b["id"] for b in P["boxes"]]
    if not ids:
        return None
    start = ids.index(after_id) if after_id in ids else (-1 if direction > 0 else 0)
    n = len(ids)
    for step in range(1, n + 1):
        j = (start + direction * step) % n
        if P["boxes"][j]["status"] in ("pending", "flag"):
            return ids[j]
    return None


def select_box(P, bid):
    P["select_id"] = bid
    P["sel_n"] += 1
    P["sel"] = bid
    if bid:
        P["t_sel"][bid] = time.time()


def reread_edited(P, embedder, names, protos, force=False):
    """Read again every unconfirmed box whose rectangle changed since it was last read (or was drawn by hand)."""
    todo = [b for b in P["boxes"] if b["status"] in ("pending", "flag") and (force or moved(b.get("read_geom"), geom(b)))]
    if not todo:
        return 0
    crops = [box_crop(P["img"], b) for b in todo]
    for b, r in zip(todo, rank_crops(crops, embedder, names, protos)):
        b.update(top=r["top"], sim=r["sim"], margin=r["margin"], read_geom=geom(b))
    return len(todo)


def edited_flag(b):
    return b.get("new") or (b.get("orig") is not None and moved(b["orig"], geom(b)))


# ------------------------------------------------------------------------------------------ decisions
def decide(P, bid, final, via, rank):
    b = box_by_id(P, bid)
    if b is None:
        return
    secs = round(time.time() - P["t_sel"].get(bid, time.time()), 1)
    b["decision"] = {"final": final, "via": via, "rank": rank, "seconds": secs}
    b["status"] = "skip" if final == NOT_GLYPH else "done"
    b["saved"] = False
    if S.get("auto_advance", True):
        nxt = next_pending(P, bid)
        if nxt:
            select_box(P, nxt)


def undo_decision(P, bid):
    b = box_by_id(P, bid)
    if b:
        b["decision"] = None
        b["status"] = "flag" if b.get("flag") else "pending"
        b["saved"] = False


def goto_box(P, direction):
    ids = [b["id"] for b in P["boxes"]]
    if not ids:
        return
    cur = ids.index(P["sel"]) if P["sel"] in ids else -1
    select_box(P, ids[(cur + direction) % len(ids)])


# ------------------------------------------------------------------------------------------ saving
def save_decisions(P, cfg, kind, reviewer=""):
    """kind 'verified' (recognition route) or 'seeds' (segmentation route). Writes the crops that are confirmed and not
    yet saved, in the folder of their sign, and appends them to logs/<kind>_log.csv. Returns the number saved."""
    out = cfg["out_dir"]
    root = os.path.join(out, kind)
    logd = os.path.join(out, "logs")
    os.makedirs(root, exist_ok=True)
    os.makedirs(logd, exist_ok=True)
    file_as = cfg.get("file_as", {}) if S.get("file_dispute", True) else {}
    rows, n = [], 0
    now = dt.datetime.now().isoformat(timespec="seconds")
    stem = os.path.splitext(P["name"])[0]
    for b in P["boxes"]:
        d = b.get("decision")
        if not d:
            continue
        top1 = b["top"][0][0] if b.get("top") else ""
        row = {"time": now, "kind": kind, "plate": P["name"], "reviewer": reviewer, "box": b["id"], "x1": round(b["x1"], 1),
               "y1": round(b["y1"], 1), "x2": round(b["x2"], 1), "y2": round(b["y2"], 1), "box_edited": int(bool(edited_flag(b))),
               "box_drawn_by_hand": int(bool(b.get("new"))), "det_score": "" if b.get("det_score") is None else f"{b['det_score']:.3f}",
               "predicted": top1, "sim": "" if b.get("sim") is None else f"{b['sim']:.4f}",
               "margin": "" if b.get("margin") is None else f"{b['margin']:.4f}", "top5": ";".join(t[0] for t in b.get("top", [])),
               "final": d["final"], "rank_in_top5": d["rank"], "correct_Y_N": "Y" if (top1 and d["final"] == top1) else "N",
               "via": d["via"], "seconds": d["seconds"], "saved_as": ""}
        if d["final"] != NOT_GLYPH and not b["saved"]:
            sign = file_as.get(d["final"], d["final"])
            folder = os.path.join(root, sign)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{stem}_{b['id']}_{uuid.uuid4().hex[:6]}.png")
            box_crop(P["img"], b, pad=0.03 if kind == "seeds" else 0.04).save(path)
            row["saved_as"] = path
            b["saved"] = True
            n += 1
        rows.append(row)
    if rows:
        lp = os.path.join(logd, f"{kind}_log.csv")
        new = not os.path.exists(lp)
        with open(lp, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
    if n and kind == "verified":
        get_dictionary.clear()
        S["dict_version"] = S.get("dict_version", 0) + 1
    return n


def zip_folder(folder):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, folder))
    return buf.getvalue()


# ------------------------------------------------------------------------------------------ screens
def settings_for(system_name, systems):
    """The configuration for this session: registry defaults, overridden by the Settings in the sidebar."""
    base = dict(systems[system_name])
    key = base["key"]
    ss = S.setdefault("cfg_" + key, {})
    with st.sidebar.expander("Settings", expanded=False):
        ss["detector_ckpt"] = st.text_input("Detector (best.pth)", ss.get("detector_ckpt", base["detector_ckpt"]), key=f"s_det_{key}")
        ss["embedding_ckpt"] = st.text_input("Embedding network (the 1,400-sign .pth)", ss.get("embedding_ckpt", base["embedding_ckpt"]), key=f"s_emb_{key}")
        ss["dictionary_dirs_txt"] = st.text_area("Dictionary folders (one per line; one sub-folder per sign)",
                                                 ss.get("dictionary_dirs_txt", "\n".join(base["dictionary_dirs"])), height=80, key=f"s_dic_{key}")
        ss["labels_csv"] = st.text_input("Glossary (labels_en.csv)", ss.get("labels_csv", base["labels_csv"]), key=f"s_lab_{key}")
        ss["variator"] = st.text_input("LPGS variator (.py)", ss.get("variator", base["variator"]), key=f"s_var_{key}")
        ss["segmenter"] = st.text_input("Segmentation script (.py)", ss.get("segmenter", base["segmenter"]), key=f"s_seg_{key}")
        ss["out_dir"] = st.text_input("Output folder", ss.get("out_dir", os.path.join(HERE, "app_output", key)), key=f"s_out_{key}")
        ss["reviewer"] = st.text_input("Reviewer", ss.get("reviewer", ""), key=f"s_rev_{key}")
        S["file_dispute"] = st.checkbox("File dispute (S1009) crops under argue (S0201)", S.get("file_dispute", True), key="s_disp")
        S["auto_advance"] = st.checkbox("Move to the next unconfirmed box after each decision", S.get("auto_advance", True), key="s_adv")
    cfg = dict(base)
    cfg.update(detector_ckpt=resolve(ss["detector_ckpt"]), embedding_ckpt=resolve(ss["embedding_ckpt"]),
               dictionary_dirs=[resolve(d.strip()) for d in ss["dictionary_dirs_txt"].splitlines() if d.strip()],
               labels_csv=ss["labels_csv"], variator=ss["variator"], segmenter=ss["segmenter"], out_dir=ss["out_dir"],
               reviewer=ss["reviewer"])
    return cfg


def check(path, what, is_dir=False):
    ok = os.path.isdir(path) if is_dir else os.path.isfile(path)
    return ("✅ " if ok else "⚠️ ") + what + ("" if ok else f" not found: {path}")


def screen_system(systems):
    st.header("1 · Which writing system?")
    name = st.selectbox("Endangered writing system", list(systems), index=0, key="pick_system")
    conf = systems[name]
    st.write(conf.get("blurb", ""))
    key = conf["key"]
    ss = S.get("cfg_" + key, {})
    det = resolve(ss.get("detector_ckpt", conf["detector_ckpt"]))
    emb = resolve(ss.get("embedding_ckpt", conf["embedding_ckpt"]))
    dics = [resolve(d) for d in conf["dictionary_dirs"]]
    with st.expander("What this writing system needs", expanded=True):
        st.write(check(emb, "1,400-sign embedding network"))
        st.write(check(det, "glyph detector"))
        st.write(check(dics[0] if dics else "", "dictionary prototypes (one folder per sign)", is_dir=True))
        st.write(check(resolve(conf["labels_csv"]), "glossary (labels_en.csv)"))
        st.caption("Missing files are only a problem for the route that needs them. Paths can be changed under Settings "
                   "once you continue.")
    if st.button("Continue", type="primary"):
        S["system"] = name
        S["step"] = "task"
        st.rerun()


def screen_task(systems):
    st.header("2 · What do you want to do?")
    st.caption(f"Writing system: **{S['system']}**")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("A · Segment a plate for LPGS")
        st.write("The Label Studio route. Cut a scanned plate into glyph candidates, adjust each box, name the glyph "
                 "(or mark it as a new sign), keep it as a seed, and make LPGS variations of the seeds. Boxes can be "
                 "edited here or sent to Label Studio and brought back.")
        if st.button("Choose A: segmentation and LPGS", type="primary", **STRETCH):
            S["task"], S["step"], S["P"] = "segment", "work", None
            st.rerun()
    with c2:
        st.subheader("B · Detect and recognise")
        st.write("Straight to reading. The detector finds the glyphs, the 1,400-sign network proposes names, and you "
                 "adjust the boxes and confirm each reading. Confirmed crops can be added to the dictionary.")
        if st.button("Choose B: detection and recognition", type="primary", **STRETCH):
            S["task"], S["step"], S["P"] = "recognise", "work", None
            st.rerun()
    if st.button("← Change writing system"):
        S["step"] = "system"
        st.rerun()


def plate_uploader(key):
    up = st.file_uploader("Plate or page image", type=[e.strip(".") for e in IMG_EXT], key=key)
    if up is None:
        return None, None
    return up.name, Image.open(io.BytesIO(up.getvalue())).convert("RGB")


def editor_block(P, cfg, labels, width, height=680):
    """The plate with its editable boxes. Returns nothing; the user's edits are folded into P."""
    boxes = [{"id": b["id"], "x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"], "num": b["num"]} for b in P["boxes"]]
    statuses = {b["id"]: b["status"] for b in P["boxes"]}
    lab = {}
    for b in P["boxes"]:
        if b.get("decision") and b["status"] == "done":
            lab[b["id"]] = gloss(b["decision"]["final"], labels)
        elif b.get("top"):
            lab[b["id"]] = "≈ " + gloss(b["top"][0][0], labels)
    out = box_editor(P["img"], boxes, statuses=statuses, labels=lab, selected=P["select_id"], select_request=P["sel_n"],
                     token=P["token"], width=width, height=height, next_new=P["next_new"], key=f"editor_{P['token']}")
    return out


def crop_preview(P, b, width=200):
    st.image(box_crop(P["img"], b), width=width)
    bits = [f"{int(b['x2'] - b['x1'])} × {int(b['y2'] - b['y1'])} px"]
    if edited_flag(b):
        bits.append("box edited" if not b.get("new") else "box drawn by hand")
    if b.get("reasons"):
        bits.append("triage: " + ", ".join(b["reasons"]))
    st.caption(" · ".join(bits))


def sign_picker(P, b, labels, protos, key, button_label, on_pick):
    """Choose a dictionary sign for box b from the full list."""
    ids = sorted(set(labels) | set(protos))
    opts = ["(choose a sign or type to search)"] + [f"{gloss(i, labels)}" + (f" [{labels[i]['cls']}]" if labels.get(i, {}).get("cls") else "") for i in ids]
    pick = st.selectbox("Another sign from the dictionary", opts, key=key)
    if pick != opts[0]:
        sid = pick.split()[0]
        st.button(f"{button_label} {pick}", key=key + "_go", on_click=on_pick, args=(sid, "list"))


def panel_recognise(P, cfg, labels, protos, by_hand):
    b = box_by_id(P, P["sel"])
    done = sum(1 for x in P["boxes"] if x["status"] in ("done", "skip"))
    st.progress(done / max(1, len(P["boxes"])), text=f"{done} of {len(P['boxes'])} boxes decided")
    if b is None:
        st.info("Click a box on the plate to select it. Drag it to move, drag a handle to resize, or drag on empty space "
                "to draw a box the detector missed. Press Delete to remove a box.")
        return
    st.subheader(f"Box {b['num']}" + ("  ·  " + gloss(b["decision"]["final"], labels) if b.get("decision") else ""))
    crop_preview(P, b)
    c1, c2, c3 = st.columns(3)
    c1.button("← Previous", key=f"prev_{b['id']}", on_click=goto_box, args=(P, -1), **STRETCH)
    c2.button("Next →", key=f"next_{b['id']}", on_click=goto_box, args=(P, 1), **STRETCH)
    if c3.button("Re-read box", key=f"rr_{b['id']}", help="Read this box again after you resized it.", **STRETCH):
        S["force_reread"] = b["id"]
    if b.get("decision"):
        st.success(f"Confirmed: {gloss(b['decision']['final'], labels)}" if b["status"] == "done" else "Marked: not a glyph")
        st.button("Undo this decision", key=f"und_{b['id']}", on_click=undo_decision, args=(P, b["id"]))
        return
    if not by_hand:
        if b.get("top"):
            st.caption(f"Nearest signs. Best similarity {b['sim']:.2f}, margin over the second {b['margin']:.3f}. "
                       "Similarity and margin do not tell right from wrong at 1,400 classes: check each name.")
            for rank, (name, sim) in enumerate(b["top"], 1):
                partner = ""
                for a, c in cfg.get("confusable", []):
                    if name == a:
                        partner = f"  (almost identical to {gloss(c, labels)})"
                    elif name == c:
                        partner = f"  (almost identical to {gloss(a, labels)})"
                cols = st.columns([1, 6])
                if name in protos:
                    cols[0].image(protos[name], width=44)
                cols[1].button(f"{rank}. {gloss(name, labels)}   ({sim:.2f}){partner}", key=f"top_{b['id']}_{rank}_{P['sel_n']}",
                               type="primary" if rank == 1 else "secondary", on_click=decide, args=(P, b["id"], name, "suggestion", rank), **STRETCH)
        else:
            st.info("This box has not been read yet. Press “Re-read box”.")

    def pick(sid, via):
        rank = next((k for k, (nm, _) in enumerate(b.get("top", []), 1) if nm == sid), 0)
        decide(P, b["id"], sid, via, rank)

    sign_picker(P, b, labels, protos, f"pick_{b['id']}", "Use", pick)
    d1, d2 = st.columns(2)
    d1.button("Not a glyph", key=f"ng_{b['id']}", on_click=decide, args=(P, b["id"], NOT_GLYPH, "not_a_glyph", 0), **STRETCH)
    with st.expander("A sign that is not in the dictionary"):
        newname = st.text_input("Name of the new sign", key=f"nn_{b['id']}").strip().replace(" ", "_")
        st.button("Add as a new sign", key=f"new_{b['id']}", disabled=not newname, on_click=decide,
                  args=(P, b["id"], newname, "new_sign", 0))


def panel_segment(P, cfg, labels, protos, suggest):
    b = box_by_id(P, P["sel"])
    done = sum(1 for x in P["boxes"] if x["status"] in ("done", "skip"))
    st.progress(done / max(1, len(P["boxes"])), text=f"{done} of {len(P['boxes'])} boxes decided")
    if b is None:
        st.info("Click a box to select it. Adjust it so it holds exactly one glyph (resize with the handles, draw a new box "
                "to split a merged glyph, Delete to remove a caption or a mark), then name it.")
        return
    st.subheader(f"Box {b['num']}" + ("  ·  " + gloss(b["decision"]["final"], labels) if b.get("decision") else ""))
    crop_preview(P, b)
    c1, c2 = st.columns(2)
    c1.button("← Previous", key=f"sprev_{b['id']}", on_click=goto_box, args=(P, -1), **STRETCH)
    c2.button("Next →", key=f"snext_{b['id']}", on_click=goto_box, args=(P, 1), **STRETCH)
    if b.get("decision"):
        st.success(f"Seed for {gloss(b['decision']['final'], labels)}" if b["status"] == "done" else "Marked: not a glyph")
        st.button("Undo this decision", key=f"sund_{b['id']}", on_click=undo_decision, args=(P, b["id"]))
        return
    if suggest and b.get("top"):
        st.caption("Suggestions from the 1,400-sign network (check each):")
        for rank, (name, sim) in enumerate(b["top"][:3], 1):
            st.button(f"{rank}. Seed for {gloss(name, labels)}  ({sim:.2f})", key=f"stop_{b['id']}_{rank}_{P['sel_n']}",
                      on_click=decide, args=(P, b["id"], name, "suggestion", rank), **STRETCH)

    def pick(sid, via):
        decide(P, b["id"], sid, via, 0)

    sign_picker(P, b, labels, protos, f"spick_{b['id']}", "Keep as a seed for", pick)
    with st.expander("A sign that is not in the dictionary", expanded=not labels):
        newname = st.text_input("Name of the new sign", key=f"snn_{b['id']}").strip().replace(" ", "_")
        st.button("Keep as a seed for this new sign", key=f"snew_{b['id']}", disabled=not newname, on_click=decide,
                  args=(P, b["id"], newname, "new_sign", 0))
    st.button("Skip: not a glyph", key=f"sng_{b['id']}", on_click=decide, args=(P, b["id"], NOT_GLYPH, "not_a_glyph", 0), **STRETCH)


def plate_legend(P, labels, protos):
    ids = []
    for b in P["boxes"]:
        if b.get("decision") and b["decision"]["final"] != NOT_GLYPH:
            ids.append(b["decision"]["final"])
        elif b.get("top"):
            ids.append(b["top"][0][0])
    ids = sorted(set(ids))
    if ids:
        with st.expander(f"Legend for this plate ({len(ids)} signs named or suggested)"):
            show_legend_table(legend_rows(ids, labels, protos), height=min(420, 60 + 38 * len(ids)))


# ---- route B
def work_recognise(cfg):
    labels = load_labels(cfg["labels_csv"])
    protos = proto_index(tuple(cfg["dictionary_dirs"]))
    out_dir = cfg["out_dir"]
    c1, c2 = st.columns([3, 2])
    with c1:
        name, img = plate_uploader("up_rec")
    with c2:
        mode = st.radio("Mode", ["With suggestions", "By hand (no suggestions)"], horizontal=True, key="mode_rec",
                        help="'By hand' hides the nearest signs; use it for the by-hand part of the timing test.")
        thr = st.slider("Detector confidence", 0.05, 0.95, 0.30, 0.05, key="thr_rec")
        min_box = st.number_input("Smallest box (px)", 4, 100, 10, key="min_rec")
    by_hand = mode.startswith("By hand")
    cache = os.path.join(out_dir, "protos_1400_app.npz")
    verified = os.path.join(out_dir, "verified")
    if img is not None and st.button("Find and read the glyphs", type="primary", key="go_rec"):
        try:
            names, protos_m = get_dictionary(cfg["embedding_ckpt"], True, tuple(cfg["dictionary_dirs"]), 1, cache, verified,
                                             S.get("dict_version", 0))
            with st.spinner("Detecting ..."):
                dets = cd.detect(img, cfg["detector_ckpt"], thr, 0.4, int(min_box))
            raw = [{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "det_score": float(s)} for (x1, y1, x2, y2), s in dets]
            P = new_plate(name, img, "recognise", raw)
            if P["boxes"] and not by_hand:
                reread_edited(P, get_embedder(cfg["embedding_ckpt"], True), names, protos_m, force=True)
            elif P["boxes"]:
                for b in P["boxes"]:
                    b["read_geom"] = geom(b)
            S["P"] = P
        except Exception as ex:
            st.error(f"{type(ex).__name__}: {ex}")
            st.stop()
    P = S.get("P")
    if not P or P["task"] != "recognise":
        st.info("Choose a plate, then press “Find and read the glyphs”.")
        return
    width = st.slider("Plate zoom (width in pixels)", 700, 2400, 1150, 50, key="zoom_rec")
    left, right = st.columns([3, 2])
    with left:
        out = editor_block(P, cfg, labels, width)
    changed = sync_from_editor(P, out)
    if not by_hand:
        try:
            names, protos_m = get_dictionary(cfg["embedding_ckpt"], True, tuple(cfg["dictionary_dirs"]), 1, cache, verified, S.get("dict_version", 0))
            emb = get_embedder(cfg["embedding_ckpt"], True)
            fr = S.pop("force_reread", None)
            if fr and box_by_id(P, fr):
                box_by_id(P, fr)["read_geom"] = None
            n = reread_edited(P, emb, names, protos_m)
            if n:
                st.toast(f"Read {n} edited box(es) again")
        except Exception as ex:
            st.error(f"{type(ex).__name__}: {ex}")
    with right:
        panel_recognise(P, cfg, labels, protos, by_hand)
    decided = sum(1 for x in P["boxes"] if x["decision"])
    s1, s2, s3 = st.columns([2, 2, 3])
    if s1.button("Save confirmed glyphs", type="primary", disabled=not decided, key="save_rec"):
        n = save_decisions(P, cfg, "verified", cfg.get("reviewer", ""))
        st.success(f"Saved {n} crops to {verified} and the log to {os.path.join(out_dir, 'logs', 'verified_log.csv')}. "
                   "They join the dictionary the next time a plate is read.")
    secs = [x["decision"]["seconds"] for x in P["boxes"] if x["decision"]]
    if secs:
        s2.metric("Median seconds per decision", f"{statistics.median(secs):.1f}")
    lp = os.path.join(out_dir, "logs", "verified_log.csv")
    if os.path.exists(lp):
        with open(lp, "rb") as f:
            s3.download_button("Download the decision log (CSV)", f.read(), file_name="verified_log.csv", mime="text/csv")
    plate_legend(P, labels, protos)
    retrain_panel(os.path.join(out_dir, "verified"))


# ---- route A
def work_segment(cfg):
    labels = load_labels(cfg["labels_csv"])
    protos = proto_index(tuple(cfg["dictionary_dirs"]))
    out_dir = cfg["out_dir"]
    seeds_dir = os.path.join(out_dir, "seeds")
    t1, t2, t3 = st.tabs(["1 · Segment and label", "2 · LPGS variations", "Label Studio (optional)"])
    with t1:
        c1, c2 = st.columns([3, 2])
        with c1:
            name, img = plate_uploader("up_seg")
        with c2:
            with st.expander("Segmentation settings"):
                ink = st.slider("Ink threshold (darker than this is ink)", 80, 250, 200, 5, key="seg_ink")
                dil = st.slider("Dilation (raise to join broken strokes, lower to separate neighbours)", 1, 9, 3, key="seg_dil")
                frac = st.slider("Ignore blobs wider or taller than this share of the page", 0.05, 0.9, 0.30, 0.05, key="seg_frac")
                area = st.number_input("Smallest component (pixels)", 1, 200, 4, key="seg_area")
            suggest = st.checkbox("Suggest names with the 1,400-sign network", False, key="seg_suggest",
                                  help="Needs the network and the dictionary prototypes; otherwise you pick each name yourself.")
        if img is not None and st.button("Segment this plate", type="primary", key="go_seg"):
            try:
                with st.spinner("Segmenting ..."):
                    raw, how = segment_plate(img, cfg, ink, dil, frac, area)
                P = new_plate(name, img, "segment", raw)
                S["seg_how"] = how
                if suggest and P["boxes"]:
                    names, protos_m = get_dictionary(cfg["embedding_ckpt"], True, tuple(cfg["dictionary_dirs"]), 1,
                                                     os.path.join(out_dir, "protos_1400_app.npz"), os.path.join(out_dir, "verified"), S.get("dict_version", 0))
                    reread_edited(P, get_embedder(cfg["embedding_ckpt"], True), names, protos_m, force=True)
                S["P"] = P
            except Exception as ex:
                st.error(f"{type(ex).__name__}: {ex}")
                st.stop()
        P = S.get("P")
        if not P or P["task"] != "segment":
            st.info("Choose a plate, then press “Segment this plate”.")
        else:
            st.caption(f"{len(P['boxes'])} candidate boxes from {S.get('seg_how', 'segmentation')}. Red boxes were flagged by "
                       "triage (size or shape unlike the rest of the page, usually several glyphs merged).")
            width = st.slider("Plate zoom (width in pixels)", 700, 2400, 1150, 50, key="zoom_seg")
            left, right = st.columns([3, 2])
            with left:
                out = editor_block(P, cfg, labels, width)
            sync_from_editor(P, out)
            if suggest:
                try:
                    names, protos_m = get_dictionary(cfg["embedding_ckpt"], True, tuple(cfg["dictionary_dirs"]), 1,
                                                     os.path.join(out_dir, "protos_1400_app.npz"), os.path.join(out_dir, "verified"), S.get("dict_version", 0))
                    reread_edited(P, get_embedder(cfg["embedding_ckpt"], True), names, protos_m)
                except Exception as ex:
                    st.error(f"{type(ex).__name__}: {ex}")
            with right:
                panel_segment(P, cfg, labels, protos, suggest)
            decided = sum(1 for x in P["boxes"] if x["decision"])
            if st.button("Save the seeds", type="primary", disabled=not decided, key="save_seg"):
                n = save_decisions(P, cfg, "seeds", cfg.get("reviewer", ""))
                st.success(f"Saved {n} seed crops under {seeds_dir}. Go to “2 · LPGS variations”.")
            plate_legend(P, labels, protos)
    with t2:
        seeds = list_seeds(seeds_dir)
        if not seeds:
            st.info(f"No seeds yet. Confirm glyphs in step 1 and press “Save the seeds” (they are stored in {seeds_dir}).")
        else:
            st.write(f"{len(seeds)} signs have seeds ({sum(len(v) for v in seeds.values())} crops).")
            rows = [{"Number": s, "Meaning": labels.get(s, {}).get("en", ""), "Seed crops": len(v)} for s, v in seeds.items()]
            st.dataframe(rows, hide_index=True, height=min(300, 40 + 35 * len(rows)), **STRETCH)
            signs = st.multiselect("Signs to generate", list(seeds), default=list(seeds), key="lp_signs")
            c1, c2, c3, c4 = st.columns(4)
            n_per = c1.number_input("Variations per sign", 5, 1000, 50, 5, key="lp_n")
            size = c2.number_input("Image size (px)", 32, 512, 100, 4, key="lp_size")
            seed = c3.number_input("Random seed", 0, 10_000, 0, key="lp_seed")
            degrade = c4.checkbox("Degradation (ink, grain, blur, lighting)", True, key="lp_deg")
            if st.button("Generate LPGS variations", type="primary", disabled=not signs, key="lp_go"):
                bar = st.progress(0.0, text="Generating ...")
                try:
                    made = lpgs_generate(cfg, seeds, signs, int(n_per), int(size), int(seed), degrade, os.path.join(out_dir, "lpgs_out"),
                                         progress=lambda f, t: bar.progress(f, text=t))
                    S["lp_made"] = made
                    bar.empty()
                    st.success(f"Made {sum(len(v) for v in made.values())} images in {os.path.join(out_dir, 'lpgs_out')}. "
                               "Check the sheet; if the variations of a sign look wrong, correct its seed and generate again.")
                except Exception as ex:
                    bar.empty()
                    st.error(f"{type(ex).__name__}: {ex}")
            made = S.get("lp_made")
            if made:
                sheet = contact_sheet(made)
                if sheet is not None:
                    st.image(sheet, caption="First variations of the first signs (rows are signs)")
                st.download_button("Download the variations (zip)", zip_folder(os.path.join(out_dir, "lpgs_out")), file_name="lpgs_out.zip",
                                   mime="application/zip", key="lp_zip")
    with t3:
        st.write("Use Label Studio for the box review if you prefer it. The boxes you have now become pre-drawn "
                 "rectangles on the plate; what you leave in Label Studio comes back here as the final boxes.")
        P = S.get("P")
        if not P or P["task"] != "segment":
            st.info("Segment a plate first (step 1).")
        else:
            if st.button("Prepare files for Label Studio", key="ls_prep"):
                d, tasks_path, cfg_path, n = ls_write(P, out_dir)
                S["ls_paths"] = (d, tasks_path, cfg_path, n)
            if S.get("ls_paths"):
                d, tasks_path, cfg_path, n = S["ls_paths"]
                st.success(f"Wrote {n} pre-drawn boxes and the plate image to {d}")
                st.markdown(
                    "1. Start Label Studio with `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` and "
                    f"`LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={os.path.join(d, 'images')}`.\n"
                    "2. Create a project, paste `label_config.xml` under Settings → Labeling Interface → Code, then import "
                    "`tasks_<plate>.json`.\n3. Correct the boxes, submit, and export as JSON. Upload the export below.")
                for p, mime in ((tasks_path, "application/json"), (cfg_path, "application/xml")):
                    with open(p, "rb") as f:
                        st.download_button(f"Download {os.path.basename(p)}", f.read(), file_name=os.path.basename(p), mime=mime, key="dl_" + os.path.basename(p))
            if S.get("ls_msg"):
                st.success(S.pop("ls_msg"))
            exp = st.file_uploader("Label Studio export (JSON)", type=["json"], key="ls_up")
            if exp is not None and st.button("Use these boxes", key="ls_use"):
                try:
                    bxs = ls_parse_export(exp.getvalue().decode("utf-8"), P["name"])
                    raw = [{"x1": a, "y1": b, "x2": c, "y2": d_} for a, b, c, d_ in bxs]
                    S["P"] = new_plate(P["name"], P["img"], "segment", raw)
                    S["ls_msg"] = f"{len(raw)} boxes loaded from Label Studio. Open the tab “1 · Segment and label” to name them."
                    st.rerun()
                except Exception as ex:
                    st.error(f"{type(ex).__name__}: {ex}")


# ------------------------------------------------------------------------------------------ optional retraining (as before)
def _run_retrain(args):
    script = os.path.join(HERE, "retrain_new_signs.py")
    for need in ("retrain_new_signs.py", "run_lpgs_arm.py", "finetune_resnet.py"):
        if not os.path.isfile(os.path.join(HERE, need)):
            st.error(f"{need} must be in the same folder as this application.")
            return False
    box, lines = st.empty(), []
    proc = subprocess.Popen([sys.executable, script] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        if "Warning" in line or "warnings.warn" in line:
            continue
        lines.append(line.rstrip())
        box.code("\n".join(lines[-25:]))
    return proc.wait() == 0


def retrain_panel(verified_dir):
    with st.expander("Optional: retrain the network with variations of the new signs"):
        st.warning("Experiment E6 (a 20-sign pilot) found no measurable gain from retraining (+1.6, 0 and +0.9 points at 1, 2 "
                   "and 5 shots), and it takes about 10 to 30 minutes on a laptop CPU. It has not been tested with the "
                   "1,400-sign network. Adding verified crops to the prototypes, which this application already does, needs "
                   "no retraining. A NEW file is written; your networks are never changed.")
        c1, c2 = st.columns(2)
        rt_module = c1.text_input("LPGS variator (.py)", "auto_glyph_variator1.py", key="rt_mod")
        rt_synth = c1.text_area("Existing synthetic corpora (one folder per line)", height=70, key="rt_syn")
        rt_proto = c1.text_input("Clean prototype folder for new signs (optional; <Sign>/<drawing>)", "", key="rt_proto")
        rt_start = c2.text_input("Start from this Adinkra network", "adinkra_resnet18.pth", key="rt_start")
        rt_out = c2.text_input("Write the new network to", "retrained_new_signs.pth", key="rt_out")
        rt_real = c2.text_input("Real training crops to add (optional; E7 recipe)", "", key="rt_real")
        rt_n = c1.number_input("LPGS images per new sign", 20, 500, 100, 10, key="rt_n")
        rt_ep = c2.number_input("Epochs", 1, 10, 3, key="rt_ep")
        base = ["--module", rt_module, "--synth-dirs"] + [d.strip() for d in rt_synth.splitlines() if d.strip()] + \
               ["--verified-dir", verified_dir, "--resnet-ckpt", rt_start, "--out", rt_out,
                "--per-class", str(int(rt_n)), "--epochs", str(int(rt_ep))]
        if rt_proto.strip():
            base += ["--prototype-dir", rt_proto.strip()]
        if rt_real.strip():
            base += ["--with-real", rt_real.strip()]
        ready = bool(rt_synth.strip())
        b1, b2 = st.columns(2)
        if b1.button("Show which signs are new", disabled=not ready, key="rt_dry"):
            _run_retrain(base + ["--dry-run"])
        if b2.button("Generate variations and retrain", disabled=not ready, key="rt_go"):
            if _run_retrain(base):
                st.success(f"Done. To use it, put '{rt_out}' in Settings as the embedding network.")
            else:
                st.error("Retraining stopped; see the messages above.")


# ------------------------------------------------------------------------------------------ main
def main():
    st.set_page_config(page_title="Reading assistant for endangered scripts", layout="wide")
    systems = load_systems()
    S.setdefault("step", "system")
    st.title("Reading assistant for endangered scripts")
    steps = {"system": 1, "task": 2, "work": 3}
    crumb = [("1 Writing system", S.get("system") if S["step"] != "system" else None),
             ("2 Task", {"segment": "A · Segment for LPGS", "recognise": "B · Detect and recognise"}.get(S.get("task")) if S["step"] == "work" else None),
             ("3 Work", None)]
    st.caption("  ›  ".join(f"**{a}**" + (f": {b}" if b else "") if i + 1 == steps[S["step"]] else a + (f": {b}" if b else "")
                            for i, (a, b) in enumerate(crumb)))
    if S["step"] == "system":
        screen_system(systems)
        return
    if S.get("system") not in systems:
        S["step"] = "system"
        st.rerun()
    cfg = settings_for(S["system"], systems)
    with st.sidebar:
        st.markdown(f"**{S['system']}**")
        if S["step"] == "work":
            st.markdown("Task: " + ("segmentation and LPGS" if S["task"] == "segment" else "detection and recognition"))
            if st.button("← Change task", **STRETCH):
                S["step"], S["P"] = "task", None
                st.rerun()
        if st.button("← Change writing system", **STRETCH):
            S["step"], S["P"] = "system", None
            st.rerun()
    if S["step"] == "task":
        screen_task(systems)
        return
    tw, tl = st.tabs(["Work", "Dictionary legend"])
    with tw:
        if S["task"] == "segment":
            work_segment(cfg)
        else:
            work_recognise(cfg)
    with tl:
        legend_tab(cfg)


if __name__ == "__main__":
    main()
