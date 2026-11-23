# Three-lane version of the review application (auto / review / unknown), as used for experiments E5, E9 and E14.
# The current app is src/nsibidi_app.py. To run this file, copy it next to classify_detections.py and auto_glyph_variator1.py.
"""
Nsibidi reading assistant -- a small review app (Streamlit, runs locally on a CPU).

    pip install streamlit
    streamlit run nsibidi_app.py

Put this file next to classify_detections.py.  The app

  1. finds glyphs on an uploaded plate or page (the single-class Faster R-CNN detector),
  2. names each one with the prototype classifier, and sorts every box into one of three lanes:
        auto     similarity >= auto threshold         (accepted unless you change it)
        review   review threshold <= similarity < auto (a person decides)
        unknown  similarity < review threshold         (a person decides; may be a new sign)
  3. lets the reviewer accept, relabel, mark "not a glyph", or name a NEW sign;
  4. saves every accepted crop into the 'verified' folder and logs every decision;
  5. rebuilds the dictionary from real crops + prototypes + verified crops -- no retraining;
  6. OPTIONAL (bottom of the page): makes LPGS variations of the new signs and retrains the embedding
     network into a NEW file (retrain_new_signs.py, which must sit next to this file together with
     run_lpgs_arm.py and finetune_resnet.py). Experiment E6 found no measurable gain from this step.

The three-lane thresholds default to the values used in the thesis (0.85 and 0.75).
"""
import csv
import datetime as dt
import io
import os
import subprocess
import sys
import uuid

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_detections as cd                      # noqa: E402

LANE_COLOUR = {"auto": "#1baf7a", "review": "#e0a100", "unknown": "#d6453a"}
NOT_GLYPH = "(not a glyph)"
NEW_SIGN = "(new sign ...)"


# ----------------------------------------------------------------------------- pure logic
def assign_lane(sim, auto_thr, review_thr):
    if sim >= auto_thr:
        return "auto"
    if sim >= review_thr:
        return "review"
    return "unknown"


def crop_box(img, r, pad=0.04):
    w, h = img.size
    px, py = (r["x2"] - r["x1"]) * pad, (r["y2"] - r["y1"]) * pad
    return img.crop((max(0, int(r["x1"] - px)), max(0, int(r["y1"] - py)),
                     min(w, int(r["x2"] + px)), min(h, int(r["y2"] + py))))


def annotate(img, rows):
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    lw = max(2, im.size[0] // 400)
    for i, r in enumerate(rows):
        c = LANE_COLOUR[r["lane"]]
        d.rectangle([r["x1"], r["y1"], r["x2"], r["y2"]], outline=c, width=lw)
        d.rectangle([r["x1"], r["y1"], r["x1"] + 12 + 9 * len(str(i + 1)), r["y1"] + 16], fill=c)
        d.text((r["x1"] + 3, r["y1"] + 2), str(i + 1), fill="white")
    return im


def log_decisions(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            wr.writeheader()
        wr.writerows(rows)


# ----------------------------------------------------------------------------- cached models
@st.cache_resource(show_spinner="Loading the embedding network ...")
def get_embedder(resnet_ckpt, gray_norm):
    return cd.Embedder("resnet", resnet_ckpt=resnet_ckpt, gray_norm=gray_norm)


@st.cache_resource(show_spinner="Building the dictionary ...")
def get_dictionary(resnet_ckpt, gray_norm, dirs, shots, version):
    emb = get_embedder(resnet_ckpt, gray_norm)
    good = [d for d in dirs if os.path.isdir(d) and cd.list_class_files(d)]
    bad = [d for d in dirs if d not in good and os.path.basename(os.path.normpath(d)) != "verified"]
    if not good:
        raise ValueError("No usable dictionary folder. Each folder must exist and contain one sub-folder per sign "
                         "with images in it. Not usable: " + "; ".join(os.path.abspath(d) for d in bad) +
                         ". Use full paths in the sidebar, for example <ROOT>\\dataset\\real_split\\train")
    files = cd.sample_support(good, shots, seed=42)
    return cd.build_prototypes(emb, files)


# ----------------------------------------------------------------------------- app
st.set_page_config(page_title="Nsibidi reading assistant", layout="wide")
st.title("Nsibidi reading assistant")
S = st.session_state
S.setdefault("dict_version", 0)
S.setdefault("results", None)

with st.sidebar:
    st.header("Models and dictionary")
    det_ckpt = st.text_input("Detector checkpoint (best.pth)", cd.DETECTOR_CKPT)
    res_ckpt = st.text_input("Embedding network (.pth)", "e7_clean_s0.pth")
    gray = st.checkbox("Grey-scale polarity view", True)
    dirs_txt = st.text_area("Dictionary folders (one per line; each holds one sub-folder per sign)",
                            r"<ROOT>\dataset\real_split\train", height=90)
    verified_dir = st.text_input("Folder for verified crops", "verified")
    shots = st.number_input("Crops per sign per folder (0 = all)", 0, 50, 5)
    st.header("Thresholds")
    auto_thr = st.slider("Auto lane: similarity at least", 0.5, 1.0, 0.85, 0.01)
    review_thr = st.slider("Review lane: similarity at least", 0.3, 1.0, 0.75, 0.01)
    det_thr = st.slider("Detector confidence", 0.05, 0.95, 0.30, 0.05)
    min_box = st.number_input("Smallest box (px)", 4, 100, 10)

dict_dirs = tuple([d.strip() for d in dirs_txt.splitlines() if d.strip()] + [verified_dir])

up = st.file_uploader("Plate or page image", type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"])
if up is not None and st.button("Find and name the glyphs", type="primary"):
    img = Image.open(io.BytesIO(up.getvalue())).convert("RGB")
    try:
        names, protos = get_dictionary(res_ckpt, gray, dict_dirs, int(shots), S.dict_version)
        with st.spinner("Detecting ..."):
            dets = cd.detect(img, det_ckpt, det_thr, 0.4, int(min_box))
        rows = cd.classify_boxes(img, dets, get_embedder(res_ckpt, gray), names, protos, 0.04, 0.0)
    except Exception as ex:                                     # show the real error to the user
        st.error(f"{type(ex).__name__}: {ex}")
        st.stop()
    for r in rows:
        r["lane"] = assign_lane(r["sim"], auto_thr, review_thr)
    S.results, S.image, S.source = rows, img, up.name
    S.names = names

if S.results is not None:
    rows, img = S.results, S.image
    for r in rows:                                               # thresholds can be moved afterwards
        r["lane"] = assign_lane(r["sim"], auto_thr, review_thr)
    n = {k: sum(r["lane"] == k for r in rows) for k in LANE_COLOUR}
    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(annotate(img, rows), caption=f"{len(rows)} boxes: {n['auto']} auto, "
                 f"{n['review']} review, {n['unknown']} unknown", use_container_width=True)
    with c2:
        st.subheader("Review queue")
        st.caption("Green boxes are accepted as named unless you change them. Amber and red need a decision.")
        decisions = {}
        options = [NOT_GLYPH] + list(S.names) + [NEW_SIGN]
        for i, r in enumerate(rows):
            if r["lane"] == "auto" and not st.session_state.get("show_auto", False):
                decisions[i] = (r["best_label"], None)
                continue
            with st.container(border=True):
                a, b = st.columns([1, 2])
                a.image(crop_box(img, r), width=110)
                b.markdown(f"**#{i+1}** &nbsp; :{ {'auto': 'green', 'review': 'orange', 'unknown': 'red'}[r['lane']] }[{r['lane']}]  \n"
                           f"{r['best_label']} ({r['sim']:.2f}) &nbsp;|&nbsp; next: {r['second_label']} ({r['second_sim']:.2f})")
                default = options.index(r["best_label"]) if r["lane"] != "unknown" else 0
                choice = b.selectbox("Decision", options, index=default, key=f"dec{i}", label_visibility="collapsed")
                newname = None
                if choice == NEW_SIGN:
                    newname = b.text_input("Name of the new sign", key=f"new{i}").strip().replace(" ", "_")
                decisions[i] = (choice, newname)
        st.checkbox("Also list the auto-lane boxes", key="show_auto")

        if st.button("Save decisions and grow the dictionary", type="primary"):
            os.makedirs(verified_dir, exist_ok=True)
            log, saved = [], 0
            now = dt.datetime.now().isoformat(timespec="seconds")
            for i, r in enumerate(rows):
                choice, newname = decisions.get(i, (r["best_label"], None))
                final = newname if choice == NEW_SIGN else choice
                if choice == NEW_SIGN and not final:
                    st.warning(f"Box #{i+1}: give the new sign a name, or choose another decision.")
                    st.stop()
                if final and final != NOT_GLYPH:
                    d = os.path.join(verified_dir, final)
                    os.makedirs(d, exist_ok=True)
                    crop_box(img, r).save(os.path.join(d, f"{uuid.uuid4().hex[:10]}.png"))
                    saved += 1
                log.append({"time": now, "source": S.source, "box": i + 1, "x1": int(r["x1"]), "y1": int(r["y1"]),
                            "x2": int(r["x2"]), "y2": int(r["y2"]), "lane": r["lane"], "predicted": r["best_label"],
                            "sim": f"{r['sim']:.4f}", "margin": f"{r['sim'] - r['second_sim']:.4f}",
                            "final": final or "", "changed": int(final != r["best_label"])})
            log_decisions(os.path.join(verified_dir, "decisions_log.csv"), log)
            S.dict_version += 1
            S.results = None
            st.success(f"Saved {saved} verified crops and the decision log to '{verified_dir}'. "
                       f"The dictionary will include them next time you run a page.")
            st.rerun()

    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["box", "lane", "predicted", "sim", "second", "second_sim", "det_score", "x1", "y1", "x2", "y2"])
    for i, r in enumerate(rows):
        wr.writerow([i + 1, r["lane"], r["best_label"], f"{r['sim']:.4f}", r["second_label"], f"{r['second_sim']:.4f}",
                     f"{r['det_score']:.3f}", int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])])
    st.download_button("Download the table (CSV)", buf.getvalue(), file_name="boxes.csv", mime="text/csv")


# ----------------------------------------------------------------------------- optional retraining
def _run_retrain(args):
    """Runs retrain_new_signs.py as a separate process and shows its output as it appears."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrain_new_signs.py")
    for need in ("retrain_new_signs.py", "run_lpgs_arm.py", "finetune_resnet.py"):
        if not os.path.isfile(os.path.join(os.path.dirname(script), need)):
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


st.divider()
with st.expander("Optional: retrain the network with variations of the new signs"):
    st.warning("Experiment E6 found no measurable gain from retraining (+1.6, 0 and +0.9 points at 1, 2 and 5 shots), "
               "and it takes about 10 to 30 minutes on a laptop CPU. Adding the prototype and a few verified crops, "
               "which this application already does, was just as good. Use this only to test whether retraining "
               "helps for your signs. Your existing networks are never changed: a NEW file is written.")
    c1, c2 = st.columns(2)
    rt_module = c1.text_input("LPGS variator (.py)", "auto_glyph_variator1.py")
    rt_synth = c1.text_area("Existing synthetic corpora (one folder per line)", height=70)
    rt_proto = c1.text_input("Clean prototype folder for new signs (optional; <Sign>/<drawing>)", "")
    rt_start = c2.text_input("Start from this Adinkra network", "adinkra_resnet18.pth")
    rt_out = c2.text_input("Write the new network to", "retrained_new_signs.pth")
    rt_real = c2.text_input("Real training crops to add (optional; E7 recipe)", "")
    rt_n = c1.number_input("LPGS images per new sign", 20, 500, 100, 10)
    rt_ep = c2.number_input("Epochs", 1, 10, 3)
    base = ["--module", rt_module, "--synth-dirs"] + [d.strip() for d in rt_synth.splitlines() if d.strip()] + \
           ["--verified-dir", verified_dir, "--resnet-ckpt", rt_start, "--out", rt_out,
            "--per-class", str(int(rt_n)), "--epochs", str(int(rt_ep))]
    if rt_proto.strip():
        base += ["--prototype-dir", rt_proto.strip()]
    if rt_real.strip():
        base += ["--with-real", rt_real.strip()]
    ready = bool(rt_synth.strip())
    if not ready:
        st.caption("Enter the existing synthetic corpus folder(s) first (the folder that holds one sub-folder per sign).")
    b1, b2 = st.columns(2)
    if b1.button("Show which signs are new", disabled=not ready):
        _run_retrain(base + ["--dry-run"])
    if b2.button("Generate variations and retrain", disabled=not ready):
        if _run_retrain(base):
            st.success(f"Done. To use it, put '{rt_out}' in the 'Embedding network' box at the left, then check the lanes "
                       "again: the thresholds were set for the earlier networks.")
        else:
            st.error("Retraining stopped; see the messages above.")
