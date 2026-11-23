# Nsibidi glyph pipeline

Code for the experiments in the thesis *A Data-Efficient Approach for Accelerated Development of Benchmark Character Recognition Dataset for Endangered Scripts* (Ajoku, K. K., Federal University of Technology Owerri, 2026).

The pipeline turns a handful of dictionary glyphs and a few historical plates into a recogniser that detects glyphs on a page, names the ones it knows, abstains on the rest, and lets a human verify and grow the vocabulary. Nsibidi is the working script; the Adinkra symbol set is used to pretrain the embedding network.

## Repository layout

```
nsibidi-glyph-pipeline/
├── src/          runnable scripts (pipeline stages, detector, classifier, experiments, review app)
├── cut/          Stage 4: CUT style-refinement network (train and generate)
├── legacy/       earlier three-lane version of the review app (used for E5, E9 and E14)
├── notebooks/    Adinkra pretraining of the ResNet-18 (Google Colab cell)
├── docs/         experiment_map.md: which script runs which experiment
├── requirements.txt
└── CITATION.cff
```

`src/` is kept flat on purpose: several scripts import each other by name (for example `classify_detections.py`), and the review app expects `box_editor.py`, `classify_detections.py`, `plate_segmentation_triage.py`, `auto_glyph_variator1.py` and `labels_en.csv` in the same folder. Run scripts from inside `src/` (or `cut/`).

## Thesis to code

Every stage and experiment of the thesis (E1 to E14) is mapped to its script in [docs/experiment_map.md](docs/experiment_map.md). The same code is printed, function by function, in Appendix A of the thesis (listings L1 to L24).

| Thesis | Scripts |
|---|---|
| Stage 1, segmentation and triage | `src/plate_segmentation_triage.py` |
| Stage 2, verification in Label Studio | `src/label_studio_import.py`, `src/label_studio_export_merge.py` |
| Stage 3, LPGS generator 1 (trace once) | `src/auto_glyph_variator1.py` |
| Stage 3, LPGS generator 2 (retrace per sample) | `src/auto_glyph_variator1_3_.py` |
| Stage 3, LPGS generator 3 (skeleton graph) | `src/auto_glyph_variator_deformator2.py` |
| Stage 4, CUT | `cut/` |
| Detector | `src/compose_detection_pages.py`, `src/train_rcnn_local.py`, `src/eval_detector.py` |
| Classifier, abstention, open-set | `src/classify_detections.py`, `src/sweep_abstention.py`, `src/score_open_set.py` |
| Fine-tuning and compression | `src/finetune_resnet.py`, `src/compress_classifier.py` |
| Review application (current) | `src/nsibidi_app.py`, `src/box_editor.py`, `src/labels_en.csv` (run with `streamlit run nsibidi_app.py`) |
| Review application (three-lane version used for E5, E9, E14) | `legacy/nsibidi_app_three_lane.py` |

## Setup

Versions are pinned in `requirements.txt`, taken from the Windows 11 environment the local experiments ran in (the full `pip freeze` is in `docs/pip_freeze_windows.txt`).

```
pip install -r requirements.txt
```

* **PyTorch:** the experiments used the CUDA 12.1 builds (`torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`). To match them, install `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`. A GPU is optional but is used for fine-tuning, CUT and the detector.
* **OpenCV:** the freeze contains both `opencv-python` (5.0.0.93) and `opencv-python-headless` (4.14.0.94), and both provide `cv2`. Install only one to avoid a clash. The code was written against `opencv-python`.
* **`potracer` and `cairosvg`:** `potracer` is a pure-Python port of potrace (imported as `potrace`) and `cairosvg` rasterises SVG. Neither needs an external binary, although `cairosvg` needs the Cairo library on some systems.
* **CUT (`cut/`):** it was trained on a cloud GPU, so its environment is not in the freeze. `cut/glyph_datasets.py` imports `scikit-learn`, which is listed unpinned.
* **Label Studio** is only needed for Stage 2 (`pip install label-studio`, version 1.23.0 in the freeze).
* **Adinkra pretraining:** `notebooks/01_pretrain_adinkra_resnet18.py` is a Google Colab cell with its own environment.

## Review application

`src/nsibidi_app.py` is a Streamlit app that runs on a CPU. It asks three things in order: the writing system (Nsibidi is the only one configured), the task, and then the work itself.

* **A. Segment a plate for LPGS** (the Label Studio route): cut a plate into glyph candidates, adjust each box, name it or mark it as a new sign, keep it as a seed and generate LPGS variations.
* **B. Detect and recognise**: the detector finds glyphs, the 1,400-sign network proposes names, and the reviewer adjusts the boxes and confirms each reading. Confirmed crops can be added to the dictionary.

Boxes can be moved, resized, drawn and deleted on the plate (`box_editor.py`). `labels_en.csv` is the glossary that gives each sign number S0001 to S1400 its English meaning, and the "Dictionary legend" tab shows it. Output goes under `app_output/<writing system>/`.

The earlier version of the app (`legacy/nsibidi_app_three_lane.py`) colours each box as auto, review or unknown from its similarity score (the three-lane rule of the thesis). The current app shows the nearest signs with their similarity and margin and records a reviewer decision for every box.

## Paths

The scripts were written on Windows and take folders as command-line arguments. Hard-coded example paths in headers and defaults have been replaced by placeholders:

* `<ROOT>` is your project folder (this includes the default dictionary folder in `nsibidi_app.py`) (the folder holding the real glyphs, synthetic corpora and plates).
* `<HOME>` and `<USER>` stand for a home folder and a user name.

Search and replace these before running, or pass your own paths. Each script's header describes its arguments, and `python <script> --help` lists them.

## Data

Datasets, trained weights and generated corpora are not stored in this repository. The thesis (Section 3.3) describes the sources: the historical plates, the Igbo-Nsibidi dictionary prototypes, the Adinkra symbol set and the real glyph crops. `src/labels_en.csv` is the glossary of the 1,400 dictionary signs. The Igbo-Nsibidi dictionary it relates to is licensed CC BY-NC-SA 3.0, so check the licence terms before reusing it. Check each source's own licence before redistributing anything.

## Running the pipeline in order

1. Segment a plate and triage the boxes: `plate_segmentation_triage.py`, then Label Studio (`label_studio_import.py`, `label_studio_export_merge.py`).
2. Split and check the real glyphs: `split_real.py`, `check_leakage.py` (and `resplit_real.py`, `dup_check.py` for the 1,400-sign set).
3. Generate synthetic glyphs: one of the three LPGS generators in `src/`, then optionally `cut/train_cut.py` and `cut/generate_refined_dataset_cut.py`.
4. Pretrain the Adinkra ResNet-18: `notebooks/01_pretrain_adinkra_resnet18.py`.
5. Fine-tune and evaluate: `finetune_resnet.py`, `classify_detections.py eval`.
6. Compose detection pages and train the detector: `compose_detection_pages.py`, `train_rcnn_local.py`, `eval_detector.py`.
7. Name boxes on a page and tune thresholds: `classify_detections.py classify`, `sweep_abstention.py`.
8. Review and grow the vocabulary: `streamlit run nsibidi_app.py` (or `retrain_new_signs.py` from the command line).

## Citation

If you use this code, please cite the thesis (see `CITATION.cff`).
