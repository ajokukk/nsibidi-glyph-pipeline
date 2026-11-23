# Experiment map

How each stage and experiment of the thesis (Table 3.6 and Chapter 4) maps to the code. Listing numbers (L1 to L24) refer to Appendix A of the thesis.

## Pipeline stages

| Stage | Scripts | Listing |
|---|---|---|
| Stage 1: segmentation and triage | `src/plate_segmentation_triage.py` | L1 |
| Stage 2: verification in Label Studio | `src/label_studio_import.py`, `src/label_studio_export_merge.py` | L2 |
| Stage 3: LPGS, generator 1 (trace once) | `src/auto_glyph_variator1.py` | L3 |
| Stage 3: LPGS, generator 2 (retrace per sample) | `src/auto_glyph_variator1_3_.py` | L4 |
| Stage 3: LPGS, generator 3 (skeleton graph) | `src/auto_glyph_variator_deformator2.py` | L5 |
| Stage 3: potrace trace check | `src/trace_compare.py` | not listed |
| Stage 4: CUT style refinement | `cut/` (models, patchnce, ssim_loss, style_loss, diffaugment, glyph_datasets, sanity_check, train_cut, generate_refined_dataset_cut) | L6, L7 |
| Detector | `src/compose_detection_pages.py`, `src/train_rcnn_local.py`, `src/eval_detector.py` | L8, L9, L10 |
| Classifier and fine-tuning | `notebooks/01_pretrain_adinkra_resnet18.py`, `src/classify_detections.py`, `src/finetune_resnet.py` | L11, L12, L13, L14 |
| Review application (current) | `src/nsibidi_app.py`, `src/box_editor.py`, `src/labels_en.csv` | L17, L18 |
| Review application (three-lane version, E5, E9, E14) | `legacy/nsibidi_app_three_lane.py` | L14 |
| Data splits and checks | `src/split_real.py`, `src/check_leakage.py`, `src/resplit_real.py`, `src/dup_check.py` | L24 |

## Experiments

| ID | Scripts | Listings |
|---|---|---|
| E1 | `classify_detections.py eval`, with `--backbone` dino, resnet or both (ResNet-18 weights from the Adinkra pretraining cell) | L11, L12 |
| E2 | `classify_detections.py eval`, with and without `--gray-norm` | L11, L12 |
| E3 | `classify_detections.py eval`, with real and synthetic support folders | L12 |
| E4 | Adinkra pretraining, `finetune_resnet.py` (synthetic crops only), then `classify_detections.py eval` | L11, L12, L13 |
| E5 | `classify_detections.py classify`, `number_boxes.py`, `sweep_abstention.py` | L14, L15 |
| E6 | review app, `retrain_new_signs.py`, `run_lpgs_arm.py`, `finetune_resnet.py`, then `classify_detections.py eval` | L13, L14, L19, L21 |
| E7 | `finetune_resnet.py` with `--real-train-dir` and `--real-repeat 5`, then `classify_detections.py eval` | L12, L13 |
| E8 | `make_baseline_aug.py`, `run_lpgs_arm.py`, CUT generator, `finetune_resnet.py`, then `classify_detections.py eval` | L6, L13, L20, L21 |
| E9 | `score_open_set.py`, `classify_detections.py classify` | L14, L16 |
| E10 | `compose_detection_pages.py`, `train_rcnn_local.py`, `eval_detector.py` | L8, L9, L10 |
| E11 | `ssim_multisource_comparison.py`, `ssim_synthetic_vs_real_addon.py` | L22 |
| E12 | `compress_classifier.py` | L23 |
| E13a, E13b | `finetune_resnet.py`, `classify_detections.py eval`, `resplit_real.py`, `dup_check.py` | L12, L13, L24 |
| E14 | `classify_detections.py classify`, `sweep_abstention.py`, `number_boxes.py` | L14, L15 |

## File names that differ from the working folder

| In this repository | Working-folder name |
|---|---|
| `src/compose_detection_pages.py` | `compose_detection_pages (1).py` |
| `src/train_rcnn_local.py` | `train_rcnn_local (1).py` |
| `src/auto_glyph_variator1_3_.py` | `auto_glyph_variator1 (3).py` |
| `legacy/nsibidi_app_three_lane.py` | the earlier `nsibidi_app.py` |

Earlier versions of these scripts, and other earlier LPGS variants, are not included. The one exception is `legacy/nsibidi_app_three_lane.py`, the review app with the three-lane rule used in E5, E9 and E14.
