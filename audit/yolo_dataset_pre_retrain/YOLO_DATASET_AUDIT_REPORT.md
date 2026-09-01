# YOLO dataset audit (pre-retrain)

Recorded: 2026-08-28T15:22:50.923880+00:00

This audit did **not** train, did **not** change labels, and did **not** copy or delete images.

Unique images are distinct decoded-pixel SHA-256 values. A JPG and a PNG of the same pixels count as one sample.

Active runtime classes: `cut`, `bruise`, `wound`  
Active training set for the canonical weights: `data/datasets/yolo_merged`  
Machine-readable copy: `audit/yolo_dataset_pre_retrain/yolo_dataset_audit.json`

**Do not retrain on this data until the blockers below are fixed.**

---

## Must fix before retraining

### 1. SYNTHETIC_CLAIMED_AS_REAL_OR_PUBLIC (blocker)

Every YOLO detect image is a synthetic drawing. `yolo_injury` stems are `synthetic_wound_syn_wound_*`. `yolo_real_wound` / `public_sample_*` come from PIL drawings in `ml/training/download_public_datasets.py`, while manifests claim Kaggle/Roboflow/WOUNDSEG and the folder is named `yolo_real_wound`.

Retraining cannot produce a real-injury detector. Metrics would describe drawings, not clinical photos.

### 2. SWELLING_REMAPPED_TO_WOUND (blocker)

`yolo_real_wound/manifest.csv` remaps `raw_class=swelling` → `final_class=wound` on **36** samples. Do not auto-relabel. Decide whether swelling is a YOLO class (it is not in `model.names` today) and label by hand.

### 3. ABRASION_LACERATION_REMAPPED_TO_WOUND (blocker)

`ml/training/train_yolo.py` remaps `yolo_injury` `abrasion(2)` and `laceration(3)` to `wound(2)`. Merged `wound` is abrasion + laceration + remapped swelling + a few cut/bruise boxes. Do not auto-relabel. Freeze a taxonomy, then label to it.

### 4. SEVERE_CLASS_IMBALANCE (blocker)

`yolo_merged` unique images containing class: wound **84**, cut **22**, bruise **17**. Boxes: wound **423**, cut **22**, bruise **17** (ratio 24.9:1). Train unique images: wound **56**, cut **15**, bruise **12**.

Do not duplicate existing images to inflate counts. Collect more unique cut and bruise samples, or drop classes.

### 5. FRAGMENTED_MULTI_BOX_LABELS (blocker)

`yolo_real_wound` train has **256 wound boxes on 24 images** (max **14** boxes in one file, many with `w=h=0.049`). Example: `labels/train/public_sample_0082.txt`. Do not auto-rewrite. Review by hand.

### 6. INSUFFICIENT_UNIQUE_TRAIN_IMAGES (blocker)

`yolo_merged` unique pixel hashes: train **83**, val **17**, test **23**. Too small for a production YOLO11 retrain. Collect more unique images. Do not copy files to increase N.

### 7. EXACT_DUPLICATE_IN_RAW_SYNTHETIC (blocker)

`raw/synthetic_wound`: **222** files, **221** unique pixels. `syn_wound_0000.jpg` equals `syn_wound_duplicate.jpg`. Do not treat that pair as two samples.

### 8. QC_VISUALIZATIONS_AND_MASKS_NOT_SAMPLES (blocker)

- `yolo_merged` splits: **0** JPG/PNG pairs, **0** byte duplicates, **0** stem duplicates.
- `yolo_real_wound/qc_samples`: **38** unlabeled overlay JPGs. Not training samples.
- `public_wound_dataset`: **200** JPG + **200** mask PNG. Masks are not extra images.
- `injury_dataset`: **30** JPG + **30** mask PNG.

Do not glob QC overlays or masks into train.

### 9. SPLIT_HOMOGENEITY_NOT_INDEPENDENT (blocker)

Exact train/val/test leakage in `yolo_merged`: **0** shared pixel hashes, **0** shared stems. Fake `subject_id`s do not appear in multiple splits.

Average-hash still groups **122 / 123** merged images because they are near-identical synthetic canvases. Val/test cannot prove generalization to real injuries. Collect real photos and split by real subject. Do not reshuffle these drawings.

### 10. NO_NEGATIVE_IMAGES_IN_ACTIVE_SPLITS (should_fix)

`yolo_merged` has **0** empty-label negatives. `yolo_injury/blank_skin.jpg` and `dummy_test.jpg` sit outside splits. Add true unlabeled negatives. Do not duplicate positives.

---

## 1. Unique images by split

Counts below are unique decoded-pixel SHA-256 values (same as file counts except where noted).

| dataset | train | val | test | unsplit | unique pixels (all) | files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo_merged (active) | 83 | 17 | 23 | 0 | 123 | 123 |
| yolo_injury | 59 | 12 | 14 | 2 | 87 | 87 |
| yolo_real_wound | 24 | 5 | 9 | 38 | 76 | 76 |
| raw/synthetic_wound | 151 | 35 | 35 | 0 | 221 | 222 |

`yolo_real_wound` unsplit = `qc_samples/` overlays. `yolo_injury` unsplit = `blank_skin.jpg` + `dummy_test.jpg`. `raw/synthetic_wound` train files=152, unique pixels=151 because of the duplicate JPG.

## 2. Bounding boxes per class

### yolo_merged (active classes cut / bruise / wound)

| split | cut | bruise | wound |
| --- | ---: | ---: | ---: |
| train | 15 | 12 | 288 |
| val | 2 | 1 | 70 |
| test | 5 | 4 | 65 |
| **total** | **22** | **17** | **423** |

### yolo_injury (cut / bruise / abrasion / laceration)

| split | cut | bruise | abrasion | laceration |
| --- | ---: | ---: | ---: | ---: |
| train | 15 | 12 | 18 | 14 |
| val | 2 | 1 | 3 | 6 |
| test | 4 | 3 | 4 | 3 |
| **total** | **21** | **16** | **25** | **23** |

### yolo_real_wound (claimed cut / bruise / wound)

| split | cut | bruise | wound |
| --- | ---: | ---: | ---: |
| train | 0 | 0 | 256 |
| val | 0 | 0 | 61 |
| test | 1 | 1 | 58 |
| **total** | **1** | **1** | **375** |

### raw/synthetic_wound

| split | cut | bruise | abrasion | laceration |
| --- | ---: | ---: | ---: | ---: |
| train | 47 | 40 | 31 | 34 |
| val | 6 | 13 | 8 | 8 |
| test | 7 | 9 | 8 | 11 |
| **total** | **60** | **62** | **47** | **53** |

## 3. Unique images containing each class

### yolo_merged

| split | cut | bruise | wound |
| --- | ---: | ---: | ---: |
| train | 15 | 12 | 56 |
| val | 2 | 1 | 14 |
| test | 5 | 4 | 14 |
| **total** | **22** | **17** | **84** |

## 4. Duplicate JPG/PNG files

| dataset | jpg+png same stem | byte-duplicate groups | pixel-duplicate groups |
| --- | ---: | ---: | ---: |
| yolo_merged | 0 | 0 | 0 |
| yolo_injury | 0 | 0 | 0 |
| yolo_real_wound | 0 | 0 | 0 |
| raw/synthetic_wound | 0 | 1 | 1 |

The only exact file duplicate is `syn_wound_0000.jpg` / `syn_wound_duplicate.jpg`.

Mask PNGs next to JPGs in `public_wound_dataset` and `injury_dataset` are segmentation masks, not a second image sample.

## 5–6. Duplicates across train / val / test, and duplicate stems

`yolo_merged`: **0** shared pixel hashes across splits, **0** shared stems across splits, **0** duplicate stems inside a split.

## 7–10. Invalid labels, empty labels, boxes outside image, zero-area boxes

| dataset | invalid | empty | missing | outside image | zero-area |
| --- | ---: | ---: | ---: | ---: | ---: |
| yolo_merged | 0 | 0 | 0 | 0 | 0 |
| yolo_injury | 0 | 0 | 2 | 0 | 0 |
| yolo_real_wound | 0 | 0 | 38 | 0 | 0 |
| raw/synthetic_wound | 0 | 0 | 0 | 0 | 0 |

Missing labels: `blank_skin.jpg`, `dummy_test.jpg`, and 38 `qc_samples/` overlays. Geometry fields are syntactically valid. Semantic over-boxing in `yolo_real_wound` is still a blocker (item 5).

## 11. Class imbalance

`yolo_merged` box ratio max/min = **24.882** (wound vs bruise). Train cannot learn cut/bruise comparably to wound.

## 12. Train / validation / test leakage

Exact image leakage in `yolo_merged`: **none**.

Cross-dataset overlap is expected copies into the merge: **85** shared pixel hashes (merged vs sources), **123** shared stems.

## 13. Same person / near-identical images in multiple splits

- Manifest `subject_id`s are generated (`subj_001` …). They do **not** leak across YOLO splits (count **0**).
- They are not real people.
- 8×8 average-hash groups **122/123** `yolo_merged` images: the drawings are visually homogeneous. That is not 122 copies of one file (pixel hashes are unique).

## 14. Negative / no-injury images

| location | role |
| --- | --- |
| yolo_merged splits | **0** empty-label negatives |
| yolo_injury/blank_skin.jpg | negative, outside splits |
| yolo_injury/dummy_test.jpg | 64×64 black dummy from `/api/models`, outside splits |

## 15. Dataset source classification

| dataset | claimed | actual | reason |
| --- | --- | --- | --- |
| yolo_merged | real_wound + yolo_injury | **synthetic** | both sources are drawings |
| yolo_injury | mixed/public possible | **synthetic** | all stems `synthetic_wound_syn_wound_*` |
| yolo_real_wound | real / public | **synthetic** | `public_sample_*` from PIL generator |
| raw/synthetic_wound | synthetic | **synthetic** | `scripts/prepare_yolo_dataset.py` |
| public_wound_dataset | Kaggle/Roboflow/WOUNDSEG | **synthetic** | `generate_expanded_wound_dataset()` |
| injury_dataset | synthetic benchmark | **synthetic** | manifest says AI-QTriage Synthetic Research Benchmark |

No inspected YOLO detect set is **real** or **public** photographs. Overall: **synthetic**.

---

## What not to do next

- Do not retrain YOLO on `yolo_merged` as it stands.
- Do not auto-change labels.
- Do not duplicate JPG/PNG or existing drawings to raise counts.
- Do not treat QC overlays or mask PNGs as extra samples.
