# FINAL REAL-DATA RETRAINING REPORT

Generated: 2026-08-30 (AI-QTriage Capstone — research prototype, **not clinical**)

## 1. Executive Verdict

**PARTIALLY_WORKING_RESEARCH_PROTOTYPE**

The pipeline runs end-to-end with honest error handling. **EfficientNet kaggle-v1** and **U-Net deduped-subject-v1** use real public wound imagery (classification / segmentation). **YOLO11 production remains synthetic** for cut/bruise and cannot be honestly improved until Roboflow injury-detection exports are downloaded. **XGBoost/VQC remain SYNTHETIC**. **Sensor** now has a real UCI HAR normal-activity feature manifest, but fall/impact labels are **LABEL_MAPPING_LIMITATION**.

---

## 2. Datasets Used

| Dataset | Source | License | Samples | Real? | Task | Used in training |
|---------|--------|---------|---------|-------|------|------------------|
| Injury Detection v2 | Roboflow | CC BY 4.0 | ~847 | Yes | YOLO | **BLOCKED** — no API key |
| wound2 | Roboflow | CC BY 4.0 | ~198 | Yes | YOLO | **BLOCKED** |
| Wound Detection & Segmentation | Roboflow | CC BY 4.0 | ~547 | Yes | YOLO/seg | **BLOCKED** |
| aid | Roboflow | CC BY 4.0 | ~259 | Yes | YOLO | **BLOCKED** |
| yasinpratomo wound | Kaggle | unknown card | 431 | Yes | classification | EffNet kaggle-v1 |
| ibrahimfateen wound classification | Kaggle | unknown card | 2940 | Yes | classification | EffNet kaggle-v1 |
| hf_wseg_dataset | Hugging Face | CC-BY-NC-4.0 | 2686 pairs | Yes | segmentation | U-Net deduped |
| Medetec 224 | GitHub/UWM | Medetec terms | ~112 pairs | Yes | segmentation | U-Net deduped |
| AZH patches | Mendeley extract | CC BY 4.0 | 30 pairs | Yes | segmentation | U-Net deduped |
| UCI HAR | UCI | CC BY 4.0 | 10299 windows | Yes | sensor ADL | manifest only (normal) |
| leoscode wound-seg 2760 | Kaggle | research-only | 2760 | Yes | segmentation | **BLOCKED** — no kaggle.json |
| shubhambaid skin burn YOLO | Kaggle | CC0 | 1227 | Yes | YOLO burn | on disk, not merged (burn excluded from skin taxonomy) |

Artifacts: `data/dataset_catalog.json`, `data/manifests/source_provenance.json`, `data/manifests/acquisition_report.json`

---

## 3. Dataset Integrity

| Check | Result |
|-------|--------|
| Corrupted/zero-byte image files in processed sets | 404 empty label files (expected YOLO negatives) — see `data/manifests/dataset_integrity_report.json` |
| Exact duplicate SHA groups (processed) | 203 groups — mostly cross-source wound photos; dedupe applied in EffNet/U-Net prep |
| Near duplicates | 5 hamming-neighbors in U-Net deduped prep |
| Train/val/test leakage (EffNet kaggle-v1) | **0** exact hash cross-split |
| Train/val/test leakage (yolo_real_skin_v2) | **0** exact hash cross-split |
| Train/val/test leakage (unet_deduped) | **0** subject + hash overlap |

**Roboflow YOLO merge:** not executed — `data/manifests/yolo_roboflow_prepare_report.json` status `BLOCKED_NO_ROBOFLOW_DATA`.

---

## 4. Training Details

### YOLO11 (highest priority)

| Field | Value |
|-------|-------|
| Status | **NOT RETRAINED on real injury boxes** |
| Blocker | `ROBOFLOW_API_KEY` not set; no `data/raw/roboflow/*` exports |
| Prior candidate | `yolo_real_skin_v2` — 10 epochs CPU, **KEEP_BASELINE** (failed gate1 + gate4 hand localization) |
| Production SHA | `4d6e72f5…` unchanged |
| Production classes | cut, bruise, wound — **wound has 0 training boxes** |

### EfficientNetV2

| Field | Value |
|-------|-------|
| Dataset | `efficientnet_kaggle_v1` — 3039 rows, 8 classes |
| Architecture | EfficientNetV2-S transfer learning |
| Production | kaggle-v1 SHA `95cf385d…` — **unchanged** (re-prepared dataset matches existing production lineage) |
| OOD re-eval | Blank/black/white/gray → **LOW_QUALITY_INPUT** or **OUT_OF_DISTRIBUTION** — no swelling collapse |

### U-Net ResNet34

| Field | Value |
|-------|-------|
| Dataset | `unet_deduped_subject` — 464 pairs (444 positive chronic wound, 20 synthetic empty) |
| Epochs | 8 CPU, batch 4, lr 3e-4 |
| Best epoch | 7 (val Dice 0.660) |
| Test Dice | 0.642 |
| OOD probes | black/white/gray → positive_ratio **0.0** |
| Promotion | **KEEP_BASELINE** — candidate SHA identical to production; no metric gain |

### Sensor

| Field | Value |
|-------|-------|
| UCI HAR manifest | 10299 rows → `data/manifests/sensor_uci_har_manifest.csv` |
| Limitation | **LABEL_MAPPING_LIMITATION** — normal ADL only; fall/impact need KFall/MobiFall |
| Production sensor model | unchanged synthetic v1.2.0 |

### XGBoost / VQC

| Field | Value |
|-------|-------|
| Data | **SYNTHETIC_RESEARCH_BASELINE** — no paired public multimodal clinical records found |
| VQC | **EXPERIMENTAL_ONLY** |

---

## 5. Model Results Summary

| Model | Real/Synthetic | Test set | Main metric | OOD test | Status |
|-------|----------------|----------|-------------|----------|--------|
| YOLO11 | Synthetic cut/bruise | yolo_retrain_v2 test (19 img) | mAP50 0.836 | blank OK at 0.25; **hand photo wrong region** | **NOT TRUSTWORTHY on real photos** |
| EfficientNetV2 | Real Kaggle photos | kaggle-v1 test (473) | macro-F1 0.937 | blank withheld | **READY_FOR_RESEARCH_DEMO** |
| U-Net | Real wound masks | deduped test (69) | Dice 0.642 | blank ratio 0.0 | **READY_FOR_RESEARCH_DEMO** |
| Sensor | Synthetic (+ UCI manifest) | synthetic test | acc ~high on synthetic | N/A | **SYNTHETIC baseline** |
| XGBoost | Synthetic | held-out synthetic | varies | N/A | **SYNTHETIC baseline** |
| VQC | Synthetic | held-out synthetic | experimental | N/A | **EXPERIMENTAL_ONLY** |

---

## 6. Required Next Steps (to unblock YOLO real-data retrain)

1. Set `ROBOFLOW_API_KEY` in environment (never commit to repo).
2. Run: `python ml/training/acquire_real_public_datasets.py`
3. Run: `python ml/training/prepare_yolo_roboflow_unified.py`
4. Run leakage audit: `python scratch/build_data_manifests.py`
5. Train YOLO candidate with promotion gates (hand-case + negatives + per-class support).
6. Configure Kaggle credentials for leoscode 2760 if U-Net expansion needed.

---

## 7. Final Readiness Scores

| Feature | Score |
|---------|------:|
| Backend API | 82 |
| Frontend | 80 |
| YOLO | 35 |
| EfficientNet | 78 |
| U-Net | 72 |
| Sensor | 40 |
| XGBoost | 45 |
| VQC | 30 |
| End-to-End Integration | 70 |
| Dataset Quality (real injury vision) | 55 |

**Overall Research Prototype Score: 62/100**

**231 backend tests passed** (2026-08-30).
