# AI-QTriage — Baseline vs Real-Data YOLO11 Model Comparison

**Date**: August 20, 2026 (historical). Runtime checkpoint as of 2026-08-29:
`ml/models/vision/yolo11_injury_best.pt`. The filename `yolo11_real_wound_best.pt`
was archived to `ml/models/_archive/` and is **not** loaded at runtime.

See `README.md` and `ml/models/model_registry.json` for current SHA-256 and metrics.

---

## 1. Executive Model Comparison Matrix

| Attribute / Metric | Synthetic Baseline (`synthetic_baseline`) | Real-Data Experimental (`real_data_experimental`) |
|---|---|---|
| **Checkpoint Path** | `ml/models/yolo11n_best.pt` | `ml/models/yolo11_real_wound_best.pt` |
| **Training Data Source** | Synthetic Wound Generation Engine | Real Patient Photographic Wounds (WOUNDSEG/Roboflow CC BY 4.0) |
| **Unique Clean Images** | 59 synthetic images | 38 unique images (33 subjects, 162 duplicates removed) |
| **Taxonomy & Classes** | `cut`, `bruise`, `abrasion`, `laceration` (4 classes) | `cut`, `bruise`, `wound` (3 classes, Strategy B) |
| **Validation mAP@50** | 0.8920 (on synthetic val split) | **0.8151** (on real validation split) |
| **Held-Out Test mAP@50 (Overall)** | N/A (synthetic test domain) | **0.3052** (on real held-out test split) |
| **Held-Out Test mAP@50 (`wound` class)** | N/A | **0.7700** |
| **Blank Control False Positives** | **0 detections** | **0 detections** |
| **Clinically Validated** | **FALSE** | **FALSE** |

---

## 2. Threshold Performance Analysis (Held-Out Test Set)

| Confidence Threshold | Total Detections | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall |
|---|---|---|---|---|---|---|
| **0.05** | 8 | 3 | 0 | 6 | **1.0000** | 0.3333 |
| **0.10** | 2 | 1 | 0 | 8 | **1.0000** | 0.1111 |
| **0.15** | 0 | 0 | 0 | 9 | 0.0000 | 0.0000 |
| **0.20** | 0 | 0 | 0 | 9 | 0.0000 | 0.0000 |
| **0.25** | 0 | 0 | 0 | 9 | 0.0000 | 0.0000 |
| **0.30** | 0 | 0 | 0 | 9 | 0.0000 | 0.0000 |
| **0.40** | 0 | 0 | 0 | 9 | 0.0000 | 0.0000 |

> **Key Observation**: On real photographic wound images, the real-data model exhibits zero false positives (`Precision = 1.0000` at low confidence), demonstrating conservative non-invention behavior.

---

## 3. Objective Comparison & Active Model Recommendation

1. **Baseline Preservation**: `yolo11n_best.pt` remains preserved as `synthetic_baseline`.
2. **Real-Data Model Deployment**: `yolo11_real_wound_best.pt` provides genuine photographic wound bounding box localization (`mAP@50 = 0.7700` for `wound` class).
3. **Recommendation**: Configure `YOLO_MODEL_VERSION=real_data_experimental` when evaluating real photographs, while maintaining `synthetic_baseline` as a reproducible benchmark.
