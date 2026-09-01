# YOLO Roboflow-v1 Forensic Retrain Report

Generated: 2026-08-30. Research prototype only — **not clinical**, **not production-medical**.

## 1. Executive Verdict

**PARTIALLY_WORKING_RESEARCH_PROTOTYPE**

YOLO skin detection was retrained on **real public Roboflow bounding boxes** (polygons converted to boxes), evaluated against production, and **promoted** after all documented gates passed.

The forensic real hand-cut case now keeps **Cut @ 60.3%** on the injury region at keep-threshold 0.25. The previous production model kept **Bruise @ 75.7%** on the wrist, missing the cut.

**BROWSER_GUI_VERIFIED:** no (this session).  
**API / direct-inference / pytest:** verified.

---

## 2. Current Production Model Inventory

| Model | Path | Version | Data | Runtime? |
|-------|------|---------|------|----------|
| YOLO11 skin | `ml/models/vision/yolo11_injury_best.pt` | **roboflow-v1** | Real Roboflow photos | Yes |
| EfficientNetV2 | `ml/models/vision/efficientnetv2_injury_best.pt` | kaggle-v1 | Real Kaggle photos | Yes |
| U-Net | `ml/models/vision/unet_injury_best.pt` | deduped-subject-v1 | Real wound masks | Yes |
| XGBoost | `ml/models/xgboost_best.json` | v1.2.0 | **SYNTHETIC** | Yes (fusion) |
| VQC | `ml/models/vqc/vqc_weights.npz` | v1.4.0 | **SYNTHETIC / EXPERIMENTAL_ONLY** | Isolated |
| Sensor XGB | `ml/models/sensor_motion_best.json` | v1.2.0 | Synthetic 50 Hz | **Not called in `/analyze`** |
| Fracture YOLO | `ml/models/vision/yolo11_fracture_xray_best.pt` | separate | X-ray | **Not in skin pipeline** |

---

## 3. SHA-256 Table

| Artifact | SHA-256 |
|----------|---------|
| **Live canonical YOLO** | `319a2cbc15d6ced2730060ff6e73baf2968271026611124539ce0b06486a1926` |
| Metadata `artifact_sha256` | **matches** |
| Registry / manifest | **matches** (`roboflow-v1`) |
| Pre-promotion backup | `ml/models/vision/yolo11_injury_best.pt.pre_roboflow_v1_backup` = previous `4d6e72f5…` |
| Historical v1.4.0 backup | `.pre_retrain_v2_backup` = `6cc84115…` (not runtime) |
| Candidate `best.pt` | **byte-identical** to promoted canonical |

Checkpoint names: `{0: cut, 1: bruise, 2: abrasion}`. Keep-threshold **0.25**.

---

## 4. Dataset Provenance

| Dataset | URL | License | Images on disk | Used |
|---------|-----|---------|----------------|------|
| Injury Detection v2 | universe.roboflow.com/bryans-workspace-rrftd/injury-detection-4vjih | CC BY 4.0 | 847 | Yes |
| wound2 | universe.roboflow.com/dongdong-d6lo7/wound2 | CC BY 4.0 | 476 | Yes |
| aid v2 | universe.roboflow.com/tingting-rph02/aid-lvngz | CC BY 4.0 | 259 | Yes |
| Wound Detection & Segmentation | Raghav Bharathi | CC BY 4.0 | — | **FAILED** (API: no versions) |
| Kaggle leoscode 2760 | Kaggle | research-only | — | **BLOCKED** (no kaggle.json) |
| Confirmed negatives | yolo_real_skin_v2 empties + blank_skin/dummy | mixed | 250 | Yes |

Classification-only Kaggle folders were **not** converted into YOLO boxes.

Polygons with &gt;5 YOLO tokens were converted to axis-aligned boxes via min/max vertices. That is documented, not fabricated labels.

---

## 5. Dataset Class Counts (`data/datasets/yolo_roboflow_v1`)

| Split | Images | cut boxes | bruise boxes | abrasion boxes | empty negatives |
|-------|--------|-----------|--------------|----------------|-----------------|
| train | 933 | 81 | 458 | 425 | 176 |
| val | 198 | 24 | 86 | 97 | 27 |
| test | 220 | 24 | 101 | 94 | 47 |
| **total** | **1351** | **129** | **645** | **616** | **250** |

Excluded (not remapped): Burn 212, Stab 21, scarring 13, Injury-detection 6.

**wound is not in `model.names`.** Insufficient honest wound boxes.

---

## 6. Leakage Audit

Exact SHA cross-split: train∩val = **0**, train∩test = **0**, val∩test = **0**.  
Forensic hand image SHA excluded from all splits.  
Verdict: **PASS** — training proceeded.

---

## 7. Candidate vs Baseline (same Roboflow test YAML)

Production (old 3-class cut/bruise/**wound**) evaluated on the new YAML is **not class-aligned** (index 2 was wound, now abrasion). That comparison is recorded as schema-mismatch: mAP50 **0.0**. It is **not** used as a promotion metric.

Honest comparison is **localization on the held-out forensic photo** plus held-out Roboflow test for the candidate only.

| | Old production (synthetic) | Roboflow-v1 candidate |
|------|---------------------------|----------------------|
| Training data | Synthetic drawings; wound=0 boxes | Real photos, 1390 mapped boxes |
| Test mAP50 (Roboflow test) | N/A (schema mismatch) | **0.538** |
| Test mAP50-95 | — | **0.295** |
| Precision / recall | — | **0.625 / 0.605** |

Training: YOLO11n pretrained, 8 epochs, batch 8, imgsz 416, seed 42, CPU, AdamW. Logs: `scratch/yolo_roboflow_v1_train.log`, `ml/models/yolo_roboflow_candidate_v1/run_v1/results.csv`.

Val mAP50 last epoch (Ultralytics): **0.630**.

---

## 8. Per-Class YOLO Metrics (held-out test, conf=0.25)

| Class | Precision | Recall | mAP50 | mAP50-95 | Test support (boxes) |
|-------|-----------|--------|-------|----------|----------------------|
| cut | 0.492 | 0.583 | 0.508 | 0.321 | 24 |
| bruise | 0.769 | 0.604 | 0.557 | 0.296 | 101 |
| abrasion | 0.615 | 0.628 | 0.550 | 0.267 | 94 |

Cut support is still small. Do not claim cut reliability.

---

## 9. Negative-Image Tests (keep 0.25)

| Image | Candidate detections | Old production |
|-------|----------------------|----------------|
| blank_skin.jpg | 0 | 0 |
| dummy_test.jpg | 0 | 0 |
| black / white / gray | 0 | 0 (black had a 0.01-conf ghost, filtered at 0.25) |

No increase in application-threshold false positives on canonical negatives.

---

## 10. Forensic Hand-Cut Test (`3f629ca8-…jpeg`)

Heuristic injury GT ≈ `[550, 200, 750, 480]`.

| | Old production @ 0.25 | Promoted model @ 0.25 |
|--|------------------------|-------------------------|
| Detections | 2× **bruise** (wrist / opposite edge) | 1× **cut** `[495, 277, 774, 408]` conf **0.603** |
| Covers injury? | **No** | **Yes** (IoU 0.395 vs GT) |
| Wrist bruise FP? | **Yes** | **No** at 0.25 |

At conf 0.01 the candidate still emits extra abrasion/cut boxes near the injury; keep-threshold 0.25 retains the correct cut only.

---

## 11. Promotion Decision

**PROMOTED** — all gates true:

- leakage_free  
- all_classes_nonzero_train  
- negatives_no_worse  
- blank_clean  
- hand_localization_improved_or_ok  
- sha_distinct  
- map50_reasonable (≥0.25)  
- hand_no_wrong_region_at_keep  

Backup retained. Registry + metadata + wrapper classes updated. `--promote` was applied only after gates.

---

## 12. Remaining Data Blockers

- Kaggle credentials absent → leoscode 2760 not downloaded.  
- Roboflow “Wound Detection and Segmentation” returned **no versions**.  
- No honest **wound** box class for the production head.  
- Burn/Stab excluded (taxonomy + sample size).  
- Fall/impact sensor data still missing (UCI HAR = ADL only).  
- No paired multimodal clinical records → XGB/VQC stay synthetic.

---

## 13. Non-Dataset Engineering Fixes

- YOLO wrapper untrained list: **wound / laceration / swelling** (abrasion is trained).  
- First-aid `supported_classes` loaded from live `model.names`.  
- Questionnaire routing accepts **abrasion**, not wound.  
- `/api/models` wound_note updated.  
- Tests updated to new SHA and taxonomy (not weakened).  
- Sensor classifier still **not** invoked in `/analyze` (feature-schema / product contract); only `process_sensor_data` summaries. **Not wired** — would be a fake “fall detector” otherwise.  
- Fracture YOLO remains a separate artifact.

---

## 14. Tests / Build

| Check | Result |
|-------|--------|
| `pytest backend/tests -q` | **231 passed** |
| `npm run lint` | pass |
| `npm run build` | pass |
| Browser E2E | **not run this session** → **API_VERIFIED_ONLY** |

---

## 15. Honest Readiness

| Feature | Score |
|--------|------:|
| Backend | 84 |
| Frontend | 80 |
| YOLO (real photos) | **62** (was ~35) |
| EfficientNet | 78 |
| U-Net | 72 |
| Sensor | 35 (artifact unused in analyze) |
| XGBoost | 45 synthetic |
| VQC | 30 experimental |
| E2E software | 72 |
| Dataset quality | 68 |

**Overall research prototype: 68/100**

Not clinically validated. Cut class still has only 129 boxes. mAP50 0.54 is a research demo, not a diagnostic claim.
