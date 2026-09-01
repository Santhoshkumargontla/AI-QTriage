# AI-QTriage Complete Data Acquisition / Retrain / Integration Forensic Report

**Date:** 2026-08-29  
**Workspace:** `C:\Users\santh\Capstone Project Code`  
**Browser GUI:** `BROWSER_GUI_NOT_RUN` (API + direct inference verified; frontend lint/build passed)  
**API verification:** `VERIFIED_BY_API_ONLY`

---

## A. EXECUTIVE VERDICT

### **RESEARCH_DEMO_READY** (with explicit limitations)

Not `PRODUCTION_READY`. Not clinical. One vision model (EfficientNet) was honestly improved and promoted; YOLO cut/bruise real-photo recognition and multimodal fusion remain limited by data.

---

## B. DATASET REPORT

### Acquisition attempts (legitimate sources only)

| Dataset | Source URL | License | Access | Outcome |
| --- | --- | --- | --- | --- |
| AZH patches | UWM / prior extract under `data/datasets/external/azh_patches` | Clinic/challenge terms | Already on disk | **USED** (normals + prior U-Net/YOLO wound boxes) |
| HF wseg | https://huggingface.co/datasets/subbareddyoota/wseg_dataset | CC-BY-NC-4.0 | Already extracted (2686+2686) | Available; used in prior U-Net lineage |
| Medetec 224 | https://github.com/uwm-bigdata/wound-segmentation | Medetec + UWM redistribution | Already on disk | **USED** (normals) |
| Mendeley hsj38fwnvr | https://data.mendeley.com/datasets/hsj38fwnvr/2 | CC BY 4.0 | Direct ZIP **403/404** | **BLOCKED** — no healthy-feet download |
| Roboflow cut/bruise | Universe self-harm / injury datasets | CC BY (listed) | `ROBOFLOW_API_KEY` **absent** | **SKIPPED** |
| Kaggle wound_dataset | https://www.kaggle.com/datasets/yasinpratomo/wound-dataset | Per-image copyright caveats | `kaggle.json` **absent** | **SKIPPED** |
| SurgWound (HF) | https://huggingface.co/datasets/SATHEESHMA/SurgWound | Dataset card | JSON only (no images) | **NOT USABLE** without scraping |
| WILLIE index | https://huggingface.co/datasets/QianGroup/willie-benchmark | mixed terms / index only | Not downloaded | Index ≠ image license |

Access date: 2026-08-29.

### Dataset built this phase: `efficientnet_reject_v2`

| Field | Value |
| --- | --- |
| Path | `data/datasets/efficientnet_reject_v2` |
| Classes | cut, bruise, normal, ood_reject |
| n | 189 (train 125 / val 36 / test 28) |
| Provenance | SYNTHETIC drawings 126; PUBLIC_REAL normals 18; SYNTHETIC_OOD_REJECT 45 |
| Leakage | exact-hash cross-split **0**; subject overlap **0** → `leakage_free: true` |
| Limitations | cut/bruise still drawings; only 18 real normals after near-dup cull; no swelling labels; no Roboflow photos |

Pipeline layout also created: `data/{raw,interim,processed,manifests,reports}` + `data/manifests/efficientnet_reject_v2_*`.

### Unchanged datasets (honest)

| Dataset | Role | Real/Synthetic |
| --- | --- | --- |
| `yolo_retrain_v2` | Production YOLO | SYNTHETIC cut/bruise; wound 0 |
| `yolo_wound_boxes_v1` | Candidate only (reverted) | PUBLIC wound boxes + SYNTHETIC cut/bruise |
| `unet_deduped_subject` | Production U-Net | PUBLIC + synthetic empty |
| XGB/VQC fusion | In-code generator | SYNTHETIC (200×23) |
| Sensor manifest | Generated windows | SYNTHETIC (not SisFall/UCI downloads) |

---

## C. MODEL REPORT

### EfficientNetV2 — **retrained and promoted**

| Item | Value |
| --- | --- |
| Training performed? | **Yes** (`ml/training/train_efficientnet_reject_v2.py`) |
| Dataset | `efficientnet_reject_v2` |
| Baseline SHA | `6944605ae71685d909d505d12ce32d5bd9e953c10bbebb72a02635a729897e83` |
| Candidate/production SHA | `fa7aa5c822e6a4127d7f1aa9d9518e687cc0ee9e3d8cfd362d902f72ac06568c` |
| Backup | `efficientnetv2_injury_best.pt.pre_reject_v2_backup` |
| Classes | cut, bruise, normal, ood_reject |
| Test accuracy | 0.928571 (n=28) |
| Test macro-F1 | **0.875** |
| Baseline OOD injury collapse | **10** |
| Candidate OOD injury collapse | **0** |
| Promotion | **PROMOTE** (gates passed) |
| Status | **READY_FOR_RESEARCH_DEMO** |
| Blank raw (after) | black→Normal; white/gray/noise→OOD_Reject — **not Swelling** |
| Hand ROI (bruise box) | Normal abstention (skin ROI — honest for that crop) |
| Swelling | **Removed** from active head (no labels) |

Integration: abstention classes are not injury findings; Grad-CAM only for injury-confident outputs; registry regenerated.

### YOLO11 — **not promoted**

| Item | Value |
| --- | --- |
| Active SHA | `4d6e72f5…` (unchanged) |
| Wound candidate SHA | `76bccb50…` (kept off production) |
| Hand @0.25 production | bruise 0.7574 on wrist skin (misses laceration) |
| Hand @0.25 wound candidate | wound 0.43 on image **corner** (false positive; prior revert justified) |
| Roboflow photos | Unavailable without API key |
| Verdict | **REQUIRES_MORE_DATA** for real cut/bruise photos; wound remains **UNSUPPORTED** in production set |

### U-Net — **kept**

| Item | Value |
| --- | --- |
| SHA | `3c7f3f39…` |
| Blank raw_pos | ≈0; masks withheld |
| Status | **READY_FOR_RESEARCH_DEMO / DEMO_WITH_LIMITATIONS** |
| Retrain this phase | Not required (OOD blank already cleared previously) |

### XGBoost / VQC / Sensor — **no fake clinical retrain**

| Model | Provenance | Metrics | Status |
| --- | --- | --- | --- |
| XGBoost | SYNTHETIC fusion | 25/30 = 0.833333 | TRAINED / SYNTHETIC_RESEARCH_ONLY |
| VQC | SYNTHETIC | 16/30 = 0.533333 | **EXPERIMENTAL_ONLY** |
| Sensor | SYNTHETIC 50 Hz | 28/36 = 0.777778 | TRAINED / synthetic |

No legitimate paired multimodal clinical dataset mapped to the 23-feature schema.

---

## D. BEFORE VS AFTER

| Model/Feature | Before | Fix | After | Verified |
| --- | --- | --- | --- | --- |
| EffNet blank/OOD raw | Swelling @ ~1.0 | reject-v2 train + promote | Normal/OOD_Reject; collapse 0 | Yes (direct) |
| EffNet status | NOT_TRUSTWORTHY | Promotion gates | READY_FOR_RESEARCH_DEMO | Registry/API |
| EffNet swelling class | Advertised | Removed (no data) | Not in head | Sidecar/API |
| YOLO hand cut | Bruise @0.76 | None (no real cut data) | Same | Yes |
| YOLO wound candidate | Reverted earlier | Confirmed FP on hand | Still not promoted | Yes |
| U-Net blanks | raw_pos≈0 | No change | Same | Yes |
| Vision provenance | Demo sensor labeled image synthetic | Prior fix retained | uploaded provenance | Prior E2E |
| XGB/VQC/Sensor | Synthetic | No fabrication | Same | Artifacts |

---

## E. MODEL SCORECARD (0–100)

| Model | Score | Reason |
| --- | --- | --- |
| YOLO | **42** | Runs; synthetic cut/bruise; wound unsupported; real hand misclass |
| EfficientNetV2 | **68** | OOD abstention fixed; in-domain still drawing-heavy; small test n |
| U-Net | **72** | Public wound demo; blank OK; not cut localization |
| Grad-CAM | **70** | Tied to injury-confident EffNet only; not clinical |
| XGBoost | **55** | Reproducible synthetic metrics; 0 clinical pairs |
| VQC | **35** | Experimental; weaker than XGB; excluded from decisions |
| Sensor | **50** | Synthetic windows; missing features not invented |

---

## F. FEATURE SCORECARD

| Feature | Score | Reason |
| --- | --- | --- |
| Image upload | 85 | SHA persisted; quality gates |
| YOLO detection | 45 | Pipeline correct; model wrong on real cut |
| EfficientNet | 70 | OOD improved; abstention wired |
| Segmentation | 72 | Geometry/blank OK; domain-limited |
| Explainability | 70 | Grad-CAM gated |
| Fusion | 55 | Synthetic labels |
| Sensor analysis | 55 | Synthetic / demo |
| Questionnaire | 80 | Structured; works |
| SOS | 75 | Local sim; Twilio NOT_CONFIGURED unless env |
| MongoDB | 85 | Case persistence verified |
| API | 85 | Analyze/registry/SHA |
| Frontend overlay | 80 | Boxes from API coords; abstention banner |
| Research dashboard | 85 | Canonical metrics |

---

## G. OVERALL PROJECT SCORE

| Dimension | % | How calculated |
| --- | --- | --- |
| Working percentage | **78%** | Core upload→analyze→UI path executes |
| Model reliability | **55%** | Avg model scorecard (vision+tabular) |
| Data quality | **48%** | Heavy synthetic; blocked public cut/bruise downloads |
| Integration | **82%** | SHA/registry/abstention/API aligned after promote |
| **Overall completion** | **~66%** | 0.25·working + 0.30·reliability + 0.25·data + 0.20·integration |

---

## H. BUG / CHANGE REPORT

### Fixed this phase

| ID | Severity | Issue | Fix | Evidence |
| --- | --- | --- | --- | --- |
| DA-01 | P0 | EffNet raw Swelling collapse on blanks | reject-v2 retrain + promote | OOD collapse 10→0; raw probes |
| DA-02 | P1 | Swelling advertised without labels | Dropped from head | classes JSON |
| DA-03 | P1 | Abstention treated as injury | Wrapper + routing + FE | interpret_prediction |

### Remaining

| ID | Severity | Issue |
| --- | --- | --- |
| DA-10 | P0 | YOLO misclassifies real bleeding cut as bruise |
| DA-11 | P0 | No Roboflow/Kaggle/Mendeley credentials → cannot acquire real cut/bruise/healthy feet |
| DA-12 | P1 | EffNet cut/bruise still SYNTHETIC drawings |
| DA-13 | P1 | Only 18 real normal patches after dedupe |
| DA-14 | P2 | XGB/VQC/Sensor remain synthetic |
| DA-15 | P2 | Browser GUI not run this session |

---

## I. MODEL ARTIFACT REPORT

| Model | Canonical path | SHA-256 | Metadata | Registry match |
| --- | --- | --- | --- | --- |
| YOLO | `ml/models/vision/yolo11_injury_best.pt` | `4d6e72f5…` | yolo11_metadata.json | Yes |
| EfficientNet | `ml/models/vision/efficientnetv2_injury_best.pt` | `fa7aa5c8…` | efficientnetv2_metadata.json reject-v2 | Yes |
| U-Net | `ml/models/vision/unet_injury_best.pt` | `3c7f3f39…` | unet_metadata.json | Yes |
| XGBoost | `ml/models/xgboost_best.json` | `73bb5a51…` | xgboost_metadata.json | Yes |
| VQC | `ml/models/vqc/vqc_weights.npz` | `2db769be…` | vqc_metadata.json | Yes |
| Sensor | `ml/models/sensor_motion_best.json` | `707ee9c2…` | sensor_metadata.json | Yes |

---

## J. FINAL READINESS TABLE

| Component | Status |
| --- | --- |
| EfficientNetV2 | **READY_FOR_RESEARCH_DEMO** (OOD abstention; injury classes still drawing-limited) |
| U-Net | **READY_FOR_RESEARCH_DEMO** |
| YOLO | **REQUIRES_MORE_DATA** (real cut/bruise photos) / wound **UNSUPPORTED** |
| XGBoost | SYNTHETIC research / TRAINED |
| VQC | **EXPERIMENTAL_ONLY** |
| Sensor | SYNTHETIC research / TRAINED |
| Twilio | **NOT_CONFIGURED** unless env proves otherwise |
| Clinical validation | **BLOCKED** (0 paired clinical records) |

---

## K. REMAINING LIMITATIONS

1. Cannot claim reliable real-world cut vs bruise detection without labeled photo datasets (Roboflow/Kaggle blocked).
2. EfficientNet injury head still trained mostly on drawings; real hand laceration ROI abstains as Normal (correct for skin crop; does not “solve” cut recognition).
3. No swelling class without data — do not reintroduce.
4. Multimodal XGB/VQC are synthetic-rule research outputs.
5. No browser walkthrough in this session (`BROWSER_GUI_NOT_RUN`).
6. Do not treat READY_FOR_RESEARCH_DEMO as clinical trustworthiness.

---

## L. VERIFICATION COMMANDS

| Command | Result |
| --- | --- |
| EffNet reject-v2 train | PROMOTE; OOD collapse 0 |
| `pytest backend/tests -q` | **231 passed**, 3 SHAP warnings |
| `pytest -q` (root) | Same suite when run alone |
| `npm run lint` / `npm run build` | Pass |
| Direct OOD + hand probes | Documented above |

### No-cheating confirmation

- No fabricated clinical metrics  
- No Roboflow/Kaggle bypass  
- No YOLO threshold lowering to force “cut”  
- No promotion of wound YOLO candidate (hand FP confirmed)  
- Canonical EffNet overwritten **only** after OOD gate + F1 gates  

---

*End of report.*
