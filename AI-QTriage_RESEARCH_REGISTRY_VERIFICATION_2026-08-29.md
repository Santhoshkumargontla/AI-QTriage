# AI-QTriage Research Mode / Registry Independent Verification Report

**Date:** 2026-08-29  
**Workspace:** `C:\Users\santh\Capstone Project Code`  
**Method:** VERIFY-FIRST → FIX-ONLY-IF-NEEDED → RE-VERIFY  

**Rule:** No hand-edited metrics. Canonical artifacts → backend API → frontend.

---

## J. Final Research Page Verdict

| Question | Answer |
|----------|--------|
| 1. Displayed XGB/VQC metrics mathematically correct? | **Yes** — CM diagonal confirms 25/30 and 16/30; accuracy 0.833333 / 0.533333 |
| 2. Model SHAs match active artifacts? | **Yes** — all six registry SHAs match disk |
| 3. Versions consistent? | **Yes after fix** — status cards use same version as registry (EffNet v1.3.0, U-Net `deduped-subject-v1`) |
| 4. Sample counts clearly labeled? | **Yes after fix** — `train \| val \| test` via `display_sample_count` |
| 5. VQC 16/30 represented in registry row? | **Yes after fix** — was N/A due to key mismatch; now `16 / 30` |
| 6. YOLO wound disclosed unsupported? | **Yes** — research table + `/api/models` wound_note / unsupported_classes |
| 7. U-Net readiness justified? | **Yes** — prior direct probes: blank raw_pos≈0; status READY_FOR_RESEARCH_DEMO kept as **DEMO_WITH_LIMITATIONS** |
| 8. EfficientNet honestly labeled? | **Yes** — NOT_TRUSTWORTHY; metric column states OOD collapse / gates withhold |
| 9. Sensor 28/36 + synthetic provenance? | **Yes** — matches sensor metadata; synthetic_50hz_motion_windows |
| 10. Research page trustworthy description of app? | **Yes after fixes** — for research/demo honesty (not clinical) |

---

## Issue ledger

### RR-01 — VQC registry metric showed N/A while comparison showed 16/30
| Field | Value |
|--------|--------|
| Priority | P0 |
| Status before | Inconsistent |
| Evidence | Registry metrics use `vqc_correct_predictions`; FE only read `correct_predictions` |
| Root cause | Key-name mismatch, not wrong math |
| Fix | Backend `_enrich_registry_entry` aliases + `display_held_out_metric`; FE uses display fields |
| Files | `backend/main.py`, `frontend/app/research/page.tsx`, tests |
| Final status | **VERIFIED_FIXED** |

### RR-02 — Ambiguous “Samples: 140” next to “25 / 30”
| Field | Value |
|--------|--------|
| Evidence | `sample_count` = train_samples only |
| Fix | `display_sample_count` = `train N \| val N \| test N` + footnote |
| Final status | **VERIFIED_FIXED** |

### RR-03 — YOLO mAP50 full float / U-Net Dice N/A / EffNet silent N/A
| Field | Value |
|--------|--------|
| Fix | Format mAP50 to 4 decimals; Dice from `metrics.test.mean_dice`; EffNet shows NOT_TRUSTWORTHY note |
| Final status | **VERIFIED_FIXED** |

### RR-04 — Selective coverage hardcoded “0.0%”
| Field | Value |
|--------|--------|
| Evidence | Canonical held-out has no selective_classification; FE invented 0.0% |
| Fix | API returns `status: not_available`; FE shows `unavailable` |
| Final status | **VERIFIED_FIXED** |

### RR-05 — Twilio “SMS Test” shown when not configured
| Field | Value |
|--------|--------|
| Evidence | `TWILIO_ENABLED=false`, configured=false |
| Fix | Modes text: “Local Simulation only (Twilio SMS not available)” when unconfigured |
| Final status | **VERIFIED_FIXED** |

### RR-06 — Version conflict EffNet/U-Net cards vs table
| Field | Value |
|--------|--------|
| Evidence | Prior `/api/models` could show v1.0.0; registry had v1.3.0 / deduped-subject-v1 |
| Status now | API cards match registry versions + SHA prefixes |
| Final status | **VERIFIED_FIXED** / **ALREADY_FIXED** (SHA exposure from prior phase) |

### RR-07 — Brittle inspect.getsource comparison test
| Field | Value |
|--------|--------|
| Evidence | Flaky failure mid-edit |
| Fix | Behavior assertion via TestClient comparison endpoint |
| Final status | **VERIFIED_FIXED** |

### RR-08 — Held-out metrics themselves
| Field | Value |
|--------|--------|
| Evidence | CM trace XGB=25, VQC=16 on n=30 |
| Fix | None — leave canonical artifact |
| Final status | **ALREADY_FIXED** |

---

## A. Canonical model table

| Model | Path | Full SHA-256 | Version | Dataset | Provenance | Train | Val | Test | Held-out metric | Status |
|-------|------|--------------|---------|---------|------------|------:|----:|-----:|-----------------|--------|
| XGBoost | `ml/models/xgboost_best.json` | `73bb5a5125c3e3907bffa1059165d90ecce9dd4e47ce9ce8f9c1f8937fd3f643` | v1.2.0 | synthetic_multimodal_fusion | SYNTHETIC (200 total; 0 paired clinical) | 140 | 30 | 30 | **25 / 30 (0.833333)** | TRAINED |
| VQC | `ml/models/vqc/vqc_weights.npz` | `2db769bec3abd3c2d4811f8a39c5f3c5e3b41cccdde030a10c056848a9c6389e` | v1.4.0 | synthetic_multimodal_fusion | SYNTHETIC / simulator | 140 | 30 | 30 | **16 / 30 (0.533333)** | EXPERIMENTAL_ONLY |
| EfficientNet | `ml/models/vision/efficientnetv2_injury_best.pt` | `6944605ae71685d909d505d12ce32d5bd9e953c10bbebb72a02635a729897e83` | v1.3.0 | public_wound_dataset (metadata) | SYNTHETIC drawings lineage | 140* | 30* | 30* | NOT_TRUSTWORTHY (OOD) | NOT_TRUSTWORTHY |
| U-Net | `ml/models/vision/unet_injury_best.pt` | `3c7f3f39196d71b9d8d58d1fcfc7438b4ae23d75fbd8195ae7da7b1fcb9660d1` | deduped-subject-v1 | AZH+wseg+Medetec + empties | PUBLIC_REAL + synth empties | — | 69 | 69 | Dice(test) 0.6418 | READY_FOR_RESEARCH_DEMO |
| YOLO11 | `ml/models/vision/yolo11_injury_best.pt` | `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879` | v1.4.0 | yolo_retrain_v2 | SYNTHETIC; wound=0 boxes | 87 img | 20 | 19 | mAP50 0.8358 | INFERENCE_EXECUTES / DEMO_WITH_LIMITATIONS |
| Sensor | `ml/models/sensor_motion_best.json` | `707ee9c25392d5e8e1ca9e023feb77937b6ad5a159cb0b53d3b6a09f047e1cbb` | v1.2.0 | synthetic_50hz_motion_windows | SYNTHETIC | 138 | 26 | 36 | **28 / 36 (0.777778)** | TRAINED |

\* EfficientNet train/val/test counts are from training metadata (historical run). Current `efficientnet_processed` manifest has 126 unique images (cut/bruise only). Status remains NOT_TRUSTWORTHY regardless.

Sidecars verified: VQC scaler `506c818d…`, VQC PCA `16caccdd…`, sensor scaler `04291f24…`.

---

## B. XGBoost / VQC live evaluation

From `data/results/canonical_held_out_evaluation.json` (CM diagonal verification):

| | XGBoost | VQC |
|--|--------:|----:|
| Test n | 30 | 30 |
| Correct | 25 | 16 |
| Accuracy | 0.833333 | 0.533333 |
| Macro P | 0.822222 | 0.466184 |
| Macro R | 0.769281 | 0.432680 |
| Macro F1 | 0.785020 | 0.423016 |
| MCC | 0.702692 | 0.095978 |
| Brier | 0.304532 | 0.512306 |
| ECE | 0.118284 | 0.136455 |
| CM | [[7,3,0],[0,16,1],[0,1,2]] | [[2,8,0],[2,13,2],[0,2,1]] |

Selective classification: **not present** in canonical artifact → API/UI **unavailable** (not fabricated 0%).

---

## C. Sensor evaluation

- Test: 36  
- Correct: 28 / 36  
- Accuracy: 0.777778  
- Provenance: **synthetic_50hz_motion_windows** — not real clinical sensors  

---

## D. YOLO evaluation

- SHA: `4d6e72f5…`  
- Classes in names: cut, bruise, wound  
- Train boxes: cut 47, bruise 40, **wound 0** → wound **UNSUPPORTED**  
- Independent test mAP50: **0.8358** (raw 0.835833…), mAP50-95: **0.7080**  
- Readiness: research demo with limitations; not clinical  

---

## E. EfficientNet

- SHA: `6944605a…`  
- Raw OOD: still collapses (blank→Swelling)  
- Withholding: yes  
- Readiness: **NOT_TRUSTWORTHY**  

---

## F. U-Net

- SHA: `3c7f3f39…`  
- Raw blank: positive ratio ≈ 0 (genuine)  
- Withholding: still present  
- Geometry: previously verified ROI paste  
- Readiness: **READY_FOR_RESEARCH_DEMO** / **DEMO_WITH_LIMITATIONS**  

---

## G. Twilio

- `enabled`: false  
- `configured`: false  
- Status: **NOT_CONFIGURED** / LOCAL SOS SIMULATION ONLY  
- Live SMS: **not available**  

---

## H. Frontend consistency (field → source)

| Field | Source of truth | Hardcoded? |
|-------|-----------------|------------|
| N=30 banner | `/api/evaluation/comparison`.sample_count | No |
| XGB/VQC table metrics | comparison payload ← held-out JSON | No |
| Registry SHA/status/version | `/api/models/registry` ← model_registry.json + live enrich | No |
| Registry metric column | `display_held_out_metric` (server-derived) | No |
| Split counts | `display_sample_count` from metrics train/val/test | No |
| Status cards | `/api/models` | No |
| Twilio badge | `/api/sos/config` | No |
| Disclaimer text | Static research disclaimer (accurate) | Yes (intentional) |
| Selective 0.0% | **Removed** | Was fabricated; now unavailable |

Architecture: **canonical artifacts → backend API → frontend**.

---

## I. Tests

| Suite | Result |
|-------|--------|
| `test_research_registry_consistency.py` | Pass |
| Held-out / comparison regressions | Pass |
| Frontend lint + build | Pass (prior run) |
| Full `pytest backend/tests` | **221 passed** |
| Frontend lint + build | **Pass** |

---

## Files changed

1. `backend/main.py` — registry enrichment; selective_classification honesty; VQC SHA on `/api/models`  
2. `frontend/app/research/page.tsx` — display fields, Twilio wording, YOLO/EffNet notes, selective unavailable  
3. `backend/tests/test_research_registry_consistency.py` — new  
4. `backend/tests/test_remediation_regression.py` — behavior-based comparison test  
5. This report  

**Not changed:** metric numeric values in evaluation artifacts; model weights.

---

*End of Research Mode / Registry verification.*
