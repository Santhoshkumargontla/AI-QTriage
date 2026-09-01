# AI-QTriage Complete Forensic Analysis, Testing, Fixing & Re-Verification

**Date:** 2026-08-29  
**Workspace:** `C:\Users\santh\Capstone Project Code`  
**Evidence class:** Independently re-verified from current repository + runtime (API TestClient, live uvicorn, MongoDB, Next.js browser, direct model probes). Prior reports were consulted only as leads, not as truth.

---

## 1. Executive Verdict

| Label | Applies? |
| --- | --- |
| **WORKING_RESEARCH_PROTOTYPE** | **YES** — create/upload/questionnaire/sensor/analyze/store/retrieve works |
| **READY_FOR_RESEARCH_DEMO** | **YES — WITH MAJOR LIMITATIONS** |
| PARTIALLY_WORKING | Also true for vision cut recognition on real photos |
| BROKEN | **No** (core pipeline executes) |
| READY_FOR_PRODUCTION | **NO** |
| CLINICALLY_READY | **NO** |

**One-line verdict:** The application is a functioning multimodal research prototype. EfficientNet OOD/Swelling collapse was honestly mitigated by reject-v2 promotion. YOLO remains synthetic-data-limited and mislabels many real injuries. Fusion/XGB/VQC/sensor are synthetic. Twilio is local simulation only. Not a medical device.

---

## 2. Complete Architecture (Actual Runtime Sequence)

Verified by tracing `backend/main.py` → `analyze_case` and wrappers (not comments alone).

```
Frontend (Next.js :3000)
    ↓ REST
FastAPI (backend/main.py :8000)
    ↓
Create Case → Upload Image (+ quality gate) → Questionnaire → Sensor (demo/simulate/upload)
    ↓ POST /api/cases/{id}/analyze
Vision branch:
  1. YOLO11 detect on full image (conf keep=0.25)
  2. If box: U-Net segment on ROI; else U-Net on full image (quality-gated)
  3. EfficientNetV2 classify on ROI (or full) with quality + abstention gates
  4. Grad-CAM only if EffNet injury-confident (else WITHHELD)
Fusion branch:
  5. Feature fusion (23-D vector; missing → explicit / zeros per schema — not invented clinical values for SOS)
  6. Rules engine (safety guidance)
  7. XGBoost severity (main decision input among models)
  8. VQC (PennyLane sim) — EXPERIMENTAL_ONLY, used_in_main_decision=false
  9. SOS eligibility / local demo hooks
    ↓
MongoDB case document
    ↓
GET /api/cases/{id} → Frontend tabs (overview / image / questionnaire / sensor / AI / explainability / sos / report)
```

### Component contracts

| Component | Input | Processing | Output | DB fields | Failure / fallback | Real vs simulated |
| --- | --- | --- | --- | --- | --- | --- |
| Case create | title/meta | UUID case | case_id | cases.* | 4xx | Real |
| Upload | image bytes | quality gates (blur/contrast/dark) | path + SHA-256 | image_reference, image_sha256 | **422** reject | Real I/O |
| Questionnaire | answers | template routing | answers dict | questionnaire.* | empty allowed with flags | Real |
| Sensor | CSV / simulate / demo | feature extract + motion XGB | peak_g, class, probs | sensor_summary.* | FEATURE_MISSING explicit | **Simulated demos common** |
| YOLO | RGB image | Ultralytics detect | class, conf, xyxy | visible_injury.yolo_* | empty list | Real weights, **synthetic train** |
| U-Net | ROI/full | sigmoid mask + sanity | mask, ratio, reliability | mask_*, segmentation_* | withhold empty/FP | Real weights, wound-domain |
| EfficientNet | ROI/full | softmax + gates | finding or withhold | classifier_* | OOD / LOW_QUALITY | Real weights, **reject-v2** |
| Grad-CAM | EffNet tensors | hooks overlay | overlay path or WITHHELD | gradcam_* | withhold if untrusted | Visualization only |
| Fusion | multimodal features | 23-D schema | vector | (embedded in preds) | zeros / missing flags | Engineering |
| Rules | answers + vision | heuristics | guidance | first_aid / report fields | conservative text | Rule-based |
| XGBoost | 23-D | severity class | LOW/MOD/HIGH | xgboost_prediction | clinical_claim_blocked | **SYNTHETIC labels** |
| VQC | PCA’d features | 4-qubit sim | probs | quantum_prediction | structured fail; no 1/3 fake | **SYNTHETIC / EXPERIMENTAL** |
| SOS | trigger | countdown | LOCAL_SIMULATION | sos events | Twilio off → local | **Simulated unless Twilio on** |
| MongoDB | case docs | CRUD | persistence | full case | health shows connected | Real local DB |

---

## 3. Every Model — Forensic Summary

### 3.1 YOLO11

| Field | Verified value |
| --- | --- |
| Path | `ml/models/vision/yolo11_injury_best.pt` |
| SHA-256 | `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879` |
| Classes (`model.names`) | cut, bruise, wound |
| Wound boxes in production train | **0 → UNSUPPORTED** |
| Conf / IoU | keep **0.25** (not raised to force cuts) |
| Dataset | `yolo_retrain_v2` — **SYNTHETIC** drawings (126/126), tiny held-out |
| Test mAP50 (tiny synth test) | ~0.84 (n≈19 images) — **not real-world proof** |
| Trust | **NOT_TRUSTWORTHY for real clinical photos**; RESEARCH_DEMO_LIMITED for synth bruise demos |
| Failure modes | Misses real cuts; bruise FP on wound-like; wrong location on real hand laceration (prior case `ac69884f-…`) |

**Independent probe highlights (2026-08-29 forensic suite):**

| Input | Expected | Actual YOLO | Correct? |
| --- | --- | --- | --- |
| synth_bruise | bruise-ish hit OK for synth | bruise @ 0.975 box [174,131,306,229] | Synth-domain OK |
| synth_wound | bruise/wound confusion likely | bruise @ 0.953 | Incorrect class (wound unsupported) |
| synth_cut (low contrast) | may miss | upload **422** / probe often miss | Domain fail |
| blank/black/white/gray | none | [] | Correct |
| unrelated / noise | none | [] | Correct |
| real hand cut (prior E2E) | cut near laceration | bruise ~0.76 wrong site | **Incorrect** — model error, not letterbox UI |

**Box pipeline consistency (synth bruise case `2c773af0-…`):** Direct model = API = Mongo = GET case = **same** bbox/conf. Frontend scales by `overlay_width/height` (480×360) — coordinate system consistent for this case (**verified** API+Mongo; browser showed case detail + image tab UI).

---

### 3.2 EfficientNetV2 (reject-v2)

| Field | Verified value |
| --- | --- |
| Path | `ml/models/vision/efficientnetv2_injury_best.pt` |
| SHA-256 | `fa7aa5c822e6a4127d7f1aa9d9518e687cc0ee9e3d8cfd362d902f72ac06568c` |
| Classes | **cut, bruise, normal, ood_reject** (swelling **removed**) |
| Prior collapse | Swelling @ ~0.96–1.0 on blanks (**fixed by retrain+promote**, not by hiding alone) |
| Held-out test | n=28; acc≈0.929; macro-F1=**0.875**; OOD injury collapse **0** (was 10) |
| Trust | **READY_FOR_RESEARCH_DEMO**; cut/bruise still mostly **SYNTHETIC drawings**; not clinical |
| Blank raw (post) | black→Normal; white/gray→OOD_Reject — **not Swelling** (**verified** probes) |

Gates withhold blanks as `LOW_QUALITY_INPUT`. Gates ≠ model quality; raw probs also no longer Swelling-collapse.

---

### 3.3 U-Net (ResNet34)

| Field | Verified value |
| --- | --- |
| Path | `ml/models/vision/unet_injury_best.pt` |
| SHA-256 | `3c7f3f39196d71b9d8d58d1fcfc7438b4ae23d75fbd8195ae7da7b1fcb9660d1` |
| Domain | Wound/ulcer research segmentation — **not** cut/bruise localizer |
| Blanks | raw_pos_ratio ≈ 0; withheld |
| Trust | READY_FOR_RESEARCH_DEMO / DEMO_WITH_LIMITATIONS |
| Failure | Can mark tiny FP ratios on some synth; ROI geometry previously verified `pos_outside_roi=0` |

---

### 3.4 Grad-CAM

| Field | Verified value |
| --- | --- |
| Source | EfficientNetV2 |
| Behavior | WITHHELD when classifier not injury-confident |
| Trust | Visualization only; **not** clinical explanation when base model untrusted |

Case `2c773af0-…`: `gradcam_explanation_status=WITHHELD`, reason `softmax_not_trusted_closed_set_overconfidence`.

---

### 3.5 XGBoost

| Field | Verified value |
| --- | --- |
| Path | `ml/models/xgboost_best.json` |
| SHA-256 | `73bb5a5125c3e3907bffa1059165d90ecce9dd4e47ce9ce8f9c1f8937fd3f643` |
| Features | 23-D multimodal |
| Labels | **SYNTHETIC_RULE_LABELS** |
| Held-out | 25/30 ≈ 0.833 on synthetic test |
| Clinical | `BLOCKED_NO_PAIRED_CLINICAL_LABELS` |
| Note | SHAP still surfaces legacy `prob_swelling` feature name in fusion schema |

---

### 3.6 VQC

| Field | Verified value |
| --- | --- |
| Path | `ml/models/vqc/vqc_weights.npz` |
| SHA-256 | `2db769bec3abd3c2d4811f8a39c5f3c5e3b41cccdde030a10c056848a9c6389e` |
| Status | **EXPERIMENTAL_ONLY**; `used_in_main_decision: false` |
| Held-out | 16/30 ≈ 0.533 synthetic |
| Fallback | No silent `[0.33,0.33,0.33]` observed in tested cases; structured experimental flag |

---

### 3.7 Sensor motion classifier

| Field | Verified value |
| --- | --- |
| Path | `ml/models/sensor_motion_best.json` |
| SHA-256 | `707ee9c25392d5e8e1ca9e023feb77937b6ad5a159cb0b53d3b6a09f047e1cbb` |
| Provenance | **SYNTHETIC** windows (not downloaded SisFall/UCI) |
| Known limit | `football_fall` scenario can still classify **normal_activity** @ high confidence (case `2c773af0-…`: peak 4.62g → normal_activity 0.986) |
| Missing features | Explicit `FEATURE_MISSING` path (not invented lux) |

---

## 4. Complete End-to-End Tests (Independent Forensic Suite)

Evidence files:

- `scratch/forensic_suite_2026_08_29/forensic_model_probes.json`
- `scratch/forensic_suite_2026_08_29/e2e_cases.json`
- `scratch/forensic_suite_2026_08_29/e2e_cut_hq.json`
- Live case browse: `http://localhost:3000/cases/2c773af0-3575-4e70-93fc-b05b0008f572`
- Research page browse: `http://localhost:3000/research`

### Upload / analyze matrix

| Tag | Expected | Actual | Correct? | Severity |
| --- | --- | --- | --- | --- |
| synth_bruise | Upload+analyze; YOLO may hit bruise | YOLO Bruise 0.975; EffNet withheld; XGB MODERATE; SHA chain match | Pipeline **OK**; medical accuracy **research-only** | — |
| blank_black | Reject or no injury | Upload **422** dark | Correct gate | — |
| normal_skin (flat) | Reject low contrast | Upload **422** | Correct gate | — |
| unrelated blur | Reject | Upload **422** | Correct gate | — |
| synth_cut low-contrast | May reject | Upload **422** contrast | Gate blocks weak synth | Medium for demo |
| HQ cut `e9462285-…` | Prefer cut detection | YOLO **miss**; EffNet not confident; U-Net small reliable; XGB HIGH from Q/sensor | Vision miss — **YOLO/data** | High (model) |
| Real hand cut (prior) | Cut on laceration | Bruise wrong location | **Incorrect** | Critical (model) |

### Consistency chain (bruise case)

`Direct YOLO → analyze API → Mongo → GET case`: SHA `4f8458b4…`, box and conf identical. Frontend Topbar **Case Detail**; overview separates YOLO Bruise vs classifier **Withheld** (**browser verified**).

### SOS

| Scenario | Result | Evidence |
| --- | --- | --- |
| Twilio disabled | `TWILIO_ENABLED=false`, configured=false | `/api/sos/config` |
| Local demo trigger | countdown; `delivery_mode=local_demo` | e2e_cases.json |
| Status poll | `LOCAL_SIMULATION`; `sms_sent=false`; no SID | `/api/cases/.../sos/status` |
| SMS delivered | **NOT CLAIMED** | No provider DELIVERED |

### Browser

| Check | Result |
| --- | --- |
| Cases list / case detail load | **Verified** |
| Topbar title follows route | **Fixed + verified** (“Case Detail”, “Research Results”) |
| Research registry SHA prefixes | YOLO `4d6e72…`, EffNet `fa7aa5…`, U-Net `3c7f3f…`, XGB `73bb5a…`, VQC `2db769…` |
| YOLO vs EffNet confusion | Separated in UI (YOLO label vs withheld classifier) |
| Hydration warnings | Research page still can show Next hydration overlay (partial mitigation applied); **not fully eliminated** |

---

## 5. Before / After Fixes

| Bug ID | Severity | Root Cause | Files | Fix | Verification | Status |
| --- | --- | --- | --- | --- | --- | --- |
| EFF-SWELL-COLLAPSE | Critical | Majority/OOD collapse to Swelling | EffNet train reject-v2 + promote | Retrain 4-class head; remove swelling; promote on OOD gates | Probes: blanks ≠ Swelling; collapse 10→0 | **VERIFIED_FIXED** (research scope) |
| EFF-GATES-ONLY | Critical | Withholding mistaken for model fix | wrappers + report discipline | Raw+gated both audited | Raw probs checked | **VERIFIED_FIXED** (process) |
| YOLO-WOUND-ZERO | High | nc=3 with 0 wound labels | registry + API notes | Mark UNSUPPORTED; no wound accuracy claim | `/api/models` | **PARTIALLY_FIXED** (honest labeling) |
| YOLO-REAL-CUT | Critical | Synthetic train / domain shift | — | Data acquisition blocked (no Roboflow/Kaggle keys) | Real image E2E | **REQUIRES_RETRAINING** / more data |
| PROVENANCE-DEMO | High | User upload labeled synthetic demo | `main.py` `_apply_vision_image_provenance` | Upload SHA + provenance | Cases show uploaded | **VERIFIED_FIXED** (prior cycle) |
| TOPBAR-TITLE | Medium | Default “New Assessment” always | `Topbar.tsx` | Pathname-derived title | Browser Case Detail / Research | **VERIFIED_FIXED** |
| SIDEBAR-ACTIVE | Medium | Prefix match / hydration | `Sidebar.tsx` | Safer `navIsActive` | Lint pass; My Cases active on detail | **PARTIALLY_FIXED** |
| RESEARCH-HYDRATE | Medium | Client fetch vs SSR text | `research/page.tsx` | suppressHydrationWarning + loading ellipsis | Overlay may remain until hard refresh | **PARTIALLY_FIXED** |
| SWELLING-NOTE-STALE | Low | Stale uvicorn / old note | `main.py` swelling_note | Updated copy for reject-v2 | Requires live process restart | **PARTIALLY_FIXED** |
| SENSOR-FALL-DEMO | High | Synthetic classifier | sensor model | Document; no fake labels | football_fall→normal_activity | **NOT_FIXED** (data) |
| VQC-IN-DECISION | High (trust) | Could mislead | VQC flags | `used_in_main_decision=false` | Case JSON | **VERIFIED_FIXED** (gating) |
| CLINICAL-CLAIM | Critical | Fake clinical validity | analyze path | `BLOCKED_NO_PAIRED_CLINICAL_LABELS` | All E2E cases | **VERIFIED_FIXED** |

---

## 6. Accuracy (Honest, Sample-Size Aware)

| Model | Metric | n | Value | Interpretation |
| --- | --- | --- | --- | --- |
| YOLO | mAP50 synth test | ~19 imgs | ~0.84 | **Tiny synthetic** — not clinical accuracy |
| YOLO | Real hand cut localization | 1 forensic photo | Fail | Domain failure |
| EfficientNet | Test macro-F1 | 28 | 0.875 | Research held-out; drawings-heavy |
| EfficientNet | OOD injury collapse | stress set | 0 (was 10) | Improved |
| U-Net | Blank FP ratio | blanks | ~0 | Good withhold |
| XGBoost | Synth held-out acc | 30 | ~0.833 | Synthetic only |
| VQC | Synth held-out acc | 30 | ~0.533 | Experimental |
| Sensor | Synth held-out | 36 | 28/36 | Synthetic; scenario mismatch remains |

**Do not equate `231 passed` pytest with model accuracy.**

---

## 7. Remaining Problems (Honest)

1. YOLO trained almost entirely on synthetic drawings; wound class empty; real lacerations misclassified.  
2. EfficientNet cut/bruise still drawing-dominated; swelling unsupported (no labels).  
3. U-Net is wound-domain, not general trauma.  
4. XGB/VQC/sensor fusion synthetic; paired clinical samples = **0**.  
5. Sensor `football_fall` demo can read as `normal_activity`.  
6. Twilio not configured — only `LOCAL_SIMULATION`.  
7. External data (Roboflow/Kaggle/Mendeley) blocked this campaign.  
8. Frontend hydration warnings not fully gone on Research.  
9. Legacy SHAP feature name `prob_swelling` in fusion schema.  
10. Research page “N=30” is fusion comparison samples, not vision photo count.

---

## 8. What Was Actually Fixed (This Campaign + Verified Prior)

- EfficientNet reject-v2 trained, evaluated, promoted; Swelling head removed; OOD collapse reduced.  
- Upload quality gates reject blanks (422).  
- Classifier abstention + Grad-CAM withhold for untrusted softmax.  
- Clinical claim blocked; VQC excluded from main decision.  
- YOLO wound honesty (UNSUPPORTED); swelling notes updated in code.  
- Provenance for user uploads vs demo sensor.  
- Topbar route titles.  
- Sidebar active-path matching hardened.  
- Research N= line hydration mitigation.  
- Backend tests: **231 passed** (2026-08-29).  
- Frontend: **lint clean** after Topbar/Sidebar fix; **build succeeded**.

---

## 9. What Could Not Be Fixed

| Item | Why |
| --- | --- |
| YOLO real-photo cut accuracy | No licensed large cut/bruise photo set (Roboflow/Kaggle keys absent; Mendeley 403) |
| Wound YOLO class | Zero honest boxes — promoting wound candidate previously caused OOD FP; reverted |
| Clinical readiness | No paired clinical labels |
| Real SMS delivery | Twilio disabled / unconfigured |
| Sensor fall realism | Only synthetic windows |
| Perfect browser hydration | Residual Next client-fetch overlay on Research |

---

## 10. Final Scores (Evidence-Based)

### Feature scorecard

| Feature | Before | After | Working | Bugs | Accuracy* | Reliability | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frontend | 70 | 78 | Yes | Hydration residual | n/a | Med | **78** |
| Backend | 82 | 86 | Yes | Stale-process risk | n/a | High | **86** |
| MongoDB | 85 | 88 | Yes | — | n/a | High | **88** |
| Case Creation | 90 | 92 | Yes | — | n/a | High | **92** |
| Image Upload | 80 | 88 | Yes | Strict gates | n/a | High | **88** |
| Image Quality | 75 | 90 | Yes | Blocks weak demos | n/a | High | **90** |
| YOLO | 45 | 48 | Partial | Domain fail | Tiny synth ≠ real | Low-Med | **48** |
| EfficientNetV2 | 25 | 72 | Yes (research) | Drawing bias | 0.875 F1 n=28 | Med | **72** |
| U-Net | 55 | 68 | Yes (domain) | Wrong domain for cuts | Research | Med | **68** |
| Grad-CAM | 50 | 70 | Yes (gated) | Meaningless if abstain | n/a | Med | **70** |
| XGBoost | 60 | 62 | Yes | Synthetic | 0.83 synth | Med | **62** |
| VQC | 40 | 55 | Experimental | Weak synth | 0.53 | Low | **55** |
| Sensor Analysis | 55 | 58 | Partial | Fall miss | Synth | Low-Med | **58** |
| Questionnaire | 80 | 85 | Yes | Routing OK w/ gates | n/a | High | **85** |
| Feature Fusion | 65 | 70 | Yes | Legacy swelling feat | Synth | Med | **70** |
| SOS | 70 | 80 | Local sim | No real SMS | n/a | High (honest) | **80** |
| Twilio | 20 | 35 | Disabled | Unconfigured | n/a | Honest fail | **35** |
| Reports | 75 | 80 | Yes | — | n/a | Med | **80** |
| Model Registry | 60 | 85 | Yes | Some stale flags | n/a | High | **85** |
| Research Metrics | 50 | 82 | Yes | Small n disclosed | Synth disclosed | High honesty | **82** |
| End-to-End Workflow | 70 | 84 | Yes | Vision accuracy | Mixed | Med-High | **84** |
| Testing | 80 | 90 | Yes | ≠ accuracy | 231 pytest | High | **90** |
| Data Quality | 30 | 45 | Partial | Synth dominant | — | Low | **45** |
| Architecture | 75 | 88 | Yes | Clear gating | — | High | **88** |

\*Accuracy column is **not** clinical accuracy.

### Overall ratings

| Axis | Score | Band |
| --- | --- | --- |
| Code Functionality | **86** | Strong research software |
| Model Accuracy | **52** | Working research with major limits |
| Model Reliability (OOD/honest fail) | **64** | Improved EffNet; YOLO weak |
| Dataset Quality | **42** | Partially usable / synthetic-heavy |
| Frontend | **78** | Good demo UI |
| Backend | **86** | Solid API |
| Integration | **84** | E2E works |
| Research Prototype Readiness | **74** | **GOOD RESEARCH PROTOTYPE** |
| Production Readiness | **18** | BROKEN for production |
| Clinical Readiness | **5** | BROKEN for clinical use |

### Weighted overall

```
Overall = 0.20*CodeFunc + 0.15*ModelAcc + 0.15*ModelRel + 0.10*DataQual
        + 0.10*Frontend + 0.10*Backend + 0.10*Integration
        + 0.07*ResearchReady + 0.02*Production + 0.01*Clinical
```

| | Value |
| --- | --- |
| Before repair (approx., EffNet collapse era) | **~48%** |
| **After repair (this verification)** | **~68%** |
| Label | **61–75% GOOD RESEARCH PROTOTYPE** |

Formula weights emphasize that production/clinical axes correctly drag the score down; pytest-green alone cannot push above research-demo territory.

---

## Evidence Index

| Artifact | Path |
| --- | --- |
| Model probes | `scratch/forensic_suite_2026_08_29/forensic_model_probes.json` |
| E2E cases | `scratch/forensic_suite_2026_08_29/e2e_cases.json` |
| HQ cut case | `scratch/forensic_suite_2026_08_29/e2e_cut_hq.json` |
| Live models snapshot | `scratch/forensic_api_models_live.json` |
| Prior real-image forensic | `AI-QTriage_REAL_IMAGE_END_TO_END_FORENSIC_VERIFICATION_2026-08-29.md` |
| Data/retrain campaign | `AI-QTriage_DATA_ACQUISITION_RETRAIN_FORENSIC_2026-08-29.md` |
| Pytest | **231 passed**, 3 shap PendingDeprecationWarnings |

---

## Absolute Statements

- **Verified evidence:** hashes, probe JSON, E2E Mongo/API equality, browser Topbar/Research SHA prefixes, SOS LOCAL_SIMULATION, pytest/lint/build.  
- **Inference:** YOLO failure on real cuts is primarily domain/data, not frontend CSS (supported by prior coordinate forensics).  
- **Assumption:** Local Mongo used by TestClient and uvicorn is the same research DB.  
- **Unverified / not claimed:** Real SMS DELIVERED; clinical diagnostic accuracy; production scalability; complete elimination of all Next hydration overlays.

**Final stamp:** `WORKING_RESEARCH_PROTOTYPE` + `READY_FOR_RESEARCH_DEMO_WITH_MAJOR_LIMITATIONS` + `NOT_READY_FOR_PRODUCTION` + `NOT_CLINICALLY_READY`.
