# AI-QTriage — Evidence-Based Forensic Remediation Report

**Date:** 2026-08-29 (independent verification pass)  
**Scope:** Full stack inspection vs filesystem, runtime inference, API, MongoDB, frontend build, and pytest  
**Rule:** Prior reports were not trusted; claims below cite current disk SHA / live probes / tests  

**Verdict:** **RESEARCH PROTOTYPE / PARTIALLY WORKING / NOT PRODUCTION READY**

**Overall completion (weighted):** **69%**

---

## A. Executive Verdict

| Label | Chosen |
| --- | --- |
| Executive | **RESEARCH PROTOTYPE** |
| Working state | **PARTIALLY WORKING** |
| Clinical readiness | **NOT PRODUCTION READY** |
| Clinical claims | **BLOCKED** (`clinical_claim_blocked=true`, `SYNTHETIC_RULE_LABELS`, `paired_clinical_samples=0`) |

The application runs end-to-end as a **research demo**: cases create, modalities upload/skip, analyze executes vision + fusion + VQC, Mongo persists, frontend builds and mounts exclusive case tabs. Core vision honesty issues from earlier audits are **mostly gated in the API**, but **EfficientNet production weights remain NOT_TRUSTWORTHY** (raw Swelling collapse on blank gray), and **YOLO wound-box candidate was not promoted** after a forensic-OOD false positive. Fusion labels remain synthetic.

---

## B. Overall Completion Percentage

Weighted score (section 19 weights):

| Feature | Weight | Score /100 | Contribution |
| --- | ---: | ---: | ---: |
| Backend/API | 15% | 85 | 12.75 |
| Frontend/UI | 10% | 80 | 8.00 |
| Database | 5% | 82 | 4.10 |
| YOLO | 20% | 62 | 12.40 |
| EfficientNet | 10% | 45 | 4.50 |
| U-Net | 10% | 72 | 7.20 |
| XGBoost | 8% | 55 | 4.40 |
| VQC | 5% | 50 | 2.50 |
| Sensor | 7% | 70 | 4.90 |
| SOS/Twilio | 5% | 78 | 3.90 |
| Testing/Architecture | 5% | 85 | 4.25 |
| **Total** | **100%** | | **≈ 69** |

Interpretation band: **50–69 = Partially working**.

---

## C. Dependency Map (verified runtime)

```text
Frontend (Next.js)
    ↓ api client
FastAPI (backend/main.py)
    ↓
MongoDB (cases)
    ↓ inputs: image / questionnaire / sensor
POST /api/cases/{id}/analyze
    ↓
YOLO11Detector  → ml/models/vision/yolo11_injury_best.pt
UNetSegmenter   → ml/models/vision/unet_injury_best.pt
EfficientNetV2  → ml/models/vision/efficientnetv2_injury_best.pt (+ Grad-CAM experimental)
Feature fusion (23 dims) → XGBoostClassifier → xgboost_best.json
                     ↘ VQCClassifier (experimental) → ml/models/vqc/*
SensorClassifier (when sensor present)
RulesEngine / first-aid / SOS (separate from ML class)
    ↓
MongoDB fields + CaseResponseSchema
    ↓
Frontend case tabs (exclusive mount)
```

Launch from project root or `backend/` is supported via `canonical_paths.ROOT` / `resolve_existing`.

---

## D. Before vs After (this independent pass)

| Component | Before (verified) | Fix Applied | After | Verified |
| --- | --- | --- | --- | --- |
| `canonical_manifest.json` U-Net SHA | Stale `82419176…` | Regenerated via `generate_registry.py` | `3c7f3f39…` matches disk | Yes |
| EfficientNet classes sidecar | Missing → hardcoded fallback risk | Wrote `efficientnetv2_injury_best_classes.json` = cut/bruise/swelling | Sidecar present | Yes |
| First-aid classifier channel | Fell back to YOLO `finding`/`confidence` | Use only `classifier_finding` / `classifier_probability` | Separation enforced + regression test | Yes |
| Case mask tab badge | Always “U-Net Mask Active” | Shows Withheld when `segmentation_reliable===false` | Code updated | Build OK |
| Case tabs | Exclusive mount already present | No change needed | Still exclusive | `tsc`/build OK |
| YOLO wound candidate | Built; briefly promoted | Reverted after forensic OOD FP @ ~0.25 | Production SHA `4d6e72f5…` | Tests 18+ green |
| EfficientNet promote | Candidate collapse 9→2 ≠ 0 | **KEEP_BASELINE** | Production unchanged `6944605a…` | Live probe |
| U-Net | Deduped candidate blank-collapse 0 | Already promoted (prior pass) | Production `3c7f3f39…` | Live probe + tests |
| Clinical claim | Blocked in analyze | Schema fields on GET + test | Blocked in analyze+GET | E2E TestClient |

---

## E. Every Model

### YOLO11 Detection

| Question | Evidence |
| --- | --- |
| Loads? | Yes — `YOLO11Detector` → `yolo11_injury_best.pt` |
| Inference works? | Yes |
| Actually trained? | Yes (Ultralytics); production = retrain_v2 lineage |
| Dataset source | Production: mostly synthetic drawings + limited photos; candidate `yolo_wound_boxes_v1`: 1262 PUBLIC wound boxes + 126 SYNTHETIC cut/bruise |
| Dataset size (candidate) | n=1388; train 965 / val 209 / test 214; **hash leakage 0** |
| Real or synthetic? | Mixed; wound boxes = public ulcer masks; cut/bruise largely synthetic |
| Data leakage? | Candidate manifest: **leakage_free** |
| Classes | Live `model.names` = **cut, bruise, wound** (SHA `4d6e72f5…`) |
| mAP (candidate partial) | Aggregated test mAP50 ≈ 0.47 (interrupted CPU run) — **not promoted** |
| OOD / blank | Live: gray/white/black → **0 boxes** @ keep 0.25 |
| Forensic OOD | Wound candidate kept box @ ~0.251 → promote **rejected** |
| API / Frontend | `yolo_finding` separate; swelling **not** a YOLO class |
| Trustworthy? | **Research-demo only**; cut/bruise photo scarcity; wound domain ≠ sports injury |
| Needs retraining? | Yes — more real cut/bruise photos + full epochs + broader TN set |

### EfficientNetV2 Classification

| Question | Evidence |
| --- | --- |
| Loads? | Yes — SHA `6944605a…` |
| Inference works? | Yes |
| Actually trained? | Yes historically; **NOT_TRUSTWORTHY** |
| Classes | **cut, bruise, swelling** (sidecar now on disk) |
| Closed-set problem | **Still forced** into 3 injury classes |
| Raw blank behavior | Live gray → **Swelling max≈0.96** |
| Gated API behavior | Live gray → `winner=null`, `LOW_QUALITY_INPUT` |
| Subject-normal retrain | Built `efficientnet_subject_normal` (n=144, leakage-free); best OOD injury-collapse **9→2 ≠ 0** → **KEEP_BASELINE** |
| Trustworthy? | **No** (raw); API-gated for display |
| Needs retraining? | Yes — real normal/healthy photos + collapse→0 before promote |

### U-Net Segmentation

| Question | Evidence |
| --- | --- |
| Loads? | Yes — SHA `3c7f3f39…` (`deduped-subject-v1`) |
| Inference works? | Yes |
| Dataset | `unet_deduped_subject` n=464; PUBLIC 444 + 20 synthetic empty; **leakage_free** |
| Blank/OOD | Live gray: raw positive ratio **0.0**, mask **withheld** |
| Dice (held-out) | ≈ 0.64 on deduped test (training report) |
| Coord mapping | ROI → paste into full image (no full-canvas stretch) |
| Trustworthy? | **Research-demo** ulcer/wound masks; not sports cut/bruise localization |
| Needs retraining? | Optional — domain still chronic-wound |

### Grad-CAM

| Question | Evidence |
| --- | --- |
| Depends on | EfficientNet |
| Status | Experimental; UI “MODEL VISUALIZATION — NOT CLINICAL” |
| Trustworthy? | **No** while EffNet NOT_TRUSTWORTHY |

### XGBoost

| Question | Evidence |
| --- | --- |
| Loads? | Yes — SHA `73bb5a51…`, 23 features |
| Labels | **SYNTHETIC_RULE_LABELS** |
| Clinical claim | **BLOCKED** |
| Trustworthy clinically? | **No** |

### VQC

| Question | Evidence |
| --- | --- |
| Loads? | Yes — experimental |
| Fake 0.33 fallback | Not used; unavailable path preferred |
| Trustworthy clinically? | **No** |

### Sensor classifier

| Question | Evidence |
| --- | --- |
| Loads? | Yes |
| Called? | Yes on simulate/live paths |
| Provenance | Simulated/demo when live logs absent |
| Trustworthy clinically? | **No** |

---

## F. YOLO Detailed Report

| Item | Value |
| --- | --- |
| Production checkpoint | `ml/models/vision/yolo11_injury_best.pt` |
| Production SHA-256 | `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879` |
| Task | `detect` |
| Classes | cut, bruise, wound (`model.names`) |
| Keep threshold | `DEFAULT_YOLO_INFER_CONF = 0.25` (not lowered to invent boxes) |
| Candidate dataset | `data/datasets/yolo_wound_boxes_v1` |
| Candidate boxes | bruise 63, cut 61, **wound 1262** |
| Candidate SHA | `76bccb50…` under `ml/models/yolo_wound_boxes_v1/run_wound_v1/weights/best.pt` |
| Promotion | **KEEP_BASELINE / reverted** — forensic OOD FP at keep threshold |
| Backup retained | `.pre_wound_boxes_v1_backup` |
| Known limitations | Ulcer-domain wound envelopes; cut/bruise still largely drawings; Roboflow key absent for public cut/bruise photos |

**Threshold policy:** Do not auto-use 0.10. Research-demo keep remains **0.25** with TN checks on blank_skin/dummy_test **and** forensic OOD.

---

## G. Dataset Report

| Model | Source | Real/Synthetic | Images | Train | Val | Test | Leakage | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| YOLO prod | retrain_v2 lineage | Mixed / mostly synth cut-bruise | (prod weights) | — | — | — | Prior audit | In production |
| YOLO wound v1 | AZH/Medetec/wseg + retrain_v2 | 1262 real wound + 126 synth | 1388 | 965 | 209 | 214 | **0** | Candidate only |
| EfficientNet subject-normal | drawings + AZH empty crops | 126 synth + 18 real normal | 144 | 100 | 22 | 22 | **0** | Candidate; not promoted |
| U-Net deduped | AZH/wseg/Medetec + empty | 444 real + 20 synth empty | 464 | 326 | 69 | 69 | **0** | **Production** |
| XGB/VQC/Sensor | synthetic multimodal / sim | Synthetic | — | — | — | — | N/A | Research only |

**Public data blockers:** Mendeley healthy-feet empty/403; Roboflow key revoked/absent; HF Healthy_Skin ~1.6GB not downloaded this pass.

---

## H. Bugs

| Bug ID | Severity | Status | Root cause | Fix / Verification | Remaining |
| --- | --- | --- | --- | --- | --- |
| B-MANIFEST-UNET | Medium | **FIXED** | Manifest not regenerated after U-Net promote | `generate_registry.py` → SHA `3c7f3f39…` | — |
| B-EFFNET-SIDECAR | High (latent) | **FIXED** | No classes JSON beside prod weights | Wrote sidecar | Promote of normal-class ckpt still needs matching sidecar |
| B-FA-CLASSIFIER-YOLO | Medium | **FIXED** | First-aid used YOLO finding as classifier | `first_aid_service.py` + `test_classifier_evidence_does_not_inherit_yolo_finding` | — |
| B-MASK-TAB-BADGE | Low | **FIXED** | Always “Mask Active” | Conditional Withheld badge | Browser visual check optional |
| B-TABS | Medium | **FIXED** (prior) | Non-exclusive panels | Exclusive `isTab` mount | — |
| B-CLINICAL-CLAIM | High | **FIXED** | Synthetic fusion presented as clinical | Blocked fields + UI + test | Labels still synthetic |
| B-SOS-SMS-CLAIM | Critical | **FIXED** (prior) | SMS_SENT without Twilio | Honest NOT CONFIGURED / simulation | Live Twilio needs credentials |
| B-EFFNET-SWELLING | Critical (model) | **PARTIALLY FIXED** | Closed-set + bad data | API gates; raw collapse remains | Need real normals + collapse=0 |
| B-UNET-BLANK-PAINT | Critical | **FIXED** | FP on blanks | Withhold + deduped promote | Domain ≠ sports injury |
| B-YOLO-WOUND-SUPPORT | High | **PARTIALLY FIXED** | 0 wound boxes | Dataset now 1262; candidate not promoted | Real cut/bruise photos; TN on forensic OOD |
| B-YOLO-SWELLING-AS-DET | High | **FIXED** | Field confusion | Separated fields; classes from `model.names` | — |
| B-FUSION-REAL-LABELS | High | **NOT FIXED** | 0 paired clinical multimodal records | Honest block only | Needs clinical paired data |
| B-BROWSER-E2E-BBOX | Medium | **NOT FULLY VERIFIED** | Need live browser + multi-image | Build + code path verified; full browser bbox screenshot pass not completed this session | Run browser on cut/bruise/wound/none |

---

## I. Warnings and Errors

| Source | Warning | Class |
| --- | --- | --- |
| pytest / shap | `PendingDeprecationWarning` set_bad/set_over/set_under | Technical debt / harmless for demo |
| Starlette TestClient | httpx deprecation note | Technical debt |
| Ultralytics | Package update available; slow image access on Windows path | Harmless / perf |
| Gemini AFC | Direct `generate_content` AFC not recommended | Technical debt |
| Next.js build | Ignored package-lock outside git root | Config debt |
| EfficientNet HF | Unauthenticated Hub requests during train | Ops / rate limit |

No critical runtime errors in health/analyze smoke.

---

## J. Application Feature Score Card

| Feature | Working % | Bugs | Errors | Warnings | Final Status |
| --- | ---: | --- | --- | --- | --- |
| Backend/API | 85 | Minor | None blocking | Deprecations | Research OK |
| Frontend/UI | 80 | Residual browser E2E | None | Next lock warning | Research OK |
| Database | 82 | — | — | — | Research OK |
| YOLO | 62 | Wound promote blocked | — | Ultralytics | Research demo |
| EfficientNet | 45 | Raw collapse | — | — | **NOT_TRUSTWORTHY** |
| U-Net | 72 | Domain limit | — | — | Research demo |
| XGBoost | 55 | Synthetic labels | — | — | Claim blocked |
| VQC | 50 | Synthetic | — | — | Experimental |
| Sensor | 70 | Sim provenance | — | — | Research demo |
| SOS/Twilio | 78 | Needs creds for live SMS | — | — | Honest offline |
| Testing | 85 | — | — | shap | **201 passed** |

---

## K. Verification Commands (executed)

```text
pytest backend/tests -q
→ 201 passed, 3 warnings in ~95.5s

npm run build (frontend)
→ Compiled successfully; routes /, /cases, /cases/[id], /create-case, /research

Live OOD probes (this pass):
  YOLO gray/white/black → 0 detections
  EfficientNet raw gray → Swelling ~0.96; gated → withheld
  U-Net gray → raw_positive_ratio 0.0; mask withheld

E2E TestClient analyze (no image):
  clinical_claim_blocked=true
  fusion_label_source=SYNTHETIC_RULE_LABELS
```

Artifacts: `scratch/forensic_pytest_2026-08-29.txt`, `scratch/forensic_frontend_build_2026-08-29.txt`.

---

## L. Final Honest Readiness

**READY FOR FURTHER VALIDATION** as a **RESEARCH DEMO / RESEARCH PROTOTYPE**.  
**NOT PRODUCTION READY. NOT CLINICAL.**

### What works
- FastAPI + Mongo case lifecycle; analyze pipeline executes.
- YOLO loads canonical weights; classes = `model.names`; no swelling detections.
- U-Net blank paint gated/cleared on current weights; ROI mask composition.
- EfficientNet API withholding on blank/OOD despite raw collapse.
- Clinical claim blocking; SOS does not fake SMS delivery.
- Exclusive case-page tabs; frontend production build succeeds.
- **201** backend tests green after this pass’s regression additions.

### What is reliable (research sense)
- Artifact path discipline via `canonical_paths` + regenerated registry/manifest.
- Leakage-free **processed** manifests for wound-YOLO candidate, EffNet subject-normal, U-Net deduped.
- Honest provenance labels (SYNTHETIC / PUBLIC_REAL_PHOTOS).

### What is not reliable
- EfficientNet injury labels (raw Swelling collapse).
- YOLO cut/bruise photo performance / wound-candidate promote readiness.
- XGBoost/VQC severity (synthetic rule labels).
- Grad-CAM as explanation.
- Any clinical triage claim.

### Could not be fixed without external inputs
| Need | Why blocked |
| --- | --- |
| Real cut/bruise labeled photos | No valid Roboflow/API key; no substitute invented |
| Healthy-feet / true normal photos at scale | Mendeley empty/403; large HF set not fetched |
| Paired clinical multimodal fusion labels | `paired_clinical_samples=0` |
| Live Twilio delivery proof | Requires configured credentials + real send |
| Full browser bbox multi-image evidence pack | Not completed in this session (build + code verified) |

### Do not do
- Do not lower YOLO conf to manufacture detections.
- Do not remap ulcer→swelling.
- Do not promote EffNet until OOD injury-collapse = 0 with matching class sidecar.
- Do not claim clinical accuracy.

---

*Report generated from independent filesystem, runtime, API, and test evidence on 2026-08-29.*
