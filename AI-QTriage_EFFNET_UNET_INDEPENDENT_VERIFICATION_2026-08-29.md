# AI-QTriage EfficientNet + U-Net Independent Verification Report

**Date:** 2026-08-29  
**Workspace:** `C:\Users\santh\Capstone Project Code`  
**Method:** VERIFY-FIRST → FIX-ONLY-IF-NEEDED → RE-VERIFY  
**Scope:** EfficientNet reliability/OOD/withholding/API-UI; U-Net reliability/OOD/geometry/withholding; Grad-CAM dependency; provenance honesty.

**Hard rules observed:** A withholding gate is not a model fix. Raw blank collapse ≠ acceptable. No clinical claim. YOLO not modified.

---

## K. Final verdict

| System | Verdict |
|--------|---------|
| **EfficientNet** | **NOT_TRUSTWORTHY** — raw black/white/gray still collapse to high-confidence injury (usually Swelling). Gates withhold blanks. Football can still pass as VALID Swelling. **REQUIRES_MORE_DATA** / **REQUIRES_RETRAINING** when real normals exist. |
| **U-Net** | **DEMO_WITH_LIMITATIONS** / **READY_FOR_RESEARCH_DEMO** — blank/gray **raw** positive ratio ≈ 0 (genuine model improvement, not display-only). Geometry paste keeps positives inside ROI. Still **not clinical**; not cut/bruise localization. |
| **Application** | **RESEARCH / DEMO ONLY** — clinical claims remain blocked. |

**SAFETY GATES WORKING WHERE DESIGNED — EFFICIENTNET UNDERLYING MODEL STILL NOT TRUSTWORTHY.**  
**U-NET BLANK RAW COLLAPSE CLEARED; DISPLAY WITHHOLDING ALSO IN PLACE.**

No EfficientNet or U-Net weight changes in this phase. One API honesty fix applied (SHA/path exposure).

---

## Issue ledger

### EU-01 — EfficientNet canonical artifact four-way match
| Field | Value |
|--------|--------|
| Priority | P0 |
| Status before | Reported SHA `6944605a…` / NOT_TRUSTWORTHY |
| Evidence | Disk = wrapper = registry = manifest = metadata |
| Already fixed? | Yes |
| Fix applied | None |
| Final status | **ALREADY_FIXED** (identity); quality **NOT_TRUSTWORTHY** |

### EU-02 — EfficientNet raw OOD collapse
| Field | Value |
|--------|--------|
| Priority | P0 |
| Evidence | Direct `predict_raw`: black/white→Swelling@1.0; gray_180→Swelling@0.9624 |
| Root cause | Closed-set injury head; missing normals; synthetic bias |
| Fix | None (no suitable retrain data for promotion) |
| Final status | **STILL_BROKEN** / **NOT_TRUSTWORTHY** / **REQUIRES_MORE_DATA** |

### EU-03 — EfficientNet application withholding
| Field | Value |
|--------|--------|
| Priority | P0 |
| Evidence | Blanks → `LOW_QUALITY_INPUT`, winner=None; Grad-CAM withheld for NOT_TRUSTWORTHY |
| Note | Gate hides known failure; football VALID Swelling remains |
| Final status | **ALREADY_FIXED** (gate) — not a model fix |

### EU-04 — U-Net canonical artifact four-way match
| Field | Value |
|--------|--------|
| Priority | P0 |
| Evidence | SHA `3c7f3f39…` matches disk/wrapper/registry/manifest/metadata |
| Status field | READY_FOR_RESEARCH_DEMO |
| Final status | **ALREADY_FIXED** (identity) |

### EU-05 — U-Net blank/OOD raw behavior
| Field | Value |
|--------|--------|
| Priority | P0 |
| Evidence | black/white/gray/uniform_skin **raw_pos_ratio=0.0**, raw_mean≈0.002; display withheld |
| Distinction | MODEL_OUTPUT cleared on blanks; DISPLAY also empty |
| Final status | **ALREADY_FIXED** (raw blank collapse) — still **DEMO_WITH_LIMITATIONS** |

### EU-06 — U-Net mask geometry
| Field | Value |
|--------|--------|
| Priority | P0 |
| Evidence | square/portrait/landscape + public ROI paste: **pos_outside_roi=0**, inside>0 when VALID |
| Final status | **ALREADY_FIXED** |

### EU-07 — `/api/models` missing EffNet/U-Net SHA
| Field | Value |
|--------|--------|
| Priority | P1 |
| Evidence | YOLO/XGB exposed `artifact_sha256`; EffNet/U-Net did not (`None`) |
| Root cause | Incomplete `/api/models` payload |
| Fix | Expose canonical_path, artifact_sha256, version, classes |
| Files | `backend/main.py`, `backend/tests/test_efficientnet_reliability.py` |
| Verification | TestClient → SHA matches disk |
| Final status | **VERIFIED_FIXED** |

### EU-08 — Upload blur gate vs direct blank probes
| Field | Value |
|--------|--------|
| Priority | P2 |
| Evidence | Pure gray / blank_skin rejected at upload (variance 0 &lt; 12) before analyze |
| Note | Direct wrapper probes still required for model audit; not a defect |
| Final status | **ALREADY_FIXED** (extra upload safety) |

### EU-09 — Grad-CAM on NOT_TRUSTWORTHY EffNet
| Field | Value |
|--------|--------|
| Evidence | Football: Grad-CAM WITHHELD `classifier_model_not_trustworthy` |
| Final status | **ALREADY_FIXED** |

---

## A. EfficientNet artifact verification

| Item | Value |
|------|--------|
| Canonical path | `ml/models/vision/efficientnetv2_injury_best.pt` |
| SHA-256 | `6944605ae71685d909d505d12ce32d5bd9e953c10bbebb72a02635a729897e83` |
| Wrapper path | Same absolute path; classes `cut/bruise/swelling` |
| Registry match | Yes |
| Manifest match | Yes |
| Metadata match | Yes |
| `/api/models` match | Yes **after EU-07 fix** |
| Sidecar | `efficientnetv2_injury_best_classes.json` |
| Fallbacks | Canonical-only load; candidates not auto-loaded |
| Status | **NOT_TRUSTWORTHY** |

Preprocess: RGB → resize 224 → /255 → ImageNet mean/std → softmax (T=1.5 gated).

---

## B. EfficientNet raw reliability matrix

| Image | Raw top | Raw conf | Final gated/API-visible | Withheld? | Reason |
|-------|---------|----------|-------------------------|-----------|--------|
| black | Swelling | 1.0 | None | Yes | uniform_or_blank_image |
| white | Swelling | 1.0 | None | Yes | uniform_or_blank_image |
| gray_128 | Swelling | 0.963 | None | Yes | uniform_or_blank_image |
| gray_180 | Swelling | 0.962 | None | Yes | uniform_or_blank_image |
| uniform_skin | Swelling | 1.0 | None | Yes | uniform_or_blank_image |
| noise | Swelling | 0.994 | None | Yes | high_frequency_unstructured |
| blur_lowtex | Swelling | 0.995 | None | Yes | uniform_or_blank_image |
| blank_skin | Swelling | 0.989 | None | Yes | uniform_or_blank_image |
| forensic_ood | Swelling | 0.962 | None | Yes | uniform_or_blank_image |
| football | Swelling | 0.999 | **Swelling VALID** | No | OOD_OVERCONFIDENCE passes gate |
| eff_bruise (synth test) | Swelling | 0.986 | Swelling VALID | No | Wrong class vs bruise label |

Probe artifact: `scratch/effnet_unet_phase_verify.json`

---

## C. EfficientNet training/data audit

| Item | Finding |
|------|---------|
| Dataset (`efficientnet_processed`) | 126 images, **100% SYNTHETIC** |
| Classes present | cut, bruise only (**swelling=0** in current manifest) |
| Split | cut train/val/test 44/9/9; bruise 44/10/10 |
| Hash overlap | train↔val/test = **0** |
| Subject IDs | None → `SUBJECT_LEAKAGE_STATUS_UNKNOWN` |
| Negatives / healthy | Not in promoted set |
| Subject-normal candidate | 18 PUBLIC_REAL normals; **not promoted** (prior OOD gate fail) |
| Retraining | **REQUIRES_MORE_DATA** — do not retrain on synthetic clones |

---

## D. U-Net artifact verification

| Item | Value |
|------|--------|
| Canonical path | `ml/models/vision/unet_injury_best.pt` |
| SHA-256 | `3c7f3f39196d71b9d8d58d1fcfc7438b4ae23d75fbd8195ae7da7b1fcb9660d1` |
| Registry/manifest/metadata | Match |
| `/api/models` | Match after EU-07 |
| Status | READY_FOR_RESEARCH_DEMO |
| Provenance | AZH+wseg+Medetec subject-aware + synthetic empties |
| Metrics (metadata) | test Dice≈0.64, ood_collapse=0 |
| Fallbacks | Canonical-only |

Preprocess: ROI/full RGB → resize 256 → ImageNet norm → sigmoid → threshold 0.5 → sanity/quality gates.

---

## E. U-Net raw reliability matrix

| Image | Raw pos ratio | Raw mean | Display px | Status | Withheld? |
|-------|---------------|----------|------------|--------|-----------|
| black | **0.0** | 0.0019 | 0 | LOW_QUALITY_INPUT | Yes |
| white | **0.0** | 0.0039 | 0 | LOW_QUALITY_INPUT | Yes |
| gray_128/180 | **0.0** | ~0.002 | 0 | LOW_QUALITY_INPUT | Yes |
| uniform_skin | **0.0** | 0.0023 | 0 | LOW_QUALITY_INPUT | Yes |
| noise | **0.0** | 0.0074 | 0 | UNTRUSTWORTHY_OUTPUT | Yes |
| blank_skin | **0.0** | 0.0025 | 0 | LOW_QUALITY_INPUT | Yes |
| football | 0.0125 | 0.0189 | 1120 | VALID | No (unlabeled demo) |
| public wound test | 0.0406 | 0.0441 | 4442 | VALID | No |

**MODEL_OUTPUT vs DISPLAY_OUTPUT:** On blanks, raw is already empty — withholding is belt-and-suspenders, not the sole reason blanks look clear.

---

## F. U-Net geometry evidence

| Case | Orig | ROI | Mask | Full | Outside ROI | Inside ROI |
|------|------|-----|------|------|-------------|------------|
| square | 320×320 | 160×160 center | ROI-sized | 320×320 | **0** | 544 |
| portrait | 480×320 | center | ROI-sized | full | **0** | 459 |
| landscape | 320×480 | center | ROI-sized | full | **0** | 595 |
| public_test_full | 331×331 | full | 331×331 | same | 0 | 4442 |
| public_test_roi | 331×331 | central | ROI | full paste | **0** | 2179 |

Backend `_compose_full_image_mask` pastes ROI mask into full canvas without stretching ROI over the whole photo when bbox is present.

---

## G. API / MongoDB / frontend consistency

| Check | Result |
|-------|--------|
| Direct gated EffNet vs **GET** case | Football: classifier_finding=Swelling, status=VALID, model_status=NOT_TRUSTWORTHY |
| Grad-CAM on GET | `gradcam_overlay_generated=false`, reason `classifier_model_not_trustworthy` |
| Analyze POST body | Returns summary only (xgb/vqc/clinical flags) — **no** `visible_injury` key (by design); client must GET case |
| Mongo / GET persistence | Stores classifier + segmentation + Grad-CAM fields |
| Upload of pure gray / blank_skin | Rejected (blur variance gate) before analyze |
| Frontend | Relies on case GET; shows Withheld/unavailable + NOT_TRUSTWORTHY; mask “Withheld”; Grad-CAM “NOT CLINICAL EXPLANATION” |
| Browser E2E | **Not performed** this pass (no GUI browser automation). Used direct inference + FastAPI TestClient |
| Live uvicorn | Prior uvicorn session was failed/stale; used clean in-process TestClient |

First-aid: classifier channel uses `classifier_finding` / `classifier_probability` only; YOLO not copied into classifier evidence.

---

## H. Grad-CAM status

- Source: EfficientNetV2  
- When model training status is NOT_TRUSTWORTHY: **WITHHELD** (`classifier_model_not_trustworthy`)  
- Labels: MODEL VISUALIZATION / NOT CLINICAL EXPLANATION  
- Does **not** upgrade classifier trust  

---

## I. Model readiness

### EfficientNet
- Trustworthy? **No — NOT_TRUSTWORTHY**
- Retraining required? **Yes, once data exists** (`REQUIRES_RETRAINING`)
- Real data required? **Yes** (`REQUIRES_MORE_DATA`) — real normals, diverse injury photos, subject-aware splits; OOD collapse gate = 0 before promotion

### U-Net
- Trustworthy clinically? **No**
- Research demo? **Yes — DEMO_WITH_LIMITATIONS**
- Retraining required now? **No** for blank-collapse (already cleared)
- More real negatives? Helpful for calibration; not blocking blank gate. Domain is wound/ulcer photos, not cut/bruise localization.

---

## J. Tests

| Suite | Result |
|-------|--------|
| Targeted EffNet + U-Net + Grad-CAM + remediation comparison | **23 passed** (post-fix) |
| Earlier full `pytest backend/tests` during edit race | 214 passed, 1 flaky source-inspect failure (re-verified green) |
| Frontend `npm run lint` + `npm run build` | **Pass** |
| New regression | `/api/models` EffNet+U-Net SHA exposure |

---

## Files changed this phase

1. `backend/main.py` — `/api/models` exposes EffNet/U-Net SHA, paths, versions  
2. `backend/tests/test_efficientnet_reliability.py` — API SHA regression  
3. Probe artifacts: `scratch/effnet_unet_phase_verify.json`, `scratch/effnet_unet_api_consistency.json`  
4. This report

**Not changed:** EfficientNet weights, U-Net weights, YOLO, thresholds, clinical claim policy.

---

*End of independent EfficientNet + U-Net verification.*
