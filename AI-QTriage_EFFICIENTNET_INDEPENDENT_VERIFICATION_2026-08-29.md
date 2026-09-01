# AI-QTriage EfficientNet Independent Verification Report

**Date:** 2026-08-29  
**Workspace:** `C:\Users\santh\Capstone Project Code`  
**Method:** VERIFY-FIRST → FIX-ONLY-IF-NEEDED → RE-VERIFY  
**Rule:** A withheld bad prediction is not a fixed model.

---

## K. Final verdict

**SAFETY GATE WORKING — UNDERLYING MODEL STILL NOT TRUSTWORTHY**

Raw blank / uniform / noise inputs still collapse to a high-confidence injury class (usually Swelling ≈ 0.96–1.0). Application quality gates withhold those outputs. Grad-CAM is now blocked while training status is `NOT_TRUSTWORTHY`. The classifier remains **research/demo only**, not clinically reliable, and **must not** be presented as production-ready.

**Retraining decision:** `REQUIRES_MORE_DATA` (also `REQUIRES_RETRAINING` once adequate real normals + diverse injury photos exist). No automatic retrain performed.

---

## Issue ledger

### EFF-01 — Canonical artifact identity
| Field | Value |
|--------|--------|
| Priority | P0 |
| Status before verification | Reported SHA `6944605a…` / NOT_TRUSTWORTHY |
| Evidence found | Disk, registry, manifest, metadata, sidecar, wrapper load all agree |
| Root cause | N/A — consistent |
| Already fixed? | Yes (identity) |
| Exact fix applied | None |
| Files changed | None |
| Verification | `sha256_file`, registry/manifest walk, cwd reload |
| Verification result | Match |
| Final status | **ALREADY_FIXED** (identity); model quality remains **NOT_TRUSTWORTHY** |

### EFF-02 — Raw OOD overconfidence (blank/gray/black/white)
| Field | Value |
|--------|--------|
| Priority | P0 |
| Status before verification | Reported gray→Swelling ~0.96 |
| Evidence found | Direct `predict_raw`: black/white/gray→Swelling or Bruise at 0.96–1.0 |
| Root cause | Closed-set 3-class head; missing normal/background diversity; synthetic-drawing shortcuts |
| Already fixed? | No (model). App gate yes |
| Exact fix applied | None to weights (retrain not justified without data) |
| Files changed | None (weights) |
| Verification | Multi-level gray + black/white/noise probes |
| Verification result | Collapse confirmed |
| Final status | **STILL_BROKEN** / **NOT_TRUSTWORTHY** / **REQUIRES_MORE_DATA** |

### EFF-03 — Application withholding on blanks
| Field | Value |
|--------|--------|
| Priority | P0 |
| Status before verification | Reported API withholds |
| Evidence found | `assess_input_quality` + confidence/margin/entropy gates; blanks → `LOW_QUALITY_INPUT`, winner=None |
| Root cause | Intentional safety gate (hides known failure; incomplete for textured OOD) |
| Already fixed? | Yes for uniform blanks |
| Exact fix applied | None (gate kept) |
| Verification | Gated vs raw probes |
| Final status | **ALREADY_FIXED** (gate) — **does not** upgrade model trust |

### EFF-04 — Grad-CAM on NOT_TRUSTWORTHY model
| Field | Value |
|--------|--------|
| Priority | P0 |
| Status before verification | Unknown / assumed OK |
| Evidence found | Football demo: VALID Swelling 0.988 + Grad-CAM overlay **generated** |
| Root cause | Grad-CAM keyed only on per-image VALID, ignored training_status |
| Already fixed? | No |
| Exact fix applied | Block Grad-CAM when metadata training_status contains NOT_TRUSTWORTHY |
| Files changed | `ml/explainability/grad_cam.py`, `backend/tests/test_gradcam_reliability.py` |
| Verification | Football re-probe → WITHHELD / `classifier_model_not_trustworthy` |
| Final status | **VERIFIED_FIXED** (presentation); model still **NOT_TRUSTWORTHY** |

### EFF-05 — Questionnaire routing using untrusted classifier confidence
| Field | Value |
|--------|--------|
| Priority | P1 |
| Status before verification | Finding blocked; confidence still readable |
| Evidence found | `_routing_finding` returned ""; `_routing_confidence` still took `classifier_probability` |
| Root cause | Confidence helper did not skip classifier when NOT_TRUSTWORTHY / not confident |
| Already fixed? | Partially |
| Exact fix applied | Skip classifier probability keys when untrusted/withheld |
| Files changed | `backend/main.py`, `backend/tests/test_remediation_regression.py` |
| Verification | Unit assertions → confidence 0.0 |
| Final status | **VERIFIED_FIXED** |

### EFF-06 — Frontend null classification probs
| Field | Value |
|--------|--------|
| Priority | P2 |
| Status before verification | Unknown |
| Evidence found | Withheld class probs are `null`; UI did `(prob * 100)` |
| Root cause | Missing null handling |
| Exact fix applied | Show `withheld` when prob not numeric |
| Files changed | `frontend/app/cases/[id]/page.tsx` |
| Final status | **VERIFIED_FIXED** |

### EFF-07 — Held-out metrics vs metadata claim
| Field | Value |
|--------|--------|
| Priority | P0 |
| Status before verification | Metadata test accuracy 1.0 |
| Evidence found | Live eval on current `efficientnet_processed` test (cut/bruise only, n=19): accuracy ≈ 0.16, ECE ≈ 0.84, mean conf ≈ 0.99 |
| Root cause | Metadata reflects older/different eval set with swelling class; current processed set has **no swelling images** while head is 3-class |
| Final status | **NOT_TRUSTWORTHY**; metrics claim not trusted for current disk set |

---

## A. Active EfficientNet artifact

| Item | Value |
|------|--------|
| Canonical path | `ml/models/vision/efficientnetv2_injury_best.pt` |
| Full SHA-256 | `6944605ae71685d909d505d12ce32d5bd9e953c10bbebb72a02635a729897e83` |
| Wrapper | `ml.vision.efficientnet_wrapper.EfficientNetV2Classifier` |
| Registry SHA | Same as disk |
| Manifest SHA | Same as disk |
| Metadata path | `ml/models/vision/efficientnetv2_metadata.json` |
| Metadata SHA field | Same as disk |
| Classes sidecar | `ml/models/vision/efficientnetv2_injury_best_classes.json` → `["cut","bruise","swelling"]` |
| Model version | `v1.3.0` |
| Runtime status | Loaded; `training_status` / `readiness_status` = **NOT_TRUSTWORTHY** |
| Fallback risk | Wrapper uses `EFFNET_CANONICAL` via `resolve_existing`; candidates under `ml/models/efficientnet_*_training/` are **not** auto-loaded |
| CWD independence | Verified load from `backend/` and foreign temp cwd → same SHA |

---

## B. Training data

### Production-associated processed set (`data/datasets/efficientnet_processed`)

Provenance: **100% SYNTHETIC** (PIL/drawing style). Clinical status: **NOT CLINICAL**.

| Class | Train | Val | Test | Real/synthetic | Notes |
|-------|------:|----:|-----:|----------------|-------|
| cut | 44 | 9 | 9 | SYNTHETIC | Present |
| bruise | 44 | 10 | 10 | SYNTHETIC | Present |
| swelling | 0 | 0 | 0 | — | **Absent from this manifest** despite 3-class head |

Subject IDs: **none** → `SUBJECT_LEAKAGE_STATUS_UNKNOWN`.

### Subject-normal candidate set (`efficientnet_subject_normal`) — not promoted

| Class | Train | Val | Test | Provenance |
|-------|------:|----:|-----:|------------|
| cut | 44 | 9 | 9 | SYNTHETIC |
| bruise | 44 | 10 | 10 | SYNTHETIC |
| normal | 12 | 3 | 3 | PUBLIC_REAL_PHOTOS (empty-region / empty-mask patches) |
| swelling | omitted | | | Only 2 unique drawings; not fabricated |

Promotion history: OOD injury-collapse gate failed → **KEEP_BASELINE**.

Metadata claims for the **promoted** checkpoint (older eval narrative): train 140 / val 30 / test 30 with swelling support and accuracy 1.0. That narrative **does not match** the current processed manifest class support.

---

## C. Leakage audit

| Check | Result |
|-------|--------|
| File hash overlap train↔val↔test (`efficientnet_processed`) | **0** |
| Duplicate pixel hashes within set | **0** |
| Subject IDs available (processed) | **No** → `SUBJECT_LEAKAGE_STATUS_UNKNOWN` |
| Subject split leak (`subject_normal`) | **0** multi-split subjects |
| Preprocessing leakage | Train and inference both RGB→resize 224→/255→ImageNet mean/std; consistent |
| Validation used for selection | Metadata shows val accuracy 1.0 by epoch 2 — possible optimistic selection; not independently re-proven on original train set |

---

## D. Raw OOD behavior (selected probes)

| Type | Provenance | Raw winner | Max p | Entropy | Margin | Application |
|------|------------|------------|------:|--------:|-------:|-------------|
| black | blank_control | Swelling | 1.00 | ~0 | 1.00 | WITHHELD `uniform_or_blank_image` |
| white | blank_control | Swelling | 1.00 | ~0 | 1.00 | WITHHELD |
| gray_64 | blank_control | Swelling | 1.00 | ~0 | 1.00 | WITHHELD |
| gray_128 | blank_control | Swelling | 0.963 | 0.16 | 0.93 | WITHHELD |
| gray_180 | blank_control | Swelling | 0.962 | 0.16 | 0.92 | WITHHELD |
| gray_220 | blank_control | Bruise | 1.00 | ~0 | 1.00 | WITHHELD |
| uniform_skin | blank_control | Swelling | 1.00 | ~0 | 1.00 | WITHHELD |
| near_uniform_gray (σ≈2) | near_uniform | Cut | 0.72 | 0.77 | 0.54 | WITHHELD |
| noise | synthetic_ood | Swelling | 1.00 | ~0 | 1.00 | WITHHELD `high_frequency_unstructured` |
| blurred_noise | synthetic_ood | Swelling | 0.996 | 0.03 | 0.99 | WITHHELD |
| dummy_test | synthetic_tiny | Swelling | 1.00 | ~0 | 1.00 | WITHHELD |
| blank_skin | synthetic_blank_skin | Swelling | 0.989 | 0.07 | 0.98 | WITHHELD |
| forensic_ood | unlabeled_ood | Swelling | 0.962 | 0.16 | 0.92 | WITHHELD |
| football | demo_unlabeled | Swelling | 0.999 raw / 0.988 gated | ~0.01 | ~1.0 | **APPLICATION VALID Swelling** (OOD_OVERCONFIDENCE passes gate) |
| in_domain_bruise | SYNTHETIC_TRAIN_DOMAIN | Swelling | 0.986 | 0.07 | 0.97 | APPLICATION VALID Swelling (**wrong class** vs bruise label) |

**Distinction:** RAW_MODEL_OUTPUT remains collapsed; APPLICATION_OUTPUT withholds blanks but **does not** make the model trustworthy. Football is **OOD_OVERCONFIDENCE**, not a labeled FALSE_POSITIVE (unlabeled demo).

---

## E. Withholding behavior

| Layer | Behavior |
|-------|----------|
| Raw | Always emits injury softmax; blanks → Swelling/Bruise |
| Quality gate | std/ptp/palette/blur/skin-color / high-freq noise |
| Confidence gate | min_conf 0.80, margin≥0.20, entropy cap |
| User-visible | winner/probs null when withheld; status + reason exposed |
| Grad-CAM (after fix) | No overlay while model NOT_TRUSTWORTHY |
| Questionnaire routing | Does not route from NOT_TRUSTWORTHY / non-confident classifier |
| Fusion | No EfficientNet feature keys in fusion layer; YOLO miss does not copy Swelling into `finding` |

Gate completeness: **incomplete**. Textured unrelated images (football) can still pass as VALID. Gate type: **B — hides known failures; partial genuine quality filtering**. Kept intentionally; not labeled as a model-quality fix.

---

## F. Calibration

Live evaluation on current `efficientnet_processed` **test** split (cut/bruise only; n=19 readable images):

| Metric | Value |
|--------|------:|
| Accuracy | ~0.158 |
| Macro precision | ~0.556 |
| Macro recall | ~0.107 |
| Macro F1 | ~0.172 |
| MCC | ~0.133 |
| Multiclass Brier (mean) | ~0.555 |
| ECE (rough 10-bin) | ~0.836 |
| Mean confidence | ~0.994 |

Metadata-reported accuracy 1.0 / ECE from original training narrative is **not** reproduced on the current processed test images.

**OOD_CALIBRATION_EVIDENCE_INSUFFICIENT** for labeled normals (only 18 real empty patches in unpromoted set; no held-out clinical negatives for the promoted head).

Do not infer good calibration from historical held-out accuracy.

---

## G. Root cause assessment

| Hypothesis | Verdict |
|------------|---------|
| Missing normal/background class in promoted head | **CONFIRMED_ROOT_CAUSE** (3 injury classes; blanks map to Swelling) |
| Synthetic drawing shortcut learning | **LIKELY_CONTRIBUTING_FACTOR** |
| Class imbalance / swelling over-represented in original train | **LIKELY_CONTRIBUTING_FACTOR** (metadata CM test swelling n=20 vs 5/5) |
| File-hash train/test leakage (current processed) | **NOT_SUPPORTED_BY_EVIDENCE** (0 overlap) |
| Subject leakage (processed) | **SUBJECT_LEAKAGE_STATUS_UNKNOWN** |
| Train vs inference preprocess mismatch | **NOT_SUPPORTED_BY_EVIDENCE** (aligned ImageNet normalize) |
| Poor closed-set calibration | **CONFIRMED_ROOT_CAUSE** (high conf + low live accuracy) |
| Insufficient data diversity | **CONFIRMED_ROOT_CAUSE** |

---

## H. Model status

**NOT_TRUSTWORTHY** (also **EXPERIMENTAL** research classifier)

Even with blank withholding and Grad-CAM blocked, raw OOD collapse and football VALID Swelling remain.

---

## I. Retraining decision

**REQUIRES_MORE_DATA** / **REQUIRES_RETRAINING** when data exists.

Do **not** retrain on synthetic clones of the same drawings.

Required before any promotion attempt:

1. Substantial **real** healthy/normal + diverse non-injury backgrounds (subject-aware splits)
2. Real injury photographs for cut/bruise/swelling with honest labels (or drop swelling until supported)
3. Hash-disjoint + subject-aware splits; no fabricated swelling from ulcer remaps
4. OOD injury-collapse gate = **0** on blank/gray/black/white/noise **before** promotion
5. Matching classes sidecar; live calibration on held-out + valid negatives

Subject-normal candidate already failed promotion (collapse ≠ 0). Baseline kept.

**NO_RETRAINING_REQUIRED** for this audit cycle (weights unchanged by design).

---

## J. Test results

| Suite | Result |
|-------|--------|
| Targeted EffNet + Grad-CAM + routing | **21 passed** |
| `pytest backend/tests -q` | **215 passed** |
| `pytest -q` (project) | **215 passed** |
| `frontend` `npm run lint` | Pass |
| `frontend` `npm run build` | Pass |

Regression coverage added/updated: canonical SHA, sidecar, cwd independence, multi-gray raw collapse vs withhold, routing not clinicalizing untrusted output, Grad-CAM blocked for NOT_TRUSTWORTHY.

---

## Preprocessing (Phase 3 summary)

`original RGB → cv2.resize(224,224) → float/255 → ImageNet mean/std → NCHW → logits → softmax (T=1.5 gated / T=1 raw) → gates`

Training (`train_efficientnet.py`) uses the same resize and ImageNet normalization. No preprocess fix applied.

---

## Files changed this audit

1. `ml/explainability/grad_cam.py` — withhold Grad-CAM when model NOT_TRUSTWORTHY  
2. `backend/main.py` — routing confidence ignores untrusted classifier probs  
3. `backend/tests/test_gradcam_reliability.py` — contract tests  
4. `backend/tests/test_efficientnet_reliability.py` — expanded regression suite  
5. `backend/tests/test_remediation_regression.py` — confidence assertions  
6. `frontend/app/cases/[id]/page.tsx` — withheld probs render as `withheld`  
7. This report

---

*End of independent EfficientNet verification. Do not treat safety gates as clinical validation.*
