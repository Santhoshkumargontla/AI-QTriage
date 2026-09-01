# AI-QTriage Kaggle Integration Verification Report
**Date:** 2026-08-30  
**Workflow:** VERIFY-FIRST → FIX-ONLY-IF-NEEDED → RE-VERIFY  
**Verdict:** `READY_FOR_RESEARCH_DEMO_WITH_LIMITATIONS`

---

## Issue Register

### ISS-001 — Stale `canonical_manifest.json` (EfficientNet reject-v2 SHA)
| Field | Value |
|--------|--------|
| Priority | HIGH |
| Status before | STILL_BROKEN |
| Evidence | Manifest listed SHA `fa7aa5c8…` / version `reject-v2` while disk/API had `95cf385d…` / `kaggle-v1` |
| Root cause | Manifest not regenerated after kaggle-v1 promotion |
| Fix | Ran `ml/models/generate_registry.py` |
| Files changed | `ml/models/canonical_manifest.json`, `ml/models/model_registry.json` (SHA refresh) |
| Verification | Disk = wrapper = registry = manifest = API → `95cf385d…` |
| Final status | **VERIFIED_FIXED** |

### ISS-002 — Misleading YOLO “Dataset provenance” on user uploads
| Field | Value |
|--------|--------|
| Priority | HIGH |
| Status before | STILL_BROKEN |
| Evidence | User-uploaded hand photo UI showed “SYNTHETIC drawings/graphics (126/126…)” under YOLO section |
| Root cause | `yolo_dataset_provenance` is **model training** provenance, displayed without distinguishing from case image provenance |
| Fix | Frontend label → “YOLO model training data (not this uploaded image): …” |
| Files changed | `frontend/app/cases/[id]/page.tsx` |
| Verification | Code review; image provenance line (`display_message`) unchanged and correct |
| Final status | **VERIFIED_FIXED** |

### ISS-003 — Stale API notes referencing reject-v2 / swelling as EffNet class
| Field | Value |
|--------|--------|
| Priority | MEDIUM |
| Status before | STILL_BROKEN |
| Evidence | `/api/models` swelling_note and EfficientNet `note` referenced reject-v2 4-class head |
| Fix | Updated to kaggle-v1 8-class taxonomy; swelling explicitly not in classifier |
| Files changed | `backend/main.py` |
| Verification | API strings match metadata sidecar |
| Final status | **VERIFIED_FIXED** |

### ISS-004 — Fusion layer missing kaggle-v1 class mapping
| Field | Value |
|--------|--------|
| Priority | MEDIUM |
| Status before | PARTIALLY_FIXED |
| Evidence | `feature_fusion.py` only mapped Cut/Bruise/Swelling/Normal; new classes (Burn, Wound, etc.) and OOD_Reject ignored |
| Fix | Map OOD_Reject + injury classes without XGB dims into conservative `prob_other` bucket |
| Files changed | `ml/fusion/feature_fusion.py` |
| Verification | Logic review; XGBoost schema unchanged (23-d synthetic) |
| Final status | **VERIFIED_FIXED** |

### ISS-005 — Tests pinned to old reject-v2 SHA
| Field | Value |
|--------|--------|
| Priority | MEDIUM |
| Status before | STILL_BROKEN |
| Evidence | 3 failures in `test_efficientnet_reliability.py` |
| Fix | Updated `ACTIVE_EFFNET_HASH`, `ACTIVE_CLASSES`, injury set for kaggle-v1 |
| Files changed | `backend/tests/test_efficientnet_reliability.py` |
| Verification | 231 backend tests pass |
| Final status | **VERIFIED_FIXED** |

### ISS-006 — User real-photo screen (YOLO Bruise 76%, EffNet withheld)
| Field | Value |
|--------|--------|
| Priority | INFO (not a regression) |
| Status before | WORKING AS DESIGNED (limited) |
| Evidence | Live OOD gates: blank/uniform → raw `ood_reject`, gated withheld; real hand → EffNet `OUT_OF_DISTRIBUTION`, all classes withheld; YOLO synthetic-trained → class confusion on real skin |
| Root cause | Domain gap: canonical YOLO trained on synthetic drawings; kaggle-v1 EffNet gates real OOD photos |
| Fix | None required — behavior is honest research-demo limitation |
| Final status | **RESEARCH_DEMO_LIMITED** |

### ISS-007 — Skin YOLO burn candidate promotion
| Field | Value |
|--------|--------|
| Priority | HIGH |
| Status before | ALREADY_FIXED |
| Evidence | Candidate SHA `d26b6b56…` ≠ canonical `4d6e72f5…`; TRAIN_REPORT `promote: false`, `burn.recall: 0.0` |
| Final status | **ALREADY_FIXED** (correctly unpromoted) |

### ISS-008 — Fracture YOLO modality isolation
| Field | Value |
|--------|--------|
| Priority | HIGH |
| Status before | ALREADY_FIXED |
| Evidence | Artifact exists SHA `78550094cafbc84f…`; zero references in `backend/main.py`; skin YOLO loads `yolo11_injury_best.pt` only |
| Metrics | test mAP50 = **0.00317** (honest, poor) |
| Final status | **EXPERIMENTAL** / **NOT_TRUSTWORTHY** |

### ISS-009 — Swelling as current EffNet class
| Field | Value |
|--------|--------|
| Priority | HIGH |
| Status before | ALREADY_FIXED (with historical refs) |
| Evidence | Sidecar + wrapper: 8 classes, no swelling; questionnaire “swelling” is user-reported (valid); fusion `prob_swelling` is legacy XGB schema |
| Final status | **ALREADY_FIXED** |

### ISS-010 — HF / Kaggle credential exposure in repo
| Field | Value |
|--------|--------|
| Priority | CRITICAL |
| Evidence | Pattern scan: 0 hits in tracked source; `.gitignore` updated for `.huggingface/`, `.kaggle/` |
| Note | User pasted HF token in chat — **rotate immediately**; do not commit tokens |
| Final status | **VERIFIED_FIXED** (repo clean) |

---

## A. EFFICIENTNET FINAL STATE

| Property | Value |
|----------|--------|
| Canonical path | `ml/models/vision/efficientnetv2_injury_best.pt` |
| SHA-256 | `95cf385d85419a63eeef8e46de9c9ef7e7487ef0f326fb60ca376b8bae0eec9f` |
| Version | `kaggle-v1` |
| Class order (index → class) | 0 abrasion, 1 bruise, 2 burn, 3 cut, 4 laceration, 5 wound, 6 normal, 7 ood_reject |
| Output dimension | 8 |
| Runtime confirmation | Wrapper + live `/api/models` agree |
| Test n | 473 |
| Accuracy | 0.9641 |
| Macro precision | 0.9323 |
| Macro recall | 0.9461 |
| Macro F1 | **0.9375** |
| Weighted F1 | (from metadata) ~0.96+ |
| OOD blank result | Raw → ood_reject; gated → withheld (`LOW_QUALITY_INPUT`) |
| Limitation | Real user photos often `OUT_OF_DISTRIBUTION` — withheld, not clinical |

Per-class (held-out, from `TRAIN_REPORT.json` / metadata — consistent):

| Class | P | R | F1 | Support |
|-------|---|---|-----|---------|
| abrasion | 0.917 | 0.957 | 0.936 | 23 |
| bruise | 1.000 | 0.870 | 0.930 | 46 |
| burn | 0.818 | 0.818 | 0.818 | 22 |
| cut | 0.941 | 0.941 | 0.941 | 17 |
| laceration | 0.824 | 1.000 | 0.903 | 14 |
| wound | 0.980 | 0.983 | 0.982 | 299 |
| normal | 0.979 | 1.000 | 0.989 | 46 |
| ood_reject | 1.000 | 1.000 | 1.000 | 6 |

Split integrity: train/val/test hash leakage counts = 0 (prepare report).

---

## B. TAXONOMY COMPATIBILITY

| Class | EffNet supported? | Injury finding? | First-aid | Questionnaire | Frontend | Fusion |
|-------|-------------------|-----------------|-----------|---------------|----------|--------|
| abrasion | Yes | If confident | Routed | If routed | Shown (not in YOLO) | → prob_other |
| bruise | Yes | If confident | Routed | If routed | Shown | prob_bruise |
| burn | Yes | If confident | Burn rules if routed | If routed | Shown (not in YOLO) | → prob_other |
| cut | Yes | If confident | Routed | If routed | Shown | prob_cut |
| laceration | Yes | If confident | Not silently cut | If routed | Shown (not in YOLO) | → prob_other |
| wound | Yes | If confident | Routed | If routed | Shown | → prob_other |
| normal | Yes | **No** | **No** | **No** | Withheld when gated | → prob_other |
| ood_reject | Yes | **No** | **No** | **No** | Withheld | → prob_other |
| swelling | **No** | Only user Q | Q only | User Q field | Not EffNet class | Legacy prob_swelling dim |

---

## C. YOLO (SKIN)

| Property | Value |
|----------|--------|
| Active SHA | `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879` |
| Candidate SHA | `d26b6b5626e02edb108d1544430141e77626a1e42d1325891d0bb73dc4a9c5da` |
| Burn support (test) | 8 boxes (from confusion matrix row) |
| Burn precision | 1.0 |
| Burn recall | **0.0** |
| Burn F1 | ~0 (mAP50 0.046) |
| Promotion | **NOT promoted** — burn recall gate failed |
| Reason | `burn_recall=0.0; NO_CANONICAL_OVERWRITE` |

---

## D. FRACTURE YOLO

| Property | Value |
|----------|--------|
| Artifact | `ml/models/vision/yolo11_fracture_xray_best.pt` |
| SHA | `78550094cafbc84f409f18ae0c9eedba7e41cb2d0c3a6e313bb5ebf82f809d78` |
| Modality | X-ray only |
| test mAP50 | 0.00317 |
| Wired to skin pipeline? | **No** |
| Normal photo can reach it? | **No** |
| Status | **EXPERIMENTAL** / **REQUIRES_RETRAINING** |

Fracture dataset was trained from Kaggle (pkdarabi), not Hugging Face. HF token was **not** configured in repo.

---

## E. U-NET

| Property | Value |
|----------|--------|
| SHA | `3c7f3f39196d71b9d8d58d1fcfc7438b4ae23d75fbd8195ae7da7b1fcb9660d1` |
| EffNet retrain regression | **None detected** — artifact unchanged |
| Status | **READY_FOR_RESEARCH_DEMO** |

---

## F. MULTIMODAL / SENSOR

| Component | Provenance | Changed by kaggle EffNet? |
|-----------|------------|---------------------------|
| XGBoost | SYNTHETIC 23-d fusion | No |
| VQC | SYNTHETIC / EXPERIMENTAL | No |
| Paired real multimodal | 0 clinical pairs | No |
| Sensor model | Synthetic / demo football_fall limitation | No |

---

## G. SECURITY

- Repo credential scan: **0 pattern hits**
- `.gitignore`: `.env`, `.kaggle/`, `.huggingface/` protected
- **Action required:** Rotate HF token shared in chat; never commit tokens

---

## H. END-TO-END VERIFICATION

| Layer | Verified? |
|-------|-----------|
| Direct model inference | Yes (OOD probes, wrapper load) |
| Live API (`/api/models`) | Yes — EffNet SHA `95cf385d…`, 8 classes |
| MongoDB stored cases | Not re-run this session (historical compat via tests) |
| Frontend | Label fix applied; lint + build pass |
| Browser GUI | **BROWSER_GUI_NOT_RUN** |

---

## I. FINAL TEST RESULTS

| Command | Result |
|---------|--------|
| `pytest backend/tests -q` | **231 passed** |
| `npm run lint` | **Pass** |
| `npm run build` | **Pass** |
| `pytest -q` (project root) | Not run separately (backend covers ML) |

---

## J. FINAL VERDICT

**Overall:** `READY_FOR_RESEARCH_DEMO_WITH_LIMITATIONS`

| Component | Readiness |
|-----------|-----------|
| EfficientNet kaggle-v1 | **READY_FOR_RESEARCH_DEMO** (real-photo OOD gating expected) |
| Skin YOLO | **RESEARCH_DEMO_LIMITED** (synthetic training; real-photo class errors) |
| U-Net | **READY_FOR_RESEARCH_DEMO** |
| Fracture YOLO | **NOT_TRUSTWORTHY** / **EXPERIMENTAL** |
| XGBoost | **RESEARCH_DEMO_LIMITED** (synthetic) |
| VQC | **EXPERIMENTAL_ONLY** |
| Sensor | **RESEARCH_DEMO_LIMITED** |
| Multimodal fusion | **RESEARCH_DEMO_LIMITED** |
| Browser verification | **BROWSER_GUI_NOT_RUN** |
| Twilio | Demo/stub (no real emergency dispatch) |

### Your current screen explained
For a **real user-uploaded hand photo** with demo sensor data:
1. **YOLO Bruise 76%** — canonical YOLO is synthetic-trained; misclassification on real skin is a **model limitation**, not a coordinate bug.
2. **EfficientNet all withheld / OUT_OF_DISTRIBUTION** — quality gates correctly refuse closed-set injury labels on OOD real photos.
3. **Grad-CAM withheld** — correct (`softmax_not_trusted_closed_set_overconfidence`).
4. **YOLO training provenance line** — was misleading; now labeled as model training data, not the uploaded image.

### HF fracture training
Fracture YOLO already exists from Kaggle training (weak mAP). Additional HF-based retraining was **not** started (verify-first rule). To proceed: configure HF token only in local env (never repo), then train via separate X-ray pipeline — do not merge into skin YOLO.
