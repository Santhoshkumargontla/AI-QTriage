# AI-QTriage Final State Reconciliation Audit

**Date:** 2026-08-30  
**Method:** Independent inspection of current disk artifacts, wrappers, registry, manifest, metadata, live FastAPI (`127.0.0.1:8000`), direct inference, and tests. Historical forensic markdown files were **not** treated as truth.

**NOT_PRODUCTION_READY**  
**NOT_CLINICALLY_VALIDATED**

---

## A. Executive verdict

```text
DEMO_WITH_LIMITATIONS
```

Also: **NOT_PRODUCTION_READY**, **NOT_CLINICALLY_VALIDATED**.

| Layer | Current truth |
| ----- | ------------- |
| Code / API / registry consistency | Largely aligned (5-way SHA match for all six runtime models) |
| Model loads + inference | Yes |
| Honest metrics for loaded artifacts | Yes for XGB/VQC (recomputed); EffNet/U-Net/YOLO metadata SHA matches disk |
| Clinical / real-photo reliability | **No** — YOLO fails football demo at keep-threshold 0.25; EffNet marks demo OOD; U-Net is ulcer-domain research |
| Browser GUI E2E | **BROWSER_GUI_NOT_RUN** |

---

## B. Current canonical models

| Model | Canonical path | SHA-256 | Version | Status | Dataset provenance | Runtime |
| ----- | -------------- | ------- | ------- | ------ | ------------------ | ------- |
| YOLO11 | `ml/models/vision/yolo11_injury_best.pt` | `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879` | v1.4.0 | `INFERENCE_EXECUTES` | `yolo_retrain_v2` synthetic cut/bruise drawings; **wound boxes = 0** | Yes — only path wrappers load |
| EfficientNetV2 | `ml/models/vision/efficientnetv2_injury_best.pt` | `95cf385d85419a63eeef8e46de9c9ef7e7487ef0f326fb60ca376b8bae0eec9f` | kaggle-v1 | `READY_FOR_RESEARCH_DEMO` | Kaggle multi-class photos; classes abrasion/bruise/burn/cut/laceration/wound/normal/ood_reject | Yes |
| U-Net | `ml/models/vision/unet_injury_best.pt` | `3c7f3f39196d71b9d8d58d1fcfc7438b4ae23d75fbd8195ae7da7b1fcb9660d1` | deduped-subject-v1 | `READY_FOR_RESEARCH_DEMO` | AZH+wseg+Medetec subject-aware + synthetic empty | Yes |
| XGBoost | `ml/models/xgboost_best.json` | `73bb5a5125c3e3907bffa1059165d90ecce9dd4e47ce9ce8f9c1f8937fd3f643` | v1.2.0 | `TRAINED` | Synthetic 23-d fusion (0 clinical pairs) | Yes |
| VQC | `ml/models/vqc/vqc_weights.npz` | `2db769bec3abd3c2d4811f8a39c5f3c5e3b41cccdde030a10c056848a9c6389e` | v1.4.0 | `EXPERIMENTAL_ONLY` | Same synthetic fusion; PennyLane `default.qubit` | Loaded; isolated from SOS/main decision |
| Sensor | `ml/models/sensor_motion_best.json` | `707ee9c25392d5e8e1ca9e023feb77937b6ad5a159cb0b53d3b6a09f047e1cbb` | v1.2.0 | `TRAINED` | Motion telemetry research set | Yes — `process_sensor_data` → `predict_from_summary` |

Five-way check (disk = registry = manifest): **all six OK**.

YOLO wrapper SHA on load: `4d6e72f5…` (matches disk).  
EfficientNet wrapper classes after load: 8-class kaggle taxonomy (no swelling).  
Classes file: `ml/models/vision/efficientnetv2_injury_best_classes.json`.

---

## C. Contradictions resolved

| OLD CLAIM (from historical reports) | VERIFIED CURRENT TRUTH | EVIDENCE | ACTION |
| ----------------------------------- | ---------------------- | -------- | ------ |
| EfficientNet SHA `6944605a…` collapses blank→Swelling / NOT_TRUSTWORTHY | Canonical is **`95cf385d…` (kaggle-v1)**; blanks withheld as `LOW_QUALITY_INPUT`; no Swelling in head | Disk hash; wrapper `predict` on gray/black/white; metadata classes | Keep kaggle checkpoint; document; deprecate drawing trainer as alternate path |
| YOLO SHA `f4382450…` or `6cc84115…` is active | Active is **`4d6e72f5…` (retrain_v2)**; prior hashes are backups/historical | `sha256_file(YOLO_CANONICAL)`; wrapper print | No silent replace |
| YOLO wound is a trained class | **wound has 0 training boxes**; `wound_reliability_claimed: false` | `yolo11_metadata.json` dataset_stats | Disclose UNSUPPORTED / untrained; do not claim reliability |
| VQC metrics 11/30 or 14/30 | **Recomputed 16/30 = 0.533333** matches `canonical_held_out_evaluation.json` | Live `VQCClassifier.predict` on seed-42 held-out | Trust recomputed 16/30 |
| VQC silent fallback `[0.15,0.70,0.15]` | **Absent** from `predict` source | `inspect.getsource` | None |
| U-Net ~50% blank positives / NOT_TRUSTWORTHY | Blank raw mean ≤0.004, positive_ratio **0.0**, withheld | Live `UNetSegmenter.segment` | No regression |
| XGBoost 25/30 | **Recomputed 25/30 acc 0.833333**, `n_features_in_=23` | Live predict | Matches registry |
| Sensor not wired | **Wired**; football_fall → `fall` conf 0.979 | `process_sensor_data` | None |
| Twilio `TWILIO_PHONE_NUMBER` only | Canonical `TWILIO_FROM_NUMBER` / `TWILIO_TO_NUMBER`; aliases documented | `.env.example`, `twilio_service`, `/api/sos/config` | None |
| `generate_registry.py` default EffNet classes cut/bruise/swelling | Was stale fallback (metadata overrode) | Source | **Fixed** fallback to kaggle-v1 8 classes + kaggle train command |
| Wrapper default EffNet classes included swelling | Pre-load placeholder | Source | **Fixed** placeholder to 8-class kaggle taxonomy |
| `backend/config.py` comment “blanks → swelling” | Outdated for kaggle-v1 | Source | **Fixed** comment |

Historical reports under `AI-QTriage_*.md` remain archival; **this file** is the current reconciliation.

---

## D. Current defects

| ID | Severity | Category | Root cause | Component | Runtime effect | New dataset? | Next action |
| -- | -------- | -------- | ---------- | --------- | -------------- | ------------ | ----------- |
| D01 | High (demo) | DATASET_GAP / TRAINING_GAP | YOLO trained mostly on synthetic cut/bruise drawings; wound = 0 boxes | YOLO | football demo: max conf ~0.077 → **0 boxes at keep 0.25**; synthetic val/test still detect | **REQUIRED** real-photo boxes for cut/bruise/wound | Collect labeled real photos; retrain; do not lower threshold to fake demos |
| D02 | Medium | DATASET_GAP | `wound` in `model.names` but zero labels | YOLO taxonomy | Class name exists; reliability must stay false | REQUIRED if wound is advertised | Keep UNSUPPORTED disclosure; or drop class from training nc |
| D03 | Medium | THRESHOLD_CALIBRATION_ISSUE / DOMAIN_GAP | Keep-threshold 0.25 correct for FP control; real photo below gate | YOLO + demo image | Demo looks “broken” at honest threshold | Optional more real data | Document; do not set 0.10 for marketing |
| D04 | Low | METADATA_BUG (mitigated) | Drawing trainer `train_efficientnet.py` can still promote over kaggle if gates pass | Training scripts | Accidental overwrite risk | No | Docstring warns; prefer `train_efficientnet_kaggle_v1.py` |
| D05 | Low | SCHEMA_COMPAT | XGB still has `prob_swelling` dim; EffNet has no Swelling | Fusion | Swelling dim stays 0 unless questionnaire; Burn/Wound/etc. bucketed into `prob_other` | No (unless XGB retrained) | Document; optional future schema v2 |
| D06 | Info | EXTERNAL_CONFIGURATION_GAP | Twilio disabled | SOS | Local simulation only | No | Configure env for real SMS test |
| D07 | Info | NO_CURRENT_DEFECT (verification gap) | Browser GUI not automated this session | Frontend | Unknown pixel overlay drift | No | Optional Playwright/manual E2E |

No current runtime **fake fallback prediction** found in VQC/XGBoost analyze path.  
No current need to replace U-Net or EfficientNet weights based on blank probes.

---

## E. Dataset needs

### REQUIRED DATASET

- **YOLO:** Real skin-injury photographs with honest xyxy labels for cut and bruise (and wound **only** if the class is kept). Current independent test is 8 cut + 10 bruise boxes on mostly synthetic drawings — insufficient for reliability claims.
- **Wound class:** Either add honest wound boxes or stop implying wound is trainable.

### OPTIONAL IMPROVEMENT

- EfficientNet: more balanced real-photo rare classes (burn/laceration supports are small).
- U-Net: cut/bruise localization is out of domain — separate task if needed.
- XGBoost/VQC: genuine paired multimodal clinical records (currently 0).

### NO DATASET REQUIRED

- Twilio env naming (configuration only).
- Sensor classifier wiring (already invoked).
- VQC fake-fallback removal (already done).
- Registry/manifest SHA alignment (already matching).
- Blank EffNet/U-Net withholding (working).

---

## F. Honest model readiness matrix

| Model | Readiness | Notes |
| ----- | --------- | ----- |
| YOLO skin | **DEMO_WITH_LIMITATIONS** / `INFERENCE_EXECUTES` | Synthetic cut/bruise OK at 0.25; real football demo fails; wound unsupported |
| EfficientNet | **READY_FOR_RESEARCH_DEMO** | kaggle-v1; blanks/OOD withheld; not clinical |
| U-Net | **READY_FOR_RESEARCH_DEMO** | Blank FP ~0; ulcer/chronic domain; not cut localization |
| XGBoost | **READY_FOR_RESEARCH_DEMO** | Synthetic 25/30; not clinical |
| VQC | **EXPERIMENTAL_ONLY** | 16/30; circuits match; no quantum advantage |
| Sensor | **READY_FOR_RESEARCH_DEMO** | Invoked; football_fall → fall |
| Fracture X-ray | **OUT_OF_SCOPE / SEPARATE** | Not merged into skin pipeline |
| Twilio | **NOT_CONFIGURED** / local simulation | Canonical env present; no real SMS this audit |

---

## G. Verification

### Tests

| Suite | Result |
| ----- | ------ |
| `pytest -q` (project root) | **231 passed**, 3 warnings, 171.31s |
| `npm run lint` | exit 0 |
| `npm run build` | exit 0 (Next.js 16.3.0) |

### Direct inference (2026-08-30)

| Probe | Result |
| ----- | ------ |
| YOLO blank @0.25 | 0 boxes |
| YOLO football demo @0.25 | 0 boxes (top raw ~0.077 cut below keep) |
| YOLO synthetic val bruise | bruise 0.994 |
| YOLO synthetic val cut | cut 0.374 |
| EffNet gray/black/white | `LOW_QUALITY_INPUT`, winner null |
| EffNet football demo | `OUT_OF_DISTRIBUTION`, max≈0.45 |
| U-Net blank gray/black/white | raw positive_ratio 0.0, withheld |
| XGB held-out | 25/30 = 0.833333 |
| VQC held-out | 16/30 = 0.533333; no fake fallback |
| Sensor football_fall.csv | `fall` / 0.979 / `classified` |

### API (live uvicorn)

- `/api/health` → healthy, MongoDB connected  
- `/api/sos/config` → `configured: false`, canonical env listed, `real_sms_tested: false`  
- `/api/models` → YOLO `INFERENCE_EXECUTES` v1.4.0 cut/bruise/wound; EffNet `READY_FOR_RESEARCH_DEMO` kaggle-v1 8 classes; U-Net / XGB / VQC statuses match registry  

### Browser

```text
BROWSER_GUI_NOT_RUN
```

Frontend build succeeds; live page traffic was observed earlier against the API, but full create→analyze→reload GUI verification was not executed as a structured browser E2E in this audit.

### Leakage

- YOLO/EffNet/U-Net metadata claim processed/subject-aware splits; **LEAKAGE_NOT_FULLY_VERIFIED** end-to-end in this session (no fresh full near-duplicate scan). XGB/VQC synthetic split: val excluded from test per held-out artifact.

### Remaining unverified

- Pixel-perfect YOLO overlay in browser after page reload  
- Real Twilio SID path with live credentials  
- Full near-duplicate leakage re-scan across all vision datasets  

---

## Changes made in this reconciliation (minimal)

1. `ml/models/generate_registry.py` — EffNet fallback classes/command → kaggle-v1 (8-class); U-Net version string aligned.  
2. `ml/vision/efficientnet_wrapper.py` — pre-load class placeholder → kaggle-v1 taxonomy (no swelling).  
3. `ml/training/train_efficientnet.py` — docstring: historical/alternate drawing path; prefer kaggle trainer.  
4. `backend/config.py` — remove obsolete “blanks → swelling” comment.

No model weights were replaced. No thresholds lowered. No fabricated metrics.

---

## Bottom line

**What is true now:** Canonical artifacts, registry, manifest, wrappers, and API agree on SHAs and statuses. EfficientNet is the promoted Kaggle 8-class model (not the old Swelling-collapse weights). U-Net blanks are safe. XGB/VQC metrics recompute to 25/30 and 16/30. Sensor classifier is actually called. Twilio is honestly not configured.

**What remains limited:** YOLO is a research demo on synthetic cut/bruise drawings and does **not** fire on the football demo at the honest 0.25 keep-threshold; wound is unsupported. That is a **dataset/training gap**, not an integration bug, and must not be “fixed” by lowering thresholds or special-casing filenames.
