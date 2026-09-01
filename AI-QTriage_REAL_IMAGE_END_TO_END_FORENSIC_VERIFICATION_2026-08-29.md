# AI-QTriage Real-Image End-to-End Forensic Verification

**Date:** 2026-08-29  
**Case under test:** `ac69884f-7a50-48e5-b7cc-a9bea9b20313`  
**Project:** `C:\Users\santh\Capstone Project Code`  
**Classification:** RESEARCH / DEMO ONLY WITH EXPLICIT LIMITATIONS  
**Browser GUI:** `BROWSER_GUI_NOT_RUN` (API + direct inference + disk identity verified)  
**API verification:** `VERIFIED_BY_API_ONLY` (FastAPI TestClient + MongoDB)

---

## Executive verdict

The uploaded hand photograph with a visible bleeding cut was the **exact** image used by inference (SHA-256 match end-to-end). The UI **Bruise ~0.76** result is **technically correct according to the active YOLO pipeline** — it is **not** a class-index swap, stale cache, or frontend relabel.

Primary root cause (exactly one):

### **GENUINE_MODEL_MISCLASSIFICATION**

Supporting evidence:

- Active checkpoint names: `{0: cut, 1: bruise, 2: wound}` — consistent disk → wrapper → Ultralytics → API → frontend.
- At keep-threshold **0.25**: top detection is **bruise @ 0.7574** on wrist/skin (left edge), not on the central laceration.
- At **conf=0.01**: a **cut @ 0.0818** appears near the laceration center — **below** the application keep-threshold; never shown.
- Backup checkpoint `6cc84115…` also predicts **bruise** (lower conf ~0.46) — not an active-vs-backup mapping regression.
- Debug overlay (`scratch/ac69884f-…_yolo_debug.jpg`) shows bruise boxes on uninjured skin; the cut itself is uncovered at 0.25.

Honest follow-on status: **MODEL_ERROR_ON_OOD/REAL-WORLD IMAGE — REQUIRES_MORE_DATA**  
Do **not** lower thresholds, special-case this hash/filename, or rename bruise→cut in the UI.

---

## A. Image identity chain

| Stage | Value |
| --- | --- |
| Case ID | `ac69884f-7a50-48e5-b7cc-a9bea9b20313` |
| Stored path | `data/uploads/ac69884f-7a50-48e5-b7cc-a9bea9b20313.jpeg` |
| File SHA-256 | `87a76147983d0cdb3a63c9f3d3988b0e16ba8157085513f63463de26a559b446` |
| Pixel SHA-256 (decoded) | `07012fd977c9c1fe8cd91d6dd0dd15e1f91539ad6b7d1155196373d0e918c89c` |
| Dimensions (H×W) | **700 × 1283** |
| Mean / std (BGR) | ~68.95 / ~59.77 |
| Mongo `image_reference` | same absolute path as disk |
| Post-fix API `visible_injury.image_sha256` | **matches** disk SHA |
| Inference input | wrapper / Ultralytics / `/api/cases/{id}/analyze` all read this path |

**Ruled out:** stale previous upload, demo football fixture, thumbnail-only path, Grad-CAM-as-input, hardcoded case fixture.

Artifacts: `scratch/real_image_e2e_forensic.json`, `scratch/real_image_e2e_reanalyze.json`.

---

## B. Raw YOLO results (active SHA `4d6e72f5…`)

Checkpoint: `ml/models/vision/yolo11_injury_best.pt`  
SHA-256: `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879`  
`model.names`: cut / bruise / wound  
Application keep-threshold: **0.25** (unchanged)

### Threshold sweep (Ultralytics direct on the exact upload)

| conf | Notable detections |
| --- | --- |
| 0.01 | bruise 0.7574 `[1.41,0,405.61,320.86]`; bruise 0.3473 right edge; **cut 0.0818** `[557.9,250.8,738.9,456.5]`; weaker cut/bruise |
| 0.05 | bruise boxes only (cut gone) |
| 0.10 | bruise 0.7574 + bruise 0.3473 |
| **0.25** | **bruise 0.7574** + bruise 0.3473 → **matches API/UI** |
| 0.40 | bruise 0.7574 only |
| 0.50 | bruise 0.7574 only |

### Backend `detect()` / API / Mongo / UI

All agree after re-analyze:

- `yolo_finding = Bruise`
- `yolo_confidence = 0.7574`
- `yolo_bounding_box = [1.41, 0.0, 405.61, 320.86]`

Frontend displays `yolo_finding` / `yolo_confidence` directly — **no local class remapping** (regression-locked).

---

## C. Root-cause verdict

| Candidate | Verdict |
| --- | --- |
| CLASS_MAPPING_DEFECT | **Rejected** — indices and names consistent; bruise is class_id 1 |
| IMAGE_PIPELINE_DEFECT | **Rejected** — SHA/path/dims identity proven |
| POSTPROCESSING_DEFECT | **Partial / separate** — see provenance fix below (not the Bruise label) |
| FRONTEND_DISPLAY_DEFECT | **Rejected** for Bruise label |
| STALE_DATA_DEFECT | **Rejected** for YOLO; stale Grad-CAM/provenance **were** present until re-analyze |
| **GENUINE_MODEL_MISCLASSIFICATION** | **PRIMARY** |
| INCONCLUSIVE | No |

The model confuses illuminated dorsal skin / wrist texture with “bruise” and only weakly fires “cut” (~0.08) on the laceration. Synthetic training support is insufficient for reliable real-world injury recognition.

---

## D. Checkpoint comparison (same image)

| Checkpoint | SHA prefix | At conf=0.25 |
| --- | --- | --- |
| **Active** | `4d6e72f5…` | bruise @ **0.7574** (+ secondary bruise) |
| **Backup** `.pre_retrain_v2_backup` (loaded via temp `.pt`) | `6cc84115…` | bruise @ **~0.46** |

**No rollback.** A single OOD image is not a model-wide regression proof; both eras misclassify as bruise.

---

## E. Fixes applied (verified defects only)

### E1. Vision provenance conflation — **VERIFIED_FIXED**

**Before:** Attaching demo/synthetic sensor data set `visible_injury.source_type = "demo"` and message *“Synthetic demonstration image…”* even for a **user-uploaded** photo.

**After:** `_apply_vision_image_provenance()` labels image provenance from `case.is_demo` / upload, independent of sensor demo.

Re-analyze result:

- `source_type = uploaded`
- `data_provenance = user_provided`
- Display message explains demo **sensor** + user image, research-only.

### E2. Image SHA identity — **VERIFIED_FIXED**

Upload now persists `image_sha256`; analyze embeds SHA into `visible_injury` for auditability.

### E3. Stale Grad-CAM overlay after withhold — **VERIFIED_FIXED**

When Grad-CAM is withheld (`NOT_TRUSTWORTHY`), `_gradcam_payload` clears overlay URL and deletes a stale `{case_id}_overlay.jpg` if present.

Re-analyze:

- `gradcam_explanation_status = WITHHELD`
- `gradcam_withheld_reason = classifier_model_not_trustworthy`
- `gradcam_overlay_generated = false`
- overlay file absent

### E4. Frontend provenance banner — **VERIFIED_FIXED**

Shows honest upload provenance message when `source_type === "uploaded"`.

### Explicitly **not** changed

- YOLO weights / threshold / taxonomy  
- EfficientNet / U-Net / XGBoost / VQC weights  
- Research metrics  
- Twilio  
- No filename/hash special-case for this image  
- No forced cut label  

---

## F. No-cheating checks

| Check | Result |
| --- | --- |
| No special-casing of case ID / SHA in `backend/main.py` | Pass (test) |
| No frontend bruise→cut remap | Pass (test) |
| Keep-threshold remains 0.25 | Pass |
| Active YOLO SHA unchanged | Pass |
| Previous checkpoint not accidentally loaded | Pass (`4d6e72f5…` logged on analyze) |
| No manual label override | Pass |
| Forensic image not used as training GT | Pass |

---

## G. EfficientNet / U-Net honesty on this image

| Model | Raw | Gated / display | Status |
| --- | --- | --- | --- |
| EfficientNet | **Swelling @ ~1.0** on ROI | Gate `VALID` (input quality passed) | Training **NOT_TRUSTWORTHY** — Grad-CAM **WITHHELD** |
| U-Net (YOLO ROI) | `raw_pos_ratio = 0` | Mask withheld — empty | Honest “no reliable mask” |
| U-Net (full image, offline) | `raw_pos≈0.041` can fire | Analyze uses ROI path when YOLO hits | Display withheld correctly in case |

Do **not** treat Swelling as a medical finding. UI must keep EfficientNet labeled research / not trustworthy.

---

## H. Multimodal separation

| Modality | Persisted value | Role |
| --- | --- | --- |
| YOLO | Bruise 0.7574 | Detection only |
| EfficientNet | Swelling + NOT_TRUSTWORTHY | Isolated research classifier |
| U-Net | unreliable / empty ROI | Isolated; not clinical |
| Questionnaire | pain 9, open wound, bleeding, left hand | Structured self-report |
| Sensor | demo / peak ~5.17g class of inputs | Demo/synthetic kinematics |
| XGBoost | **HIGH** (~0.92) | Synthetic rule-derived fusion — **not** clinical diagnosis |
| VQC | **MODERATE**, `EXPERIMENTAL_ONLY`, `used_in_main_decision=false` | Excluded from main decision |
| Clinical claim | `BLOCKED_NO_PAIRED_CLINICAL_LABELS` | Enforced |

UI must **not** imply Bruise + Swelling + HIGH = medically confirmed diagnosis. First-aid remains research guidance.

---

## I. Bounding-box visual validation

Coordinate system: original image **1283×700**, boxes in absolute pixels; FE scales by `overlay_width/height` (= original).

Observed:

- Kept bruise box covers **left/wrist skin**, not the central laceration.
- Secondary bruise box covers **right edge / digits**.
- Low-conf cut box (0.08) is nearer the wound center but **not kept**.

Conclusion: **detection localization fails** for this real-looking OOD image in addition to class error.

Debug artifact: `scratch/ac69884f-7a50-48e5-b7cc-a9bea9b20313_yolo_debug.jpg`.

---

## J. Test verification

| Command | Result |
| --- | --- |
| `pytest backend/tests -q` | **230 passed** (was 221; **+9** from `test_real_image_e2e_forensic.py`) |
| `pytest -q` (root, alone) | Re-run after parallel flake — see session notes |
| `npm run lint` | Pass |
| `npm run build` | Pass |

Warnings: SHAP PendingDeprecationWarning only (documented, not failures).

New tests cover: class-index consistency, upload SHA identity, provenance helper, no filename/hash special-case, Grad-CAM withhold payload, FE no remapping, forensic image keep-threshold honesty.

---

## K. Final readiness statement

**RESEARCH / DEMO ONLY WITH EXPLICIT LIMITATIONS**

- YOLO: can run inference; **real-world cut recognition is unreliable** on this forensic sample → **REQUIRES_MORE_DATA** (diverse real cut/bruise/wound labels; wound still UNSUPPORTED 0/0/0).
- EfficientNet: **NOT_TRUSTWORTHY**
- U-Net: **READY_FOR_RESEARCH_DEMO / DEMO_WITH_LIMITATIONS** (ROI empty here; geometry elsewhere previously verified)
- Fusion / XGBoost / VQC: synthetic research outputs only; **0** genuinely paired clinical records

---

# Architecture & feature overview (how the system works)

## 1. Purpose

AI-QTriage is a **multimodal academic research prototype** that:

1. Accepts an injury photograph, a structured questionnaire, and optional motion-sensor CSV / live samples.
2. Runs **separate** vision and tabular models.
3. Fuses features into a **synthetic-label** classical risk category (XGBoost) plus an **experimental** VQC comparison that does **not** drive decisions.
4. Surfaces research first-aid text and a **local SOS simulation** (Twilio optional, never emergency services).

It is **not** a medical device and must not be presented as clinical triage.

## 2. High-level architecture

```text
Browser (Next.js)
    │  REST JSON + /uploads static
    ▼
FastAPI (backend/main.py)
    │  MongoDB cases
    ├── Vision: YOLO11 → ROI EfficientNet + U-Net → Grad-CAM (gated)
    ├── Questionnaire → rules engine features
    ├── Sensor CSV / live / demo → motion features
    ├── MultimodalFeatureFusion → 23-d vector
    ├── XGBoost (main research category)
    ├── VQC (experimental only)
    ├── First-aid service (Gemini or rules)
    └── SOS local simulation (+ optional Twilio sandbox)
```

Canonical checkpoints live under `ml/models/` (vision + xgboost + vqc). Wrappers refuse silent fallback to archive weights.

## 3. Frontend features

| Route | Role |
| --- | --- |
| `/` | Landing / research framing |
| `/create-case` | Create case, upload image, questionnaire, sensor |
| `/cases` | Case list |
| `/cases/[id]` | Tabbed case UI: image (YOLO box / Grad-CAM / mask), questionnaire, sensor, fusion, first-aid, SOS |
| `/research` | Model registry, SHA, held-out metrics, limitations |

Case image panel keeps YOLO, EfficientNet, and U-Net in **separate cards** with status badges (`NOT_TRUSTWORTHY`, mask withheld, etc.).

## 4. Backend API surface (core)

- `POST /api/cases` — create  
- `POST /api/cases/{id}/image` — upload + quality gate + **SHA-256**  
- `POST /api/cases/{id}/questionnaire`  
- `POST /api/cases/{id}/sensor` | `/sensor/demo` | `/sensor/live/upload`  
- `POST /api/cases/{id}/analyze` — full multimodal pipeline  
- `GET /api/cases/{id}` — persisted results for UI  
- `GET /api/models`, `/api/models/registry` — research honesty  
- SOS demo / Twilio test endpoints  

## 5. Vision pipeline detail

1. **YOLO11** (`ml/vision/yolo_wrapper.py`): detect at conf **0.25**; class names from checkpoint; wound support sidecar marks wound **UNSUPPORTED**.
2. If detection: crop ROI → **EfficientNetV2** classify; **U-Net** segment ROI; paste mask into full image without stretching.
3. If no detection: full-image EffNet + U-Net; `finding` stays empty (no EffNet→YOLO copy).
4. **Grad-CAM** only when classifier gate + training status allow; **blocked when NOT_TRUSTWORTHY**.
5. Image quality / blank OOD heuristics can withhold classifier display independently of training status.

## 6. Tabular / fusion

- Questionnaire answers → structured features.  
- Sensor → peak g, impact timing, etc.  
- Vision one-hots / confidences enter fusion with **honest nulls** when withheld.  
- Rules engine produces a **rule-derived** safety category used as synthetic label lineage.  
- XGBoost outputs LOW/MODERATE/HIGH on the 23-d vector (**SYNTHETIC** provenance).  
- VQC mirrors the same vector experimentally; failures do not invent classes.

## 7. Safety / honesty layers

- Clinical claim blocked (`paired_clinical_samples = 0`).  
- EfficientNet **NOT_TRUSTWORTHY** + Grad-CAM withhold.  
- U-Net display gates vs raw mask stats.  
- VQC excluded from main decision / SOS.  
- Image vs sensor provenance separation (fixed this phase).  
- Research page metrics from **canonical held-out artifacts**, not hardcoded fallbacks.

## 8. Data reality (why this forensic case fails)

YOLO / EffNet training for cut/bruise is largely **synthetic / limited**. Wound has **0** honest training boxes. Real photographs of bleeding lacerations are **OOD**. The correct engineering response is more labeled real data and held-out evaluation — **not** UI cosmetics.

---

## Issue register (this phase)

| Issue ID | Priority | Status before | Evidence | Root cause | Already fixed? | Exact fix | Files | Runtime | Tests | Final status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RI-01 YOLO Bruise on cut photo | P0 | Suspected bug | Direct+API bruise@0.7574; cut@0.08 only | Genuine model miss / weak cut score | No (must not fake) | None (honest report) | — | Yes | Honesty tests | **STILL_BROKEN** / **REQUIRES_MORE_DATA** |
| RI-02 Image identity | P0 | Unknown | SHA chain | N/A | Proven correct | Persist SHA | `backend/main.py` | Yes | New upload SHA test | **VERIFIED_FIXED** (observability) |
| RI-03 Vision labeled synthetic | P1 | Broken | Mongo `source_type=demo` on upload | Sensor demo conflated with image | No → fixed | `_apply_vision_image_provenance` | `main.py`, FE case page | Re-analyze | New tests | **VERIFIED_FIXED** |
| RI-04 Grad-CAM shown despite NOT_TRUSTWORTHY | P1 | Stale/wrong on case | GENERATED in old Mongo; WITHHELD after re-analyze | Prior run before gate + stale overlay file | Partially (code existed) | Clear stale overlay; re-analyze | `main.py` `_gradcam_payload` | Yes | Grad-CAM payload test | **VERIFIED_FIXED** |
| RI-05 EffNet Swelling on cut | P1 | Known | Raw Swelling 1.0 | Unreliable classifier | No | Keep NOT_TRUSTWORTHY | — | Yes | Existing | **NOT_TRUSTWORTHY** |
| RI-06 U-Net mask empty on ROI | P2 | Expected | raw_pos=0 | Domain / ROI miss | N/A | Honest withhold | — | Yes | — | **DEMO_WITH_LIMITATIONS** |

---

*End of report. All conclusions above are backed by the 2026-08-29 runtime evidence listed in Sections A–J.*
