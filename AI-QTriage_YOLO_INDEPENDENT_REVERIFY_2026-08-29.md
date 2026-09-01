# YOLO Independent Re-Verification Report — AI-QTriage

**Date:** 2026-08-29 (strict VERIFY-FIRST pass; prior report not trusted)  
**Workspace:** `C:\Users\santh\Capstone Project Code`  
**Weights modified:** **No**  
**Rollback:** **Not required**

---

## Final verdict

**RESEARCH / DEMO ONLY WITH EXPLICIT LIMITATIONS**

| Overall YOLO | cut | bruise | wound |
| --- | --- | --- | --- |
| DEMO_WITH_LIMITATIONS | RESEARCH_DEMO_LIMITED | RESEARCH_DEMO_LIMITED | **UNSUPPORTED** |

**Promotion decision:** `PROMOTION_VALIDATED_WITH_LIMITATIONS`  
**Retraining:** `REQUIRES_MORE_DATA` (real labeled photos + diverse negatives)

---

## A. Active checkpoint (independently computed)

| Source | Full SHA-256 |
| --- | --- |
| Disk `yolo11_injury_best.pt` | `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879` |
| Wrapper `artifact_sha256` | `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879` |
| `model_registry.json` | `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879` |
| `canonical_manifest.json` | `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879` |
| Retrain `run_v2/weights/best.pt` | `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879` (byte-identical) |
| Live `/api/models` | `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879` |
| **Previous backup** `.pre_retrain_v2_backup` | `6cc84115e4cb85c8b82715211c3935200b815b76efbc95f83855c2cc988dce4f` |

| Check | Result |
| --- | --- |
| Canonical path | `ml/models/vision/yolo11_injury_best.pt` (= `YOLO_RUNTIME`) |
| Backup loaded by runtime? | **No** |
| `YOLO_CONF_THRESHOLD` env | unset (default 0.25) |
| Keep-threshold | **0.25** |
| `model.names` | `{0:cut, 1:bruise, 2:wound}` |

---

## B. Class support (independent label recount)

Dataset: `data/datasets/yolo_retrain_v2`  
Sidecar mismatch vs recount: **none**

| Class | Train images w/ class | Train boxes | Val images | Val boxes | Test images | Test boxes | Provenance | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cut | 47 | **47** | 6 | **6** | 8 | **8** | SYNTHETIC | RESEARCH_DEMO_LIMITED |
| bruise | 40 | **40** | 13 | **13** | 10 | **10** | SYNTHETIC | RESEARCH_DEMO_LIMITED |
| wound | 0 | **0** | 0 | **0** | 0 | **0** | none | **UNSUPPORTED** |

Manifest: 126/126 `SYNTHETIC`. Not real clinical data.

Empty-label negatives: 1 in val (`blank_skin`), 1 in test (`dummy_test`).

---

## C. Threshold verification (raw vs application)

**A = raw model @ listed thr · B = application wrapper @ 0.25**

| Image | Role | Raw max @0.01 | n@0.01 | n@0.05 | n@0.10 | n@0.25 | n@0.40 | n@0.50 | App @0.25 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synth cut test | labeled | 0.5069 | 7 | 2 | 2 | 2 | 1 | 1 | 2 cut |
| synth bruise test | labeled | 0.9988 | 4 | 1 | 1 | 1 | 1 | 1 | 1 bruise |
| synth cut val | labeled | 0.3743 | 3 | 3 | 3 | 1 | 0 | 0 | 1 cut |
| blank_skin | empty-label TN | 0.0 | 0 | 0 | 0 | **0** | 0 | 0 | **0** |
| dummy_test | empty-label TN | 0.0106 | 1 | 0 | 0 | **0** | 0 | 0 | **0** |
| football_injury | demo unlabeled | 0.0772 | 23 | 2 | 0 | **0** | 0 | 0 | **0** |
| forensic_non_injury | unlabeled OOD | 0.0 | 0 | 0 | 0 | **0** | 0 | 0 | **0** |
| forensic_black | blank control | 0.0106 | 1 | 0 | 0 | **0** | 0 | 0 | **0** |
| forensic_gray | blank control | 0.0 | 0 | 0 | 0 | **0** | 0 | 0 | **0** |
| forensic_white | blank control | 0.0 | 0 | 0 | 0 | **0** | 0 | 0 | **0** |

**Critical distinction:** football has raw max **0.0772** and 23 boxes at 0.01, but **zero application detections at 0.25**. That is thresholding, not “model never fires.”

---

## D. Negative / OOD audit

| Image | Classification | GT negative? | Notes |
| --- | --- | --- | --- |
| blank_skin | synthetic blank + empty YOLO label | **Valid empty-label TN** (synthetic) | Clean even at 0.01 |
| dummy_test | synthetic tiny + empty label | **Valid empty-label TN** (synthetic) | Raw 0.0106 only; app clean @0.25 |
| forensic black/gray/white | blank controls | Blank controls | Not real-world negatives |
| football_injury | demo photo | **Unlabeled** — not GT negative | App clean @0.25; raw weak FPs below 0.10 |
| forensic_non_injury | unlabeled OOD | **Unlabeled OOD** — not GT negative | Currently 0 raw boxes on active model |

TN diversity is **insufficient** for real-world FP claims (synthetic/blank-only gates).

Forensic upload test asserts `max_conf < infer_conf (0.25)` when raw boxes exist — **valid app-level check**, not an invalid “max&lt;0.05 on unlabeled” requirement.

---

## E. Checkpoint comparison (active vs backup)

Same images / thresholds (backup loaded via temp `.pt` copy — Ultralytics rejects `.pre_retrain_v2_backup` suffix directly).

| Image | Active n@0.25 | Backup n@0.25 | Active rawmax@0.01 | Backup rawmax@0.01 |
| --- | ---: | ---: | ---: | ---: |
| blank | 0 | 0 | 0 | 0 |
| dummy | 0 | 0 | 0.0106 | 0.0117 |
| football | **0** | **3** | 0.0772 | 0.2982 |
| forensic_ood | 0 | 0 | 0 | 0 |

Active is **better** on football FN/FP at keep-threshold (0 vs 3). **No rollback.**

---

## F. Filename special-case

- `yolo_wrapper.detect()` contains no `football` / `is_demo` logic.
- Same bytes under `football_injury.jpg`, `random_name.jpg`, UUID filename → **identical** raw@0.01 and app@0.25 outputs.

---

## G. Promotion policy

`train_yolo_retrain_v2.decide()` checks: SHA uniqueness, training log length, box loss decrease, cut recall improvement, bruise non-regression (&gt;5pp), blank_skin/dummy_test TN at keep=0.25, missing sweep rows. Notes wound unsupported + synthetic TN limits. Loss alone is not sufficient.

---

## H. API / frontend honesty

| Surface | Wound / provenance |
| --- | --- |
| `/api/models` | `unsupported_classes: [wound]`, `validated_classes: [cut,bruise]`, `promotion_status`, `dataset_provenance`, `wound_note` |
| Case page | Shows unsupported list + UNSUPPORTED banner if detected |
| Research page | **Was missing** disclosure → **fixed this pass** to show unsupported / provenance / promotion / wound_note |

---

## Issue table

| Issue ID | Priority | Before | Evidence | Already fixed? | Fix | Final |
| --- | --- | --- | --- | --- | --- | --- |
| R1 Active SHA consistency | P0 | Claimed match | Disk/wrapper/registry/manifest/API all `4d6e72f5…` | Yes | None | **VERIFIED_FIXED** / **ALREADY_FIXED** |
| R2 Backup not loaded | P0 | Claimed | Backup `6cc84115…`; runtime path excludes it | Yes | None | **ALREADY_FIXED** |
| R3 Keep thr 0.25 | P0 | Claimed | DEFAULT + wrapper infer_conf = 0.25; env unset | Yes | None | **ALREADY_FIXED** |
| R4 Wound zero labels | P0 | Claimed UNSUPPORTED | Recount 0/0/0 boxes; sidecar matches | Yes | None (weights) | **UNSUPPORTED** (honest) |
| R5 Raw vs app threshold | P0 | Easy to conflate | Football raw 0.077 / app 0 | Documented | Report only | **ALREADY_FIXED** (behavior) |
| R6 Filename bias | P1 | Claimed none | Clone test identical | Yes | None | **ALREADY_FIXED** |
| R7 Active vs backup | P1 | Claimed better | Football 0 vs 3 @0.25 | Yes | No rollback | **ALREADY_FIXED** |
| R8 Research UI taxonomy | P2 | Case OK, research incomplete | Research omitted unsupported | **Partially** | `research/page.tsx` | **VERIFIED_FIXED** |
| R9 Real clinical data | P0 | Missing | 100% SYNTHETIC train set | No | Cannot invent | **REQUIRES_MORE_DATA** |
| R10 TN diversity | P1 | Blank/dummy only | Confirmed | No | Document | **REQUIRES_MORE_DATA** |

**Files changed this pass:** `frontend/app/research/page.tsx` only.

---

## I. Test results

| Suite | Result |
| --- | --- |
| `test_yolo_reliability` + `test_yolo_promotion_gates` + `test_canonical_paths` | **31 passed** |
| Full `pytest backend/tests` | **204 passed**, 3 shap warnings |
| `npm run lint` + `npm run build` | **OK** |

Warnings: shap colormap PendingDeprecation (harmless technical debt).

---

## Bottom line

Independent re-verification **confirms** the prior YOLO reliability narrative with evidence:

1. Active checkpoint `4d6e72f5…` is correct everywhere — **do not roll back**.  
2. Wound remains in `model.names` but is **UNSUPPORTED** (0 honest labels).  
3. Keep-threshold **0.25** correctly suppresses weak raw FPs (e.g. football).  
4. Only code fix this pass: research page taxonomy honesty.  
5. Status remains **RESEARCH / DEMO ONLY WITH EXPLICIT LIMITATIONS**.
