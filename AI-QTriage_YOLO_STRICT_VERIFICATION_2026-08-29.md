# YOLO Strict Verification Report — AI-QTriage

**Date:** 2026-08-29  
**Workflow:** VERIFY-FIRST → FIX-ONLY-IF-NEEDED → RE-VERIFY  
**Keep-threshold:** **0.25** (unchanged)  
**Production checkpoint:** **NOT rolled back**

---

## Final readiness

| Field | Value |
| --- | --- |
| Final YOLO readiness | **DEMO_WITH_LIMITATIONS** |
| Retraining decision | **REQUIRES_MORE_DATA** |
| Promotion / rollback | **No rollback** — active SHA verified correct |
| Promotion status label | **PROMOTION_VALIDATED_WITH_LIMITATIONS** |

---

## A. Active checkpoint (verified)

| Item | Evidence |
| --- | --- |
| Canonical path | `ml/models/vision/yolo11_injury_best.pt` (= `YOLO_RUNTIME`) |
| Active SHA-256 | `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879` |
| Previous SHA (backup) | `6cc84115e4cb85c8b82715211c3935200b815b76efbc95f83855c2cc988dce4f` |
| Backup path | `ml/models/vision/yolo11_injury_best.pt.pre_retrain_v2_backup` |
| Retrain artifact | `ml/models/yolo_retrain_v2/run_v2/weights/best.pt` |
| Active ≡ retrain best? | **Yes** (byte-identical SHA) |
| Wrapper loads | Active path only; backup **not** loaded |
| Registry SHA | Matches (`model_registry.json` → YOLO11 Detection) |
| Manifest SHA | Matches (`canonical_manifest.json`) |
| Live `model.names` | `{0: cut, 1: bruise, 2: wound}` |
| Infer conf | **0.25** |

**Wound candidate** `76bccb50…` exists under `yolo_wound_boxes_v1` but is **not** production.

---

## B. Class support table (production training: `yolo_retrain_v2`)

| Class | Train boxes | Val boxes | Test boxes | In `model.names` | Reliability evidence | Final status |
| --- | ---: | ---: | ---: | --- | --- | --- |
| cut | 47 | 6 | 8 | Yes | Synthetic labels; detects on held-out synth cuts @ 0.25 | **RESEARCH_DEMO_LIMITED** |
| bruise | 40 | 13 | 10 | Yes | Synthetic labels; high-conf on held-out synth bruises | **RESEARCH_DEMO_LIMITED** |
| wound | **0** | **0** | **0** | Yes (head index) | **Zero honest labels** — cannot claim accuracy | **UNSUPPORTED** |

Decision for wound: **C — marked UNSUPPORTED** (least destructive). Kept in `model.names` for contract compatibility; UI/API disclose unsupported; no accuracy claims.

Sidecar: `ml/models/vision/yolo11_class_support.json`

---

## C. Dataset provenance (`yolo_retrain_v2`)

| Attribute | Value |
| --- | --- |
| Manifest n | 126 |
| Provenance | **SYNTHETIC 126/126** |
| Sources | `raw_synthetic_wound` 122, `yolo_real_wound` 2, `yolo_injury_negative` 2 |
| Real clinical? | **No** — do not call this clinical data |
| Hash train∩val / train∩test | **0** |
| Hash val∩test | 1 group: empty-label negatives `blank_skin` (val) vs `dummy_test` (test) share **manifest pixel_sha256** but **different file SHA-256** (`1659376a…` vs `6c7f94c5…`) — not labeled-injury leakage |
| YAML note | Explicitly: `wound has 0 labels` |

---

## D. Threshold results (active model, keep=0.25)

| Image | Role | Max conf (raw) | @0.25 boxes | Wrapper @0.25 |
| --- | --- | --- | ---: | --- |
| synth cut test (×2) | labeled SYNTHETIC | 0.51 / 0.73 | 2 / 2 | cut matches |
| synth bruise test (×2) | labeled SYNTHETIC | ~0.98 | 1 | bruise matches |
| `blank_skin.jpg` | verified empty-label TN | 0 | **0** | [] |
| `dummy_test.jpg` | verified empty-label TN | 0.0106 @0.01 only | **0** | [] |
| `football_injury.jpg` | demo photo, no GT | 0.077 @0.01 | **0** | [] |
| `forensic_non_injury.png` | unlabeled OOD | 0 | **0** | [] |
| forensic black/gray/white | synthetic blanks | ≤0.0106 @0.01 | **0** | [] |

**Do not lower threshold:** football shows many weak boxes only below 0.10; at 0.25 it is clean.

---

## E. False-positive analysis

| Image | Max conf | Class | Box | Keep 0.25 | Provenance | Region note |
| --- | ---: | --- | --- | --- | --- | --- |
| blank_skin | — | — | — | clean | synthetic blank TN | — |
| dummy_test | 0.0106 | cut | corner-ish tiny | clean | 64×64 synthetic | only below keep |
| football | 0.077 | cut | border/clothing-ish | clean | demo photo | FP risk only if thr≪0.25 |
| forensic_non_injury | — | — | — | clean | unlabeled OOD | **not** a GT negative; currently 0 raw boxes on prod |
| forensic_black | 0.0106 | cut | edge | clean | synthetic | — |

**TN diversity:** blank_skin + dummy_test are **insufficient** for real-world FP claims (synthetic/blank/tiny only).

---

## F. Promotion status

| Check | Result |
| --- | --- |
| Active checkpoint decision | **KEEP active** (`4d6e72f5…`) |
| Rollback required? | **No** |
| Promotion limitations | Wound unsupported; synthetic data; small test support; TN gate synthetic-only |
| Policy note updated | `PROMOTION_VALIDATED_WITH_LIMITATIONS` in class-support sidecar + retrain `decide()` note |

---

## G. Retraining decision

**REQUIRES_MORE_DATA**

| Class | Current honest images (prod set) | Test support | Quality | Engineering target (not certification) | Retrain now? |
| --- | ---: | ---: | --- | --- | --- |
| cut | 61 labeled (synth) | 8 boxes | synthetic drawings | +100–300 **real** labeled photos, subject-aware split | No — not enough real |
| bruise | 63 labeled (synth) | 10 boxes | synthetic | +100–300 real labeled | No |
| wound | **0** | **0** | none in prod set | +200+ real wound boxes with subject split | No for prod promote |
| normal / no-injury | 2 empty-label synth | — | blank/dummy | +50–100 verified empty-label real photos + OOD | Needed before FP claims |

---

## H. Demo filename audit

- `YOLO11Detector.detect()` has **no** `football` / `is_demo` branching (source inspected + test).
- Demo uses explicit `is_demo` metadata on cases outside inference.
- `test_analyze_is_demo_not_filename_based` already exists in remediation suite.

---

## Issue-by-issue

| Issue ID | Priority | Before | Evidence | Root cause | Already fixed? | Fix applied | Files | Verify | Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Y1 Active SHA | P0 | Claimed `4d6e72f5…` | Disk+wrapper+registry+manifest match | — | Yes | None (no weight change) | — | Live SHA | **VERIFIED_FIXED** / **ALREADY_FIXED** |
| Y2 Backup not loaded | P0 | Claimed | Backup exists `6cc84115…`; runtime path excludes backup | — | Yes | Regression test | `test_yolo_reliability.py` | pytest | **ALREADY_FIXED** |
| Y3 Wound zero labels | P0 | Listed as supported | train/val/test boxes = 0 | nc=3 head without labels | Partially (YAML note) | Sidecar + API/UI UNSUPPORTED | `yolo11_class_support.json`, `yolo_wrapper.py`, `main.py`, `page.tsx` | pytest | **PARTIALLY_FIXED** → honest **UNSUPPORTED** |
| Y4 Keep thr 0.25 | P0 | Claimed | `DEFAULT_YOLO_INFER_CONF=0.25`; football/blank clean | — | Yes | None | — | Sweep | **ALREADY_FIXED** |
| Y5 Unlabeled OOD as TN | P1 | Historical misuse | forensic_non_injury / upload tests document OOD | Process | Mostly | Tests assert OOD ≠ blank TN | tests | pytest | **ALREADY_FIXED** |
| Y6 Football special-case | P1 | Suspected | No detect() filename branch | — | Yes | Explicit test | tests | pytest | **ALREADY_FIXED** |
| Y7 TN diversity | P1 | Blank/dummy only | Confirmed insufficient | Data gap | No | Documented; no invented data | sidecar | Audit | **REQUIRES_RETRAINING** / data |
| Y8 Synthetic provenance | P1 | Easy to overclaim | Manifest 100% SYNTHETIC | Data | Disclosure | Provenance in API/UI | wrapper/API/FE | Live info | **PARTIALLY_FIXED** |
| Y9 Retrain now | P0 | Tempting | Insufficient real labeled data | Data | — | **No retrain** | — | Decision | **REQUIRES_MORE_DATA** |

---

## Files changed (this pass)

- `ml/models/vision/yolo11_class_support.json` (new)
- `ml/vision/yolo_wrapper.py`
- `backend/main.py`
- `frontend/app/cases/[id]/page.tsx`
- `ml/training/train_yolo_retrain_v2.py` (limitation note)
- `backend/tests/test_yolo_reliability.py`

**Weights:** unchanged.

---

## Verification commands

```text
pytest backend/tests/test_yolo_reliability.py backend/tests/test_yolo_promotion_gates.py
→ 20 passed

pytest backend/tests -q
→ 204 passed, 3 warnings

cd frontend && npm run lint && npm run build
→ both OK
```

---

## Bottom line

1. **Active YOLO checkpoint is correct** — do not roll back.  
2. **Wound is UNSUPPORTED** (0 honest labels) — disclosed in API/UI.  
3. **Keep-threshold stays 0.25** — blank/dummy/football/forensic OOD clean at 0.25.  
4. **No retrain** until real labeled cut/bruise/wound (+ diverse negatives) exist.  
5. Readiness: **DEMO_WITH_LIMITATIONS** / research-only — not clinical.
