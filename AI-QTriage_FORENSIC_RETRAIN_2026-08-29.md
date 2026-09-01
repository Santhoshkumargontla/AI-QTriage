# AI-QTriage Forensic Remediation Report — 2026-08-29

Independent READ → MAP → RUN → TRAIN → COMPARE → RETEST pass.  
Workspace: `C:\Users\santh\Capstone Project Code`  
Hardware: CPU only (`torch 2.13.0+cpu`, `CUDA False`)  
Python: `backend\venv\Scripts\python.exe`

This report does **not** reuse previous audit numbers as truth. Every metric below is from this session’s execution unless labeled ASSUMPTION.

**Not clinical. Not production. Not a medical device.**

---

## A. Executive summary

The application **runs**: MongoDB connected, FastAPI healthy, Next.js UI loads, pytest **199 passed**.

| Model | Loads | Meaningful output? |
|---|---|---|
| YOLO11 | Yes | Boxes on some synthetic/held-out drawings; **0 boxes** on `football_injury.jpg` at keep-threshold 0.25 |
| EfficientNetV2 | Yes | Production still **collapses to Swelling** on blanks and the demo photo |
| U-Net | Yes | **Promoted** public-wound checkpoint: blanks no longer paint; held-out Dice **0.72** |
| Grad-CAM | Yes | Overlay of an **untrusted** classifier — not clinically meaningful |
| XGBoost | Yes | **SYNTHETIC** 23-d fusion |
| VQC | Yes | **EXPERIMENTAL_ONLY**, synthetic |
| Sensor | Yes | Simulated / synthetic classifier |

**Fixed in this pass (verified after change):**
- U-Net blank/black/white **raw** positive ratio **0.996 → 0.0** (production SHA `2b4967aa…` → `82419176…`)
- Dataset provenance documents that `public_wound_dataset` is **generated drawings**, not Kaggle
- Hardcoded revoked Roboflow API key removed from training/download scripts
- Questionnaire routing ignores classifier `Normal` / `Other`
- Fusion maps `Normal` onto `prob_other` (injury dims stay 0)

**Not fixed:**
- EfficientNet production still predicts Swelling ≈ 1.0 on black/white/gray
- YOLO still misses the demo photo; Roboflow download **401 revoked key**
- Mendeley healthy-feet zip **no public direct URL** (S3 403)
- XGBoost / VQC / sensor remain synthetic
- Twilio not configured — local SOS simulation only
- Browser could not open the TestClient case on the live uvicorn (`404`)

---

## B. Overall working percentage

**Evidence-based score: 60%**

Formula: `sum(weight_i * score_i) / 100`, scores 0–100 from this session only.

| Area | Weight | Score | Reason |
|---|---:|---:|---|
| Backend | 8 | 85 | Health, analyze, SOS local path, Mongo connected |
| Frontend | 8 | 68 | App loads; cases list; hydration overlay; TestClient case not visible to live API |
| YOLO | 12 | 42 | Loads; names `cut,bruise,wound`; demo miss; tiny synthetic set |
| EfficientNetV2 | 12 | 32 | Production collapse to Swelling; candidate 9→1 **not promoted** |
| U-Net | 12 | 74 | Promoted; blanks raw 0.0; Dice 0.72 on public ulcers; domain ≠ sports injury |
| Grad-CAM | 5 | 38 | Runs; tied to untrusted classifier |
| XGBoost | 5 | 55 | Loads; synthetic 23-d; TEST 25/30 from prior live eval |
| VQC | 4 | 30 | Experimental; synthetic; no fake fallback in this pass |
| Sensor | 5 | 52 | Simulated endpoints work; dataset not SisFall |
| Feature fusion | 4 | 62 | 23-d explicit missingness; Normal→other mapping added |
| SOS | 4 | 70 | Countdown/local persist |
| Twilio | 3 | 22 | `TWILIO_ENABLED=false`; SMS NOT CONFIGURED |
| Database | 4 | 80 | Mongo connected; TestClient persist works |
| Testing | 6 | 88 | **199 passed**, 3 shap warnings |
| Dataset quality | 8 | 48 | Real public wound photos for U-Net; YOLO/EN still drawings |
| Security / errors | 4 | 58 | Revoked key stripped; gates exist; Grad-CAM still follows bad classifier |
| **Total** | **100** | **60** | |

Previous session’s ~53% was before U-Net promotion. The +7 is the blank-paint fix plus public-photo Dice, not a cosmetic bump.

---

## C. Feature score table

| Feature | Working % | Bugs remaining | Errors | Warnings | Status | Evidence |
|---|---:|---|---|---|---|---|
| FastAPI + Mongo | 85 | Live vs TestClient case visibility | 0 | Starlette httpx deprecation | READY_FOR_RESEARCH_DEMO | `/api/health` healthy/connected |
| Next.js UI | 68 | Hydration overlay; header vs route mismatch | Case 404 in browser for TestClient id | Next issues badge | READY_FOR_RESEARCH_DEMO | BROWSER_VERIFIED cases list + error card |
| YOLO11 | 42 | Demo FN; tiny set; near-dup leakage | 0 | — | NEEDS_RETRAINING | Direct+API n=0 on football_injury |
| EfficientNetV2 | 32 | Swelling collapse | 0 | — | NOT_TRUSTWORTHY | Raw Swelling 1.0 black/white |
| U-Net | 74 | Ulcer domain; drawing overlays weak | 0 | — | READY_FOR_RESEARCH_DEMO | Raw blank 0.0; test Dice 0.7169 |
| Grad-CAM | 38 | Follows untrusted EN | 0 | — | EXPERIMENTAL_ONLY | Analyze produced overlay URL |
| XGBoost | 55 | Synthetic only | 0 | Option A missing-modality notice | EXPERIMENTAL_ONLY / SYNTHETIC | Analyze class MODERATE |
| VQC | 30 | Weak, synthetic | 0 | — | EXPERIMENTAL_ONLY | Analyze HIGH, unused in decision |
| Sensor | 52 | Simulated kinetics | 0 | — | EXPERIMENTAL_ONLY | skip + simulated scenarios 200 |
| SOS / Twilio | 40 | No real SMS | 0 | — | LOCAL SOS SIMULATION ONLY | sos_config + trigger 200 |
| Questionnaire | 70 | Model-dependent template still limited | 0 | — | READY_FOR_RESEARCH_DEMO | POST 200; routing ignores untrusted EN |
| Tests | 88 | None failing | 0 | 3 shap PendingDeprecation | READY_FOR_RESEARCH_DEMO | 199 passed |

---

## D. Before vs after

| Issue | Before | Root cause | Fix | After | Verification |
|---|---|---|---|---|---|
| U-Net paints black/white | Raw pos. ratio 0.996 / 0.999 | Frozen encoder + no empty-mask targets + 0/255 vs 0/1 mask bug on Medetec | Train on wseg+Medetec + 31 synthetic empty; `mask>0`; unfreeze layer4; promote only if CORE_WATCH area ≤ 0.05 | Raw 0.0 on black/white/gray | Live wrapper 2026-08-29; pytest `test_black_white_gray_withheld_but_raw_fp_recorded` |
| U-Net metadata claimed Dice 0.98 public Kaggle | Stale metadata / generated drawings | Fabricated provenance in `download_public_datasets.py` | Honest metadata from this train | test n=79 Dice 0.717 | `unet_metadata.json` SHA `82419176…` |
| EN Swelling on every blank | Raw Swelling ~1.0 | Closed 3-class softmax, majority swelling, no reject class | Trained candidate `cut/bruise/normal` | Collapse **9 → 1** (white still cut@1.0) | KEEP_BASELINE; production SHA unchanged `6944605a…` |
| Roboflow key in repo | Hardcoded default | Secret in source | Env-only | 401 if unset | Scripts no longer embed a key |
| Routing `Normal` | Would become questionnaire template | Missing reject handling | `_routing_finding` drops normal/other | `""` | `test_routing_helpers_tolerate_none_confidence` |
| YOLO new cut/bruise photos | No new data | Roboflow 401 | None (blocked) | Unchanged | Download log |

U-Net outcome: **VERIFIED_FIXED** for blank painting.  
EfficientNet outcome: **PARTIALLY_FIXED** (candidate only) / production **NOT_FIXED**.  
YOLO outcome: **BLOCKED_BY_EXTERNAL_CREDENTIALS** for new public boxes.

---

## E. Model status table

| Model | Loads | Trained | Inference | Accuracy/metrics | Blank/OOD | API | Frontend | Status |
|---|---|---|---|---|---|---|---|---|
| YOLO11 | Yes | Yes (synthetic retrain_v2) | Yes | Prior: cut recall 0.875 on 8 test boxes; demo 0 det | 0 boxes @0.25 on true blanks | Yes | Boxes only if detected | INFERENCE_EXECUTES / NEEDS_RETRAINING |
| EfficientNetV2 | Yes | Yes (old 3-class) | Yes | Candidate test acc 1.0 on 27 drawings+synthetics (**not generalization**) | Production: Swelling 1.0 | Yes | Shows Swelling | NOT_TRUSTWORTHY |
| U-Net | Yes | Yes (this session, 8 ep) | Yes | test Dice 0.717 / IoU 0.591 / FP area 0.023 | Raw 0.0 black/white | Yes | Overlay URL on analyze | READY_FOR_RESEARCH_DEMO |
| Grad-CAM | Yes | N/A | Yes | N/A | Follows EN | Yes | Overlay | EXPERIMENTAL_ONLY |
| XGBoost | Yes | Synthetic | Yes | Prior held-out 25/30 | N/A | Yes | Shown | TRAINED / SYNTHETIC |
| VQC | Yes | Synthetic | Yes | Prior 16/30 | No fake fallback | Yes | Labeled experimental | EXPERIMENTAL_ONLY |
| Sensor | Yes | Synthetic | Yes | N/A | Explicit skip | Yes | Motion cards | TRAINED / SYNTHETIC |

---

## F. YOLO detailed report

**Runtime checkpoint:** `ml/models/vision/yolo11_injury_best.pt`  
**SHA-256:** `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879`  
**Backup (do not delete):** `yolo11_injury_best.pt.pre_retrain_v2_backup` (`6cc84115…`)  
**model.names:** `cut`, `bruise`, `wound`  
**Keep-threshold:** 0.25 (not lowered to manufacture boxes)

**Dataset (active):** `data/datasets/yolo_retrain_v2` — SYNTHETIC drawings + 2 empty negatives. Split 87/20/19. Exact-hash disjoint; **near-duplicate cross-split** remains. Test wound boxes: **0**.

**This session:** no YOLO retrain. Roboflow Universe candidates (self-harm bruises/cuts, injury-segmentation) **failed 401**. No new boxes to beat the true-negative + cut-recall gates.

**Unseen / demo:** `football_injury.jpg` — **0 detections** direct and API.  
**Blank:** 0 boxes @0.25 (unchanged).

**Recommended threshold:** keep **0.25**. Do not drop to 0.10.

**Wrong localization:** not re-evaluated with new weights; previous forensic still applies (drawings, wound-heavy history, bruise support small).

---

## G. Dataset report

### Comparison of candidates (this session)

| ID | License | Size | Domain match to cut/bruise/swelling | Real? | Downloaded? | Use |
|---|---|---|---|---|---|---|
| Mendeley hsj38fwnvr v2 | CC BY 4.0 | 8129 (incl. 2575 normal feet) | No (feet/wound vs sports injury) | Real photos | **No** — API 400, S3 403, public-files 404 | Would have been best `normal` class |
| HF `subbareddyoota/wseg_dataset` | **CC-BY-NC-4.0** | 2686+2686 | No (binary wound) | Real photos | **Yes** 50.3 MB zip | U-Net positives (338 unique after cap/filter) |
| UWM Medetec 224 | Medetec terms + UWM redistribution | 152 train + 8 test pairs | No (foot ulcer) | Real photos | **Yes** git sparse | U-Net positives (152 kept) |
| FUSeg | Challenge / AZH permission | ~1210 | No | Real | Skipped (CPU + license not CC BY) | — |
| Roboflow self-harm / injury | CC BY 4.0 listed | 293 / unknown | **Partial** (cuts/bruises) | Unverified | **No** 401 | YOLO/EN blocked |
| WILLIE benchmark | mixed / index only | index | No | N/A | No | Images not granted |
| Local `public_wound_dataset` | Claimed CC BY | 200 | Labels cut/bruise/swelling | **SYNTHETIC PIL drawings** | Already on disk | Not “Kaggle” |
| `efficientnet_processed` | internal | 126 unique | cut 62 / bruise 64 / swelling **2 unique** | SYNTHETIC | Yes | EN injury classes |
| YOLO retrain_v2 | internal | 126 | cut/bruise/wound | SYNTHETIC | Yes | YOLO |

**Do not remap** ulcer/healthy-foot/surgical classes to swelling. That was not done.

**U-Net train set actually used:** `data/datasets/unet_public_real/` n=521 (490 positive, 31 empty), splits 363/79/79, hash overlap 0. Exclusions 70 (empty/full/unreadable).  
**License mix:** CC-BY-NC wound photos + Medetec stock + synthetic empty canvases. Academic/non-commercial for wseg.

---

## H. Training report

### EfficientNetV2 (`ml.training.train_efficientnet_with_normal`)

- **Executed:** Yes, CPU, 9 epochs (early stop patience 4, best @5)
- **Classes:** `cut`, `bruise`, `normal` (swelling omitted: 2 unique drawings)
- **n:** 176 (bruise 64, cut 62, normal 50 SYNTHETIC_REJECT)
- **Unfreeze:** classifier + blocks.4 + blocks.5
- **Val acc:** 1.0 by epoch 5 (memorization of drawings + canvases)
- **Test acc:** 1.0 on 27 rows — **not** a quality proof
- **OOD collapse:** baseline 9 → candidate **1** (white → cut @ 1.0)
- **Promotion:** **KEEP_BASELINE** — production **not** overwritten
- **Candidate SHA:** `b569e6cc044f7560027b1fe0638a8213f2738f7080ff816f7c3744705d7f62c3`
- **Retrain still required:** Yes — need real healthy-skin / diverse whites or a dedicated OOD head

### U-Net (`ml.training.train_unet_public`)

- **Executed:** Yes, CPU, 8 epochs, ~19 min after EN
- **Best epoch:** 6 (val loss 0.5577, val Dice 0.694)
- **Test:** n=79, Dice **0.716938**, IoU 0.591, prec 0.746, rec 0.776, FP area 0.023
- **Empty-mask train:** 21
- **OOD CORE_WATCH:** 5 → **0**
- **Promotion:** **PROMOTE**
- **New SHA:** `82419176b5cf3ae4049a6fbab9d4af3cc956a74fef76e55031b0dcbf573f3e93`
- **Backup:** `ml/models/vision/unet_injury_best.pt.pre_public_real_backup`
- **Retrain still required:** For sports-injury masks, yes. For binary chronic-wound demo, this is the best honest checkpoint we have.

### YOLO / XGB / VQC / Sensor

Not retrained this session (no new licensed boxes; no paired clinical multimodal data).

---

## I. Model output report

| Input | YOLO | EfficientNet raw | U-Net raw pos. | U-Net display | Verdict |
|---|---|---|---|---|---|
| Black 224 | — | Swelling 1.0 | **0.0** | withheld LOW_QUALITY | EN incorrect; U-Net correct empty |
| White 224 | — | Swelling 1.0 | **0.0** | withheld | same |
| Gray 180 | — | Swelling 0.96 | **0.0** | withheld | same |
| football_injury.jpg | 0 boxes | Swelling 0.999 | 0.022 | VALID 1930 px | YOLO miss; EN untrustworthy; U-Net small mask **uncertain** (not a labeled ulcer) |
| Held-out wseg_45277581 | n/a | — | 0.055 | VALID 5946 px | **partially correct** binary wound (public photo) |
| EN candidate on white | — | **cut 1.0** | — | — | still untrustworthy |
| EN candidate on black/gray | — | **normal ~1.0** | — | — | reject class works except white |

Analyze E2E (`ef953f3b-…` via TestClient): YOLO miss; classifier Swelling 0.988; finding **null** (untrusted classifier not used for routing); U-Net affected_ratio **0.021**; XGB MODERATE synthetic; VQC HIGH experimental unused.

---

## J. Bug report

### Fixed (VERIFIED_FIXED)

| Sev | Bug |
|---|---|
| HIGH | U-Net near-full masks on black/white |
| MEDIUM | U-Net metadata advertised leaked ~0.98 Dice as public Kaggle |
| MEDIUM | Hardcoded revoked Roboflow key in scripts |
| LOW | Routing would treat `Normal` as an injury template |

### Partially fixed

| Sev | Bug |
|---|---|
| CRITICAL | EfficientNet Swelling collapse — candidate 9→1, production unchanged |

### Remaining

| Sev | Bug |
|---|---|
| CRITICAL | EfficientNet still Swelling on blanks **and** demo photo through API |
| HIGH | YOLO misses unlabeled real photos; no new licensed detection set |
| HIGH | No real cut/bruise/swelling photo taxonomy |
| HIGH | XGB/VQC/sensor synthetic; must not be cited as clinical |
| MEDIUM | Live uvicorn 404 for TestClient-created case (process/env split) |
| MEDIUM | Frontend hydration error overlay; “New Assessment” header on case routes |
| MEDIUM | Grad-CAM still explains an untrusted class |
| LOW | wseg CC-BY-NC: non-commercial only |
| LOW | Shap colormap PendingDeprecation warnings |

Technical debt: duplicate YOLO checkpoints under archive; `download_public_datasets.py` still *can* generate fake “public” drawings if someone runs it; large `data/datasets/external/` should not be committed blindly.

---

## K. Application data organization

| Kind | Location |
|---|---|
| Uploads / overlays | `data/uploads/` (`{case_id}.jpg`, `_overlay.jpg`, `_mask.png`) |
| Cases | MongoDB `ai_qtriage.cases` |
| Canonical models | `ml/models/vision/*_injury_best.pt`, `ml/models/xgboost_best.json`, `ml/models/vqc/` |
| Path helper | `ml/models/canonical_paths.py` (ROOT-absolute) |
| U-Net public set | `data/datasets/unet_public_real/` |
| EN reject experiment | `data/datasets/efficientnet_with_normal/` |
| External downloads | `data/datasets/external/` (wseg zip+extract, Medetec git) |
| Training reports | `ml/models/unet_public_training/`, `ml/models/efficientnet_normal_training/` |
| U-Net backup | `unet_injury_best.pt.pre_public_real_backup` |

Cleanup: keep backups; prefer `archive/legacy/` for old YOLO duplicates already documented; do not delete `pre_retrain_v2_backup`.

---

## L. Final readiness table

| Component | Status |
|---|---|
| Backend / Mongo / tests | READY_FOR_RESEARCH_DEMO |
| Frontend | READY_FOR_RESEARCH_DEMO (with hydration noise) |
| YOLO11 | NEEDS_RETRAINING |
| EfficientNetV2 | NOT_TRUSTWORTHY / NEEDS_RETRAINING |
| U-Net | READY_FOR_RESEARCH_DEMO (ulcer-domain binary masks only) |
| Grad-CAM | EXPERIMENTAL_ONLY |
| XGBoost | EXPERIMENTAL_ONLY (SYNTHETIC) |
| VQC | EXPERIMENTAL_ONLY |
| Sensor | EXPERIMENTAL_ONLY (SYNTHETIC) |
| SOS | READY_FOR_RESEARCH_DEMO (local) |
| Twilio | BLOCKED (no credentials) |
| Production clinical use | NOT_READY |

---

## M. Final honest verdict

1. **Is the application working?** Yes, as a research prototype: APIs, UI, Mongo, analyze pipeline.
2. **How much is working?** About **60%** by the weighted table in §B.
3. **What is still broken?** EfficientNet Swelling collapse; YOLO on real photos; no real multimodal data; no SMS; limited browser/API process coupling.
4. **Most reliable model?** **U-Net** after this promotion (empty on blanks, modest Dice on public wound photos). Still not sports-injury segmentation.
5. **Least reliable?** **EfficientNetV2 production** (confident Swelling on blanks) and **VQC** (experimental synthetic).
6. **Is YOLO detecting the correct region?** On the demo photo, **it detects nothing**. On some synthetic test drawings it can box bruise/cut. Not trustworthy on unseen photos.
7. **Is EfficientNet still predicting swelling for every image?** **Yes, on production.** Black/white/gray/demo still Swelling at the raw softmax. Quality gates hide uniforms; they do **not** hide Swelling on `football_injury.jpg` (API 0.988).
8. **Is U-Net generating false masks?** **Not on blanks anymore** (raw 0.0). On the demo it emits a small VALID mask (2.1%) — unlabeled, so **uncertain**, not a full-image paint.
9. **Does the frontend match the backend?** TestClient analyze vs get_case matched. Live browser could **not** load that case (`404`). Overlay geometry was previously source-verified; **not re-browser-verified** on the new U-Net mask.
10. **Is Twilio sending messages?** **No.** Local simulation only.
11. **Suitable for a research demonstration?** **Yes, with loud caveats** — especially the new U-Net blank behavior and honest labels.
12. **Suitable for production?** **No.**
13. **Clinically validated?** **No.**

---

## Verification log (this session)

| Check | Result |
|---|---|
| pytest | **199 passed**, 3 warnings, ~101s |
| `/api/health` | healthy, database connected |
| U-Net SHA live | `82419176b5cf3ae4049a6fbab9d4af3cc956a74fef76e55031b0dcbf573f3e93` |
| EN SHA live | `6944605ae71685d9…` (unchanged) |
| YOLO SHA live | `4d6e72f5f671fd60…` (unchanged) |
| Analyze E2E | HTTP 200, Mongo analyzed |
| Browser | BROWSER_VERIFIED: `/cases` list, case-not-found error UI. Overlay of new mask: **SOURCE_LEVEL_VERIFIED** only |

---

## Architecture (verified)

Frontend (`/`, `/create-case`, `/cases`, `/cases/[id]`, `/research`)  
→ FastAPI `backend/main.py`  
→ case create / image upload / questionnaire / optional sensor  
→ analyze: YOLO → U-Net → EfficientNet → Grad-CAM → 23-d fusion → rules → XGBoost → VQC (experimental, unused in main decision)  
→ Mongo `ai_qtriage.cases`  
→ SOS is **not** inside analyze; Twilio only if configured.
