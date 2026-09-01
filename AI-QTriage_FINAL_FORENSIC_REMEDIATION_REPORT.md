# AI-QTriage Final Forensic Remediation Report

> **Current source of truth (2026-08-29 cleanup):** repository `README.md` plus
> `ml/models/model_registry.json`. Duplicate checkpoints were moved to
> `ml/models/_archive/` (not deleted). Pytest counts and some SHA/path claims
> in this report may be older than the live registry.

**Date:** 2026-08-28  
**Workspace:** `c:\Users\santh\Capstone Project Code`  
**Method:** Independent inspection and execution of the current repository. Prior audit JSON, README claims, registry notes, and “verified” labels were not trusted unless reproduced.

---

## 1. Executive verdict

```text
PARTIALLY_WORKING_RESEARCH_PROTOTYPE
```

The application pipeline runs: FastAPI loads models, analyze writes MongoDB fields, pytest is green, and a TestClient end-to-end demo produced the **same YOLO box** as direct inference. That is **code works + inference executes**. It is **not** clinical accuracy, and several vision models remain **not trustworthy** on blank/OOD inputs.

| Claim | Status |
| ----- | ------ |
| Code works | Yes (pytest 111 passed; analyze 200) |
| Model loads | Yes for YOLO, EfficientNet, U-Net, XGBoost, VQC, sensor |
| Model inference executes | Yes |
| Model is actually trained | YOLO candidate: real Ultralytics train. EfficientNet/U-Net: real `loss.backward()` / `opt.step()`. VQC: 1 real epoch, matching circuit. XGBoost: prior real train on synthetic fusion data |
| Model is accurate | **No clinical claim.** YOLO val set is 17 images. EfficientNet test acc 1.0 on 30 images **and** collapses on blank images. U-Net Dice 0.98 **and** paints blank images. VQC test acc 0.367 |
| End-to-end workflow works | Yes for demo image + questionnaire + analyze + local SOS simulation. Sensor simulate requires `football_fall` / `sudden_fall` / `sudden_impact` / `normal_movement`. Twilio is **not configured** (honest). Live browser click-through was **not** executed in this session |

---

## 2. Model status table

| Model | Loads | Inference | Actually Trained | Real Metrics | API | Frontend | E2E | Status |
| ----- | ----- | --------- | ---------------- | ------------ | --- | -------- | --- | ------ |
| YOLO11 detect | Yes | Yes | Yes (8-epoch Ultralytics on `yolo_merged`) | Yes (`ultralytics.val`) | Yes | Box from API `bounding_box` (code) | Direct = stored box on demo | `TRAINED_AND_EVALUATED` / `INFERENCE_EXECUTES` |
| EfficientNetV2 | Yes | Yes | Yes (8-epoch head-only, public_wound_dataset) | Yes (held-out + blank probes) | Yes | Classifier fields | Analyze used it | `NOT_TRUSTWORTHY` (blank collapse) |
| ResNet34-UNet | Yes | Yes | Yes (6-epoch BCE+Dice) | Yes (held-out + blank probes) | Yes | Overlay withheld if untrustworthy | Analyze produced a mask on demo ROI | `MODEL_OUTPUT_NOT_TRUSTWORTHY` (blank/black/white) |
| Grad-CAM | Yes | Heatmap runs | N/A (visualization) | N/A | Stored overlay | Labeled unreliable | Overlay written | `VISUALIZATION_NOT_CLINICALLY_OR_MODEL_RELIABLE` |
| XGBoost | Yes | Yes | Yes (synthetic 23-d fusion) | Yes (25/30 = 0.833) | Yes | Shows class | Analyze `MODERATE` | `TRAINED_AND_EVALUATED` |
| VQC | Yes | Yes | Partial (1 epoch, 24 train samples, matching circuit) | Yes (11/30 = 0.367) | Failure is `MODEL_UNAVAILABLE`, not a fake label | No hardcoded 0.33 scores | Analyze `EXPERIMENTAL` / `LOW` | `EXPERIMENTAL` |
| Sensor classifier | Yes if artifacts exist | Yes when features present | Prior train | Prior | Flattened onto `sensor_summary` | Motion class shown | Simulate 400 if wrong scenario name | `INFERENCE_EXECUTES` when features present |
| Twilio | Config reader | Send path exists | N/A | N/A | `/api/sos/config` | **SMS NOT CONFIGURED** / **LOCAL SOS SIMULATION ONLY** | Local countdown 200; no SMS | `SIMULATED` / not configured |

---

## 3. YOLO forensic report

### Actual checkpoint (runtime)

| Item | Value |
| ---- | ----- |
| Canonical path | `ml/models/vision/yolo11_injury_best.pt` |
| SHA-256 | `6cc84115e4cb85c8b82715211c3935200b815b76efbc95f83855c2cc988dce4f` |
| Task | `detect` |
| `model.names` | `{0: cut, 1: bruise, 2: wound}` |
| Advertised classes | **Exactly** `model.names`. `abrasion`, `laceration`, `swelling` = **UNTRAINED_CLASS** |
| Previous forensic checkpoint (kept) | `ml/models/vision/yolo11_injury_best.pt.pre_retrain_backup` and `ml/models/yolo_real_training/run_real_wound/weights/best.pt` SHA-256 `f4382450494c2e9009409a751a9535cf0cb1c5381b6c3866b4dacc070f06dd63` |

Wrapper loads `YOLO_CANONICAL` only (env `YOLO_MODEL_VERSION=synthetic_baseline` still points at `ml/models/yolo11n_best.pt`).

Byte-identical copies of the **promoted candidate** (SHA `6cc84115…`):

- `ml/models/vision/yolo11_injury_best.pt`
- `ml/models/vision/yolo11_merged_candidate.pt`
- `ml/models/yolo11_real_wound_best.pt`
- `ml/models/yolo_candidate_training/run_merged/weights/best.pt`

They remain as duplicates. Prefer the canonical vision path; extras are leftover copies, not a second production model.

### Dataset (active merged training set)

Built by `ml/training/train_yolo.py` into `data/datasets/yolo_merged` (real_wound identity map + yolo_injury abrasion/laceration remapped → wound).

| Class | Train images | Train boxes | Val images | Val boxes | Test images | Test boxes |
| ----- | -----------: | ----------: | ---------: | --------: | ----------: | ---------: |
| cut | 15 | 15 | 2 | 2 | 5 | 5 |
| bruise | 12 | 12 | 1 | 1 | 4 | 4 |
| wound | 56 | 288 | 14 | 70 | 14 | 65 |
| abrasion | 0 | 0 | 0 | 0 | 0 | 0 |
| laceration | 0 | 0 | 0 | 0 | 0 | 0 |

`abrasion` / `laceration` / `swelling` = **UNTRAINED_CLASS**. Original `yolo_real_wound` train/val had **0 cut and 0 bruise** boxes (wound-only). That was the main reason the old checkpoint missed typical images at conf 0.10.

JPG/PNG duplicate stems exist in the source Roboflow-style folders; merge copies unique stems as `.jpg`.

### Retrain vs baseline (executed)

| | Precision | Recall | mAP50 | mAP50-95 |
| - | --------: | -----: | ----: | -------: |
| Baseline `f4382450…` on merged val | 0.0166 | 0.638 | 0.349 | **0.268** |
| Candidate 8-epoch YOLO11n CPU | 0.822 | 0.523 | 0.820 | **0.679** |

Candidate was **better** on the same merged val split. Canonical weights were replaced; baseline kept as `.pre_retrain_backup`.

Ultralytics train-val printout (17 images, **tiny** cut/bruise support):

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
| ----- | -----: | --------: | -: | -: | ----: | -------: |
| cut | 2 | 2 | 1.0 | **0.0** | 0.535 | 0.319 |
| bruise | 1 | 1 | 0.515 | 1.0 | 0.995 | 0.895 |
| wound | 14 | 70 | 0.952 | 0.568 | 0.929 | 0.823 |
| all | 17 | 73 | 0.822 | 0.523 | 0.820 | 0.679 |

**Best operating threshold (from sweep, not a clinical optimum):** application `detect(conf=0.10)` now fires on typical val images. At 0.10 the demo image has **13** boxes (noisy). At 0.25, 3 boxes; at 0.40, 0 on the demo. A research demo should treat 0.10 as high-recall / high-false-positive.

### Threshold sweep (promoted checkpoint)

Evidence: `scratch/yolo_threshold_sweep.json`.

| Image | 0.01 | 0.05 | 0.10 | 0.25 | 0.40 | 0.50 |
| ----- | ---: | ---: | ---: | ---: | ---: | ---: |
| demo `football_injury.jpg` | 86 | 19 | **13** | 3 | 0 | 0 |
| val `public_sample_0058.jpg` | 28 | 14 | **6** | 4 | 2 | 0 |
| val `public_sample_0088.jpg` | 27 | 13 | **10** | 3 | 1 | 0 |
| train `public_sample_0006.jpg` | 7 | 0 | **0** | 0 | 0 | 0 |

**Before retraining**, the same val images had **zero** boxes at 0.10 (only sub-0.05). **Root cause of missing detections was weak/low-confidence training on a wound-only 24-image set, not frontend serialization.** After merge+retrain, detections exist at the app threshold on most sampled val images.

### Direct vs API vs stored (demo)

TestClient 2026-08-28:

| Stage | finding | confidence | bbox |
| ----- | ------- | ---------: | ---- |
| Direct wrapper `conf=0.10` (first box) | wound | 0.2982 | `[0.0, 260.14, 81.84, 300.0]` |
| GET case `visible_injury` | Wound | 0.2982 | `[0.0, 260.14, 81.84, 300.0]` |

`low_confidence` is true because wrapper `conf_threshold=0.40`. Frontend draws if `bounding_box` is set and `finding_detected !== false`. Live browser overlay geometry was **not** clicked in this session.

Plots/CSV from Ultralytics live under `C:\Users\santh\runs\detect\ml\models\yolo_candidate_training\run_merged\` (outside the repo copy).

---

## 4. Training report

### YOLO11

- **training status:** real Ultralytics `model.train` (8 epochs, CPU, batch 4, imgsz 640), then val comparison  
- **dataset size:** 83 train / 17 val / 23 test unique images (`yolo_merged`)  
- **dataset type:** mixed public bounding boxes (real_wound + remapped yolo_injury)  
- **metrics:** see table above  
- **retraining required:** more cut/bruise images before claiming those classes work. Cut recall 0 on val (n=2)

### EfficientNetV2

- **training status:** real PyTorch loop (`loss.backward()`, `opt.step()`, ImageNet backbone frozen, classifier head trained, 8 epochs)  
- **dataset:** `data/datasets/public_wound_dataset` 140/30/30, classes cut/bruise/swelling  
- **metrics (held-out test n=30):** accuracy 1.0, confusion `[[5,0,0],[0,5,0],[0,0,20]]`  
- **blank probes:** gray swelling 0.962; black/white swelling 1.0 → `blank_image_collapsed=true` → status **`NOT_TRUSTWORTHY`**  
- **retraining required:** yes, before any trust. Perfect test accuracy on 30 images with swelling majority is **not** generalization proof  
- Uniform images are **withheld** at inference (`std < 3`)

### U-Net

- **training status:** real loop, ResNet34 encoder frozen, BCE+Dice, 6 epochs  
- **dataset:** same public image/mask pairs 140/30/30  
- **test:** mean Dice 0.982, IoU 0.965, precision 0.987, recall 0.978, FP area 0.0009  
- **blank probes:** gray positive_ratio 0.16; **black 0.996; white 0.999** → **`MODEL_OUTPUT_NOT_TRUSTWORTHY`**  
- Wrapper returns empty mask on uniform ROI (`std < 3`)  
- **retraining required:** yes for blank-image safety without a gate

### XGBoost

- **training status:** already trained on **synthetic** 23-feature fusion (not genuine paired clinical records)  
- **metrics:** 25/30 accuracy 0.833 (from registry/metadata)  
- **schema:** exactly 23 features; wrong dim raises `ValueError`; analyze no longer auto-trains `np.random.randn`  
- **retraining required:** only if real multimodal labels appear

### VQC

- **training status:** `VQCClassifier.train()` — AngleEmbedding + StronglyEntanglingLayers — **same circuit as inference**. 1 epoch, 24 train samples  
- **metrics:** 11/30 = 0.367, MCC 0.0  
- **fake fallback `[0.15, 0.70, 0.15]` removed**; failures raise / API returns `class: null`, `status: MODEL_UNAVAILABLE`  
- **label:** EXPERIMENTAL simulator. **Not production-ready**  
- **retraining required:** more epochs would still be a classical PennyLane toy

### Sensor classifier

- `SensorClassifier.predict_from_summary()` **is called** from `process_sensor_data`  
- Missing features → `feature_missing`, no invented jerk/lux  
- Persists `predicted_motion_class`, `motion_confidence`, `motion_probabilities`, `classifier_status`

---

## 5. Twilio report

| Item | Result |
| ---- | ------ |
| Configured | **No.** `TWILIO_ENABLED=false` |
| Canonical env | `TWILIO_ENABLED`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `TWILIO_TO_NUMBER` in `backend/.env.example` and `backend/env.example` |
| Legacy aliases | `TWILIO_PHONE_NUMBER` → from; `EMERGENCY_CONTACT_PHONE` / `EMERGENCY_PHONE_NUMBER` → to |
| `/api/sos/config` | `configured: false`, message `Twilio integration disabled (TWILIO_ENABLED=false).` |
| Actual sending path | `TwilioService.send_test_sos_message` → Twilio REST `Messages.json` only if configured. Success status is Twilio’s `queued` (or similar), **not** a fabricated `SMS_SENT` |
| Disabled UX | Frontend twilio_test mode: **SMS NOT CONFIGURED** / **LOCAL SOS SIMULATION ONLY** |
| Local SOS | TestClient trigger 200, `sos_status: countdown`, message **Simulation only** |
| Real Twilio message | **Not sent.** Do not claim SMS delivery |
| Unit tests | Mocks exist; they are **not** a live Twilio receipt |

Mismatch that was real: docs/env used `TWILIO_PHONE_NUMBER` while code read `TWILIO_FROM_NUMBER`. Code now accepts both; examples use the canonical names.

---

## 6. All bugs

| Bug ID | Severity | File | Root Cause | Fix | Verification |
| ------ | -------- | ---- | ---------- | --- | ------------ |
| B01 | High | `ml/training/train_yolo.py` (old) | Official trainer copied weights + fake metrics; runtime YOLO too weak at conf 0.10 | Real Ultralytics train on merged set; promote if mAP50-95 ≥ baseline | Baseline 0.268 vs candidate 0.679; sweep val images now detect at 0.10 |
| B02 | High | `ml/vision/yolo_wrapper.py` | Hardcoded abrasion/laceration not in `model.names` | Classes from `model.names` | `/api/models` classes == live `model.names` |
| B03 | High | `ml/training/train_efficientnet.py` (old) | Untrained `state_dict` + hardcoded accuracy | Real training + blank probes; status `NOT_TRUSTWORTHY` if collapse | Metadata `training_was_real: true`, gray 0.96 swelling |
| B04 | High | `ml/training/train_unet.py` (old) | Same fake-train pattern | Real Dice/IoU; blank probes; wrapper empty mask on uniform ROI | black/white positive_ratio ~1.0 recorded; gray ROI withheld |
| B05 | Critical | `ml/classifiers/vqc_classifier.py` | `except Exception: return 1, [0.15, 0.70, 0.15]` | Raise; API structured failure | Source test; analyze returned real scores |
| B06 | High | `backend/main.py` analyze | Auto-train XGB/VQC with `np.random.randn` if missing | `require_model_artifacts()` → HTTP 503 `MODEL_ARTIFACT_MISSING` | `test_analyze_source_has_no_random_autotrain` |
| B07 | Medium | `ml/classifiers/sensor_classifier.py` | Invented missing kinetic features / dummy 0.33 probs | Explicit `feature_missing` / `MODEL_UNAVAILABLE` | `test_sensor_classifier_missing_features_are_explicit` |
| B08 | Medium | `backend/.env.example` | `TWILIO_PHONE_NUMBER` vs `TWILIO_FROM_NUMBER` | Canonical names + aliases | `/api/sos/config` |
| B09 | Medium | Frontend SOS | Implied Twilio might send when unconfigured | **SMS NOT CONFIGURED** / **LOCAL SOS SIMULATION ONLY** | Code + config API |
| B10 | Medium | Frontend VQC | Fallback text `0.33, 0.33, 0.33` | Show scores or error; no fake probs | `frontend/app/cases/[id]/page.tsx` |
| B11 | Medium | `backend/main.py` fusion | `peak_g_force or 1.0` (0.0 became 1.0) | Use 0.0 when missing; do not substitute g from m/s² | Source inspection |
| B12 | Low | `ml/explainability/evidence_consistency.py` | Counterfactuals crashed if VQC missing | Skip VQC when untrained | Source |
| B13 | Medium | Registry | Path/class/metric drift vs disk | `generate_registry.py` hashes disk + metadata | YOLO SHA `6cc84115…` matches file |
| B14 | Low | Tests | Required abrasion/laceration; 4 EfficientNet classes | Tests updated to match `model.names` / 3-class head | pytest 111 passed |
| B15 | Remaining | EfficientNet/U-Net | Real train still OOD-collapses | Honest status + inference gates | Blank probes |
| B16 | Remaining | YOLO | conf 0.10 over-detects; cut recall 0 | Documented; not “fixed” by lowering tests | Sweep + per-class val |

---

## 7. Remaining limitations

- **Not a medical device.** No clinical validation.  
- YOLO val n=17; cut/bruise counts are tokens, not a dataset. conf 0.10 is noisy.  
- EfficientNet 100% test accuracy **and** 96–100% swelling on blank images: do not trust classifier on real photos without a separate OOD study.  
- U-Net high Dice on in-domain masks **and** ~100% positive on black/white: overlays are gated on uniform images only, **not** on all failure modes.  
- Grad-CAM is attention on a **not-trustworthy** classifier.  
- XGBoost/VQC trained on **synthetic** fusion vectors. VQC is a 4-qubit **CPU simulator**.  
- Twilio: local simulation only until env is set; even then a 200 from Twilio is **queued**, not “SMS delivered to a bystander.”  
- Live **browser** overlay/SOS click-through was not run in this session. Frontend changes are source-level.  
- Duplicate YOLO `.pt` files still on disk (byte-identical candidate copies).  
- `check_and_create_vision_weights` / silent `except Exception: pass` may still exist in older scripts; analyze path was cleaned for auto-train and SOS stale cleanup.  
- Sensor simulate rejects `fall` / `normal`; UI uses the long names (OK).  

---

## 8. Exact run instructions (Windows)

MongoDB must already be running (`mongodb://localhost:27017/`, database `ai_qtriage`).

```bat
cd /d "c:\Users\santh\Capstone Project Code"

backend\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

cd frontend
npm install
npm run dev
```

Tests:

```bat
cd /d "c:\Users\santh\Capstone Project Code"
backend\venv\Scripts\python.exe -m pytest -q
```

Last executed result: **111 passed**, 5 warnings, 103.19s.

YOLO training + comparison (does not copy weights unless candidate mAP50-95 ≥ baseline):

```bat
backend\venv\Scripts\python.exe ml\training\train_yolo.py
```

EfficientNet / U-Net / VQC:

```bat
backend\venv\Scripts\python.exe ml\training\train_efficientnet.py
backend\venv\Scripts\python.exe ml\training\train_unet.py
backend\venv\Scripts\python.exe ml\training\train_vqc.py
```

Evaluation / registry / YOLO sweep:

```bat
backend\venv\Scripts\python.exe scratch\yolo_threshold_sweep.py
backend\venv\Scripts\python.exe ml\models\generate_registry.py
```

Evidence dumps: `scratch/yolo_threshold_sweep.json`, `scratch/yolo_baseline_vs_candidate.json`, `scratch/e2e_testclient_verify.json`, `ml/models/vision/*_metadata.json`, `ml/models/model_registry.json`.

---

## 9. Final model readiness

| Model | Readiness |
| ----- | --------- |
| YOLO11 | **READY_FOR_RESEARCH_DEMO** with caveats (noisy at 0.10; cut not actually learned) |
| EfficientNetV2 | **NOT_READY** / **NEEDS_RETRAINING** — trained but `NOT_TRUSTWORTHY` on blanks |
| U-Net | **NOT_READY** / **NEEDS_RETRAINING** — `MODEL_OUTPUT_NOT_TRUSTWORTHY` on blanks |
| Grad-CAM | **NOT_READY** as explainability |
| XGBoost | **READY_FOR_RESEARCH_DEMO** (synthetic features only) |
| VQC | **EXPERIMENTAL_ONLY** |
| Sensor classifier | **READY_FOR_RESEARCH_DEMO** when required features exist |
| Twilio SMS | **NOT_READY** (disabled / not configured) |

---

## What was proven FIXED (execution)

1. YOLO runtime path is canonical; classes = `model.names`; candidate beat baseline mAP50-95 (0.679 vs 0.268) and was selected; val images detect at conf 0.10.  
2. EfficientNet and U-Net official trainers perform real optimization; metrics written from predictions/probes; blank collapse recorded, not hidden.  
3. VQC train/infer circuits match; silent fake softmax removed.  
4. Analyze does not auto-train random models.  
5. Sensor classifier is wired; missing features are explicit.  
6. Twilio env names unified; unconfigured mode does not claim SMS sent.  
7. Registry SHA for YOLO matches disk `6cc84115…`.  
8. pytest **111 passed** after contract-correct test updates (not by stubbing models).  
9. Demo analyze: stored YOLO box equals wrapper box (`wound`, 0.2982, `[0.0, 260.14, 81.84, 300.0]`).  

**Not proven:** clinical accuracy, Twilio delivery, frontend pixel-perfect overlay in a real browser, or that EfficientNet/U-Net are safe without the uniform-image gate.
