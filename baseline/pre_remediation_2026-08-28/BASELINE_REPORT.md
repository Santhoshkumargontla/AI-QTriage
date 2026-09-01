# AI-QTriage pre-remediation baseline

**Recorded:** 2026-08-28 (UTC timestamps below)  
**Workspace:** `C:\Users\santh\Capstone Project Code`  
**Branch:** `fix/application-remediation`  
**Application logic:** not modified  
**ML weights:** not modified, not deleted  

Evidence folder: `baseline/pre_remediation_2026-08-28/`

---

## 1. Git status

### Home git (`C:\Users\santh`) — already existed

| Item | Value |
| ---- | ----- |
| Root | `C:/Users/santh` |
| Branch | `master` |
| Commits | **none** (`fatal: your current branch 'master' does not have any commits yet`) |
| Capstone tracking | **untracked** (`?? "Capstone Project Code/"`) |

A branch was **not** created on this home repo. Branching `C:\Users\santh` would have affected the entire user profile tree.

### Project git (created this session)

`git init` was run **inside** the Capstone workspace so remediation has a local branch without touching the home repo.

| Item | Value |
| ---- | ----- |
| Root | `C:/Users/santh/Capstone Project Code` |
| Branch | `fix/application-remediation` |
| HEAD | **no commits** (`fatal: ambiguous argument 'HEAD'`) |
| Working tree | all files untracked (`??`) |
| Extra noise | `warning: could not open directory '.tmp_pytest/': Permission denied` |

**Failure / gap:** the branch exists but has **no snapshot commit**. File restore is currently the on-disk tree plus the metadata copies in this folder. A first commit was not created because it was not requested.

---

## 2. Canonical YOLO

| Item | Value |
| ---- | ----- |
| Constant | `ml/models/vision/yolo11_injury_best.pt` (`YOLO_CANONICAL`) |
| Wrapper loaded path | `ml/models/vision/yolo11_injury_best.pt` |
| Live `/api/models` path | `ml\models\vision\yolo11_injury_best.pt` |
| `YOLO_MODEL_VERSION` | unset |
| Task | `detect` |
| `model.names` | `{0: cut, 1: bruise, 2: wound}` |
| Status | `INFERENCE_EXECUTES` |
| SHA-256 | `6cc84115e4cb85c8b82715211c3935200b815b76efbc95f83855c2cc988dce4f` |
| Size | 5,468,506 bytes |

---

## 3. Runtime model hashes

Hashed from files the default loaders use. Weights were read-only.

| Model | Path | Bytes | SHA-256 |
| ----- | ---- | ----: | ------- |
| YOLO11 | `ml/models/vision/yolo11_injury_best.pt` | 5468506 | `6cc84115e4cb85c8b82715211c3935200b815b76efbc95f83855c2cc988dce4f` |
| EfficientNetV2 | `ml/models/vision/efficientnetv2_injury_best.pt` | 81621023 | `6944605ae71685d909d505d12ce32d5bd9e953c10bbebb72a02635a729897e83` |
| U-Net | `ml/models/vision/unet_injury_best.pt` | 97918031 | `2b4967aa04f6af309d3aa14fef2e098350bdd02fd0e4575e11baa033eb0424a2` |
| XGBoost | `ml/models/xgboost_best.json` | 199969 | `73bb5a5125c3e3907bffa1059165d90ecce9dd4e47ce9ce8f9c1f8937fd3f643` |
| VQC weights | `ml/models/vqc/vqc_weights.npz` | 1086 | `7d8febd0256dfcad793cbf1a1a2303d746c39e03ed4513536eb574dea73e027d` |
| VQC scaler | `ml/models/vqc/scaler.pkl` | 960 | `94f1e85b1e45f6c2443312dfd7d98becb9fa4de13bd87d2c229316eb8c1fb4f8` |
| VQC PCA | `ml/models/vqc/pca.pkl` | 1198 | `a12c32c26238a4188d621454cc77b6ae2e11ecff2528dd2f50439db4751acac8` |
| Sensor model | `ml/models/sensor_motion_best.json` | 135603 | `707ee9c25392d5e8e1ca9e023feb77937b6ad5a159cb0b53d3b6a09f047e1cbb` |
| Sensor scaler | `ml/models/sensor_scaler.pkl` | 775 | `04291f244c187b14d42947328235cd7f23fd9818aebe237aa6131b4e9ad9d959` |

JSON dump: `runtime_model_hashes.json`

---

## 4. Metadata / registry backup

Copied (not moved) into `baseline/pre_remediation_2026-08-28/metadata/`:

- `ml/models/model_registry.json`
- `ml/models/vision/yolo11_metadata.json`
- `ml/models/vision/efficientnetv2_metadata.json`
- `ml/models/vision/unet_metadata.json`
- `ml/models/vision/yolo_baseline_vs_candidate.json`
- `ml/models/vqc/vqc_metadata.json`
- `ml/models/xgboost_metadata.json`
- `ml/models/sensor_metadata.json`
- `ml/models/classical/xgboost_metadata.json`
- `ml/models/yolo_training/training_metadata.json`
- `ml/models/artifact_integrity_report.json`
- `ml/models/canonical_paths.py` (path constants only)

No `.pt` / `.pth` files were copied or deleted.

---

## 5. Test results (executed)

Command:

```bat
backend\venv\Scripts\python.exe -m pytest -q --tb=line --junitxml=baseline\pre_remediation_2026-08-28\pytest-junit.xml
```

| Metric | Count |
| ------ | ----: |
| Passed | **111** |
| Failed | **0** |
| Errors | **0** |
| Skipped | **0** |
| Warnings | **5** |
| Duration | 212.53 s (JUnit time 212.512) |
| Exit code | 0 |

Warnings (not failures):

1. SHAP `PendingDeprecationWarning` `set_bad`
2. SHAP `set_over`
3. SHAP `set_under`
4–5. PyTorch `FutureWarning`: Grad-CAM `register_backward_hook` is not a full backward hook (`test_one_click_complete_demo`, `test_grad_cam_hook_and_overlay`)

JUnit: `pytest-junit.xml`

---

## 6. Backend health (executed)

Live process: `uvicorn backend.main:app --host 127.0.0.1 --port 8000` (already running).

`GET http://127.0.0.1:8000/api/health` → **200**

```json
{"status":"healthy","database":"connected","timestamp":"2026-08-28T14:26:11.453477"}
```

`GET http://127.0.0.1:8000/api/models` → **200**. Live statuses (not hidden):

| Model | Live status | weights_loaded |
| ----- | ----------- | -------------- |
| YOLO11 | INFERENCE_EXECUTES | true |
| EfficientNetV2 | **NOT_TRUSTWORTHY** | true |
| U-Net | **MODEL_OUTPUT_NOT_TRUSTWORTHY** | true |
| XGBoost | TRAINED_AND_EVALUATED | true |
| VQC | EXPERIMENTAL | true |

Uvicorn log also shows `GET /` → **404** (no root route). That is expected for API-only FastAPI, not a health failure.

---

## 7. Frontend build (executed)

```bat
cd frontend
npm run build
```

| Item | Result |
| ---- | ------ |
| Exit code | **0** |
| Next.js | 16.3.0 (Turbopack) |
| Compile | success (10.2s) |
| TypeScript | finished (4.4s) |
| Routes | `/`, `/_not-found`, `/cases/[id]` (dynamic), `/create-case`, `/research` |

**Warning (not a failed build):**

```text
Next.js ignored package-lock.json in C:\Users\santh because it is outside
the current Git repository (C:\Users\santh\Capstone Project Code).
To use this directory, set `turbopack.root` in your Next.js config.
```

`npm warn Unknown env config "devdir"` also printed.

---

## 8. Known failures and gaps (not hidden)

These did **not** fail pytest or health. They are still true of this baseline:

1. Project git has **no commits**; rollback is not `git reset --hard`.
2. Home git at `C:\Users\santh` is an empty `master`; Capstone is untracked there.
3. `git status` cannot read `.tmp_pytest/` (permission denied).
4. EfficientNet and U-Net advertise **untrustworthy** on the live `/api/models` endpoint.
5. Frontend build warns that `package-lock.json` under `C:\Users\santh` is outside the new nested git root.
6. No `/cases` list page; sidebar hash links are dead (from prior inspection).
7. U-Net mask PNG is ROI-sized vs original canvas (from prior inspection).
8. Stale `data/results/evaluation_results.json` (25 Aug, n=18) vs registry XGB 25/30 (28 Aug).
9. Twilio not configured (`TWILIO_ENABLED` unset).
10. Browser overlay click-through was **not** re-run in this baseline session.

---

## 9. What this session did not do

- Did not change application logic
- Did not modify or delete ML weight files
- Did not commit
- Did not push
- Did not create a branch on `C:\Users\santh`
