# AI-QTriage Phase 2 — Repository Inventory and Discovery Baseline

**Document:** `reports/phase2/repository_inventory.md`  
**Execution Timestamp:** 2026-10-12T17:55:00Z  
**Repository Root:** `C:\Users\santh\Capstone Project Code`  
**Git Branch:** `main`  
**Synchronized Git Commit:** `e8ecd2d59a6a1876fc5f1f241495034b1a29ba38`  
**Tested Code Revision:** `273dc2789dc5734b1226c60905242815ae8c9b32`  
**Working Tree Status:** Clean (0 untracked modifications in tracked repo files)  

---

## 1. Executive Discovery Summary

This document establishes the authoritative baseline inventory for Phase 2 research validation, model evaluation, and software quality assurance. All paths, checksums, runtime mappings, and artifact dependencies have been inspected from live disk storage and executable code.

---

## 2. Active Application Runtime Models & Code Tracing

The active application (FastAPI backend at `backend/main.py` and Next.js frontend at `frontend/app/`) loads six primary inference components and their associated preprocessing transformers. Below is the direct code trace from training script and canonical checkpoint to runtime wrapper and evaluation artifact:

| Component | Architecture & Classes | Canonical Checkpoint Path | Checkpoint SHA-256 | Training Script | Runtime Loader & Wrapper | Evaluation Artifact | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11 Detection** | YOLO11n (2.6M params); 6 classes: `cut`, `bruise`, `abrasion`, `burn`, `wound`, `laceration` | `ml/models/vision/yolo11_injury_best.pt` (5,433,882 B) | `857880192ebfbc4b6934a033cf7df78310885291ef00a5dd691576c6fbc4bff9` | `ml/training/train_yolo_retrain_v2.py` | `ml.vision.yolo_wrapper.YOLO11Detector` | `ml/models/yolo_threshold_eval/THRESHOLD_SWEEP_REPORT.json` | `READY_FOR_RESEARCH_DEMO` |
| **EfficientNetV2 Classification** | EfficientNetV2-S (`timm`); 8 classes: `abrasion`, `bruise`, `burn`, `cut`, `laceration`, `wound`, `normal`, `ood_reject` | `ml/models/vision/efficientnetv2_injury_best.pt` (81,650,563 B) | `8cb4263e70b1e020ecc6324dca9b760f38074fd7778a6b814bcb10018d58a57a` | `ml/training/train_efficientnet_kaggle_v1.py` | `ml.vision.efficientnet_wrapper.EfficientNetV2Classifier` | `ml/models/vision/efficientnetv2_metadata.json` | `READY_FOR_RESEARCH_DEMO` |
| **ResNet34-UNet Segmentation** | ResNet-34 Encoder + U-Net Decoder; binary mask (`wound_mask`) | `ml/models/vision/unet_injury_best.pt` (97,917,463 B) | `3c7f3f39196d71b9d8d58d1fcfc7438b4ae23d75fbd8195ae7da7b1fcb9660d1` | `ml/training/train_unet.py` | `ml.vision.unet_wrapper.UNetSegmenter` | `ml/models/vision/unet_metadata.json` | `READY_FOR_RESEARCH_DEMO` |
| **Sensor Motion Event Classifier** | XGBoost Booster on 7 kinematic summary features; 3 classes: `normal_activity`, `fall`, `impact` | `ml/models/sensor_motion_best.json` (582,892 B) | `4980d2150f379dadf1fef4d8ed61eaf298e67cff8ec495f053d80e9b13e6569d` | `ml/training/train_sensor_model.py` | `ml.classifiers.sensor_classifier.SensorClassifier` | `ml/models/sensor_metadata.json` | `TRAINED` |
| **Sensor Scaler** | `StandardScaler` (joblib) fitted on SisFall/UCI HAR train windows | `ml/models/sensor_scaler.pkl` (775 B) | `43ab14c958dd887cc1a23be87cb056e82b646ee42c7363bb438add5891c09aef` | `ml/training/train_sensor_model.py` | `SensorClassifier.scaler` | `ml/models/sensor_metadata.json` | `TRAINED` |
| **XGBoost Multimodal Classifier** | GBDT ensemble (100 trees, max depth 4); 3 classes: `LOW`, `MODERATE`, `HIGH` across 23 features | `ml/models/xgboost_best.json` (199,210 B) | `b57c5cd4f844202aad62feecc8e30e03e46303f2e3e98a3337690107244f9e25` | `ml/training/train_xgboost.py` | `ml.classifiers.xgboost_classifier.XGBoostClassifier` | `data/results/canonical_held_out_evaluation_corrected.json` | `TRAINED` |
| **Experimental 4-Qubit VQC** | Variational Quantum Circuit (`AngleEmbedding` + 2-layer `StronglyEntanglingLayers`); 3 classes | `ml/models/vqc/vqc_weights.npz` (1,086 B) | `c818299beb1ff7e44bc1fa729a14befc6fd0ab63474205948b76897960e79412` | `ml/training/train_vqc.py` | `ml.classifiers.vqc_classifier.VQCClassifier` | `data/results/canonical_held_out_evaluation_corrected.json` | `EXPERIMENTAL_ONLY` |
| **VQC Preprocessing (Scaler & PCA)** | Classical `StandardScaler` + `PCA(n_components=4)` | `ml/models/vqc/scaler.pkl` (960 B)<br>`ml/models/vqc/pca.pkl` (1,198 B) | Scaler: `1c4f2821...`<br>PCA: `64107285...` | `ml/training/train_vqc.py` | `VQCClassifier.scaler`<br>`VQCClassifier.pca` | `ml/models/vqc/vqc_metadata.json` | `TRAINED (On Train Only)` |

---

## 3. Dataset Inventory and Manifest Baseline

| Dataset Identifier | Local Directory | Manifest / Configuration File | Total Samples Verified | Verified Split Breakdown | Modality / Annotation Format |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO Expanded V1** | `data/datasets/yolo_expanded_v1` | `PREPARE_REPORT.json`, `data.yaml` | **2,813 images** | Train: 1,935 (68.8%)<br>Val: 442 (15.7%)<br>Test: 436 (15.5%) | Photographic images + normalized YOLO bounding box text files (`.txt`) |
| **EfficientNet Kaggle V1** | `data/datasets/efficientnet_kaggle_v1` | `manifest.csv` | **3,039 photos** | Train: 2,104 (69.2%)<br>Val: 462 (15.2%)<br>Test: 473 (15.6%) | Single-label clinical/trauma wound photos across 8 classes |
| **U-Net Deduped Subject** | `data/datasets/unet_deduped_subject` | `manifest.csv` | **464 pairs** (463 subjects) | Train: 326 (70.3%)<br>Val: 69 (14.9%)<br>Test: 69 (14.9%) | RGB wound photos + 8-bit binary PNG segmentation masks |
| **SisFall + UCI HAR Sensor** | `data/datasets/manifests/sensor_real_manifest.csv` | `sensor_real_manifest.csv` | **6,505 windows** (68 subjects) | Train: 4,247 (65.3%)<br>Val: 1,030 (15.8%)<br>Test: 1,228 (18.9%) | 150-sample (2.5s @ 50 Hz) 6-axis IMU kinematic windows |
| **Synthetic Multimodal Fusion** | `data/datasets/multimodal_synthetic_manifest.csv` | `multimodal_synthetic_manifest.csv` | **200 tabular vectors** | Train: 140 (70.0%)<br>Val: 30 (15.0%)<br>Test: 30 (15.0%) | 23-dimensional synthetic numerical feature vectors |

---

## 4. Evaluation Artifacts and Stale Metric Conflicts Identified

Live inspection of `data/results/` and `ml/models/canonical_paths.py` revealed an important discrepancy that must be reconciled during Phase 2 and Phase 5:

1. **Active Evaluated Artifact:**
   - `data/results/canonical_held_out_evaluation_corrected.json` contains the verified leak-free metrics (XGBoost 25/30, VQC 17/30, 0 cross-split duplicates).
   - `data/results/independent_ablation_results.json` contains the verified 7-model independent modality ablation results.
2. **Stale Artifacts Persisting on Disk:**
   - `data/results/evaluation_results.json` (timestamp: September 6, 2026) still holds obsolete metrics (VQC 16/30, empty `ablation_study: []`, older SHA-256 hashes).
   - `data/results/canonical_held_out_evaluation.json` (timestamp: September 6, 2026) holds obsolete pre-remediation metrics.
3. **Endpoint Vulnerability in `backend/main.py`:**
   - `ml/models/canonical_paths.py` defines `EVAL_HELD_OUT = "data/results/canonical_held_out_evaluation.json"`, which points to the uncorrected September file.
   - `/api/evaluation/ablation` loads from `EVAL_RESULTS` (`evaluation_results.json`), returning an empty array `[]` instead of the 7 independently trained ablation models.
   - **Remediation Plan (Phase 5):** Point canonical evaluation constants directly to `canonical_held_out_evaluation_corrected.json` and `independent_ablation_results.json`, ensuring the API and frontend consume the authoritative leak-free evaluation results.

---

## 5. Phase 2 Step-by-Step Change Plan

1. **Phase 2 (Dataset Provenance & Leakage Audit):** Compute exact subject overlap, pixel hash collisions, and window leakage across all 5 datasets; generate `dataset_provenance.md`, `dataset_inventory.csv`, and `leakage_audit.json`.
2. **Phase 3 (Model Artifact & Training Audit):** Verify learned weights vs untrained initializations, parameter shapes, and training script configurations; assign explicit statuses (`TRAINED`, `EXPERIMENTAL_ONLY`).
3. **Phase 4 (Reproducible Baseline Evaluation):** Independently re-evaluate all active checkpoints on held-out test splits; compute confusion matrices, Dice, mAP, and Brier scores directly from raw predictions.
4. **Phase 5 (Pipeline Fixes):** Update canonical evaluation constants in `canonical_paths.py` and `backend/main.py` to eliminate stale JSON ingestion; verify frontend displays authoritative metrics.
5. **Phase 6 (Modality Ablation):** Validate the 7 independent ablation models against held-out test data; report integer confusion matrices and confidence intervals.
6. **Phase 7 (Vision Investigation):** Document YOLO11 precision/recall trade-offs and U-Net boundary dynamics; assess whether retraining is justified or contraindicated.
7. **Phase 8 (Application Integration & Safety):** Verify FastAPI endpoints, questionnaire validation, sensor ingest, SOS countdown, and research disclaimers.
8. **Phase 9 (Regression Testing):** Execute full backend test suite (`pytest`) and frontend production build (`npm run build`).
9. **Phase 10 (Final Evidence Freeze):** Package complete audit deliverables, issue register, and handoff documentation for Phase 3.
