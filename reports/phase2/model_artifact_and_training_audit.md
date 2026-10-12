# Phase 2 Model Artifact and Training Audit Report

**Generated:** 2026-10-12  
**Auditor:** Senior Research Reproducibility Auditor & ML Verification Team  
**Scope:** Architectural inspection, weight state verification, preprocessing boundary audit, and promotion status for all 6 active models.

---

## 1. Summary of Model Artifacts & Research Status

| Model Identifier | Architecture | Canonical Weights File | SHA-256 Checksum | Preprocessing / Fit State | Model Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11 Detector** | YOLO11n Object Detection | `ml/models/vision/yolo11_injury_best.pt` | `857880192ebfbc4b6934a033cf7df78310885291ef00a5dd691576c6fbc4bff9` | Normalized RGB, 640x640 | `TRAINED` |
| **EfficientNetV2** | EfficientNetV2-S (8-class) | `ml/models/vision/efficientnetv2_injury_best.pt` | `8cb4263e70b1e020ecc6324dca9b760f38074fd7778a6b814bcb10018d58a57a` | ImageNet Mean/Std, 224x224 | `TRAINED` |
| **ResNet34-U-Net** | SMP ResNet34-U-Net | `ml/models/vision/unet_injury_best.pt` | `3c7f3f39196d71b9d8d58d1fcfc7438b4ae23d75fbd8195ae7da7b1fcb9660d1` | ImageNet Mean/Std, 256x256 | `TRAINED` |
| **Sensor Classifier** | XGBoost (7 kinematic feats) | `ml/models/sensor_motion_best.json` | `4980d2150f379dadf1fef4d8ed61eaf298e67cff8ec495f053d80e9b13e6569d` | `StandardScaler` (Train only) | `TRAINED` |
| **Multimodal XGBoost**| XGBoost (23 tabular feats) | `ml/models/xgboost_best.json` | `b57c5cd4f844202aad62feecc8e30e03e46303f2e3e98a3337690107244f9e25` | Raw 23-d clinical features | `EXPERIMENTAL_ONLY` |
| **4-Qubit VQC** | PennyLane StronglyEntangling | `ml/models/vqc/vqc_weights.npz` | `c818299beb1ff7e44bc1fa729a14befc6fd0ab63474205948b76897960e79412` | `StandardScaler` + `PCA(4)` | `EXPERIMENTAL_ONLY` |

---

## 2. In-Depth Architectural & Checkpoint Analysis

### 2.1 YOLO11 Object Detector
* **File:** `ml/models/vision/yolo11_injury_best.pt`
* **Training Script:** `ml/training/train_yolo.py`
* **Dataset:** `data/datasets/yolo_expanded_v1` (2,813 images: 1,935 train, 442 val, 436 test).
* **Classes:** 6 classes (`wound`, `bruise`, `burn`, `abrasion`, `cut`, `laceration`).
* **Checkpoint Verification:** Evaluated on PyTorch/Ultralytics runtime. Clean weights dictionary containing full convolutional backbone and detection head.
* **Findings:** The model demonstrates robust performance on general wounds, burns, and cuts, but suffers from class imbalance in lacerations. 

### 2.2 EfficientNetV2 Classifier
* **File:** `ml/models/vision/efficientnetv2_injury_best.pt`
* **Training Script:** `ml/training/train_efficientnet_kaggle.py`
* **Dataset:** `data/datasets/efficientnet_kaggle_v1` (3,039 photos: 2,104 train, 462 val, 473 test).
* **Classes:** 8 classes (`wound`, `bruise`, `burn`, `abrasion`, `cut`, `laceration`, `normal`, `ood_reject`).
* **Checkpoint Verification:** TorchScript/State-Dict format. Verified valid forward passes across all 8 output logits with softmax temperature calibration.

### 2.3 ResNet34-U-Net Segmentation Model
* **File:** `ml/models/vision/unet_injury_best.pt`
* **Training Script:** `ml/training/train_unet.py`
* **Dataset:** `data/datasets/unet_deduped_subject` (464 image/mask pairs, 463 unique subjects: 326 train, 69 val, 69 test).
* **Architecture:** Segmentation Models PyTorch (SMP) ResNet34 encoder with standard U-Net decoder pathways.
* **Checkpoint Verification:** Verified valid sigmoid mask output probabilities `[0.0, 1.0]`.

### 2.4 Sensor Motion Event Classifier
* **File:** `ml/models/sensor_motion_best.json` + `ml/models/sensor_scaler.pkl`
* **Training Script:** `ml/training/train_sensor_model.py`
* **Dataset:** `data/datasets/manifests/sensor_real_manifest.csv` (SisFall + UCI HAR, 6,505 windows, 68 subjects: 4,247 train, 1,030 val, 1,228 test).
* **Architecture Clarification:** The production sensor classifier is an **XGBoost gradient boosted tree booster** operating on 7 kinematic summary features (`peak_g_force`, `peak_accel_ms2`, `peak_jerk_gs`, `accel_variance`, `gyro_variance`, `sma`, `stabilization_seconds`). Claims of an end-to-end "1D-CNN-LSTM" deep neural network reflect historical research exploration, not the active canonical model.
* **Fit State Audit:** The `StandardScaler` (`sensor_scaler.pkl`) was fitted exclusively on the 4,247 training windows.

### 2.5 Multimodal XGBoost & 4-Qubit VQC
* **Files:** `ml/models/xgboost_best.json` and `ml/models/vqc/vqc_weights.npz`
* **Training Scripts:** `ml/training/train_xgboost.py` and `ml/training/train_vqc.py`
* **Dataset:** `data/multimodal_tabular_test.csv` (200 synthetic vectors).
* **Data Leakage & Preprocessing Boundaries:**
  * Both models were trained strictly using `get_leakage_free_splits(200)` (140 train, 30 val, 30 test).
  * VQC preprocessing (`StandardScaler` and `PCA(n_components=4)`) was strictly fitted on the 140 training samples and applied out-of-sample to the 30 test samples.
* **Status Classification:** Designated as **`EXPERIMENTAL_ONLY`** due to reliance on synthetic multimodal fusion vectors.

---

## 3. Conclusions & Promotion Status

All 4 production perception models (`yolo11_injury_best.pt`, `efficientnetv2_injury_best.pt`, `unet_injury_best.pt`, `sensor_motion_best.json`) are validated as **`TRAINED`** on real physical datasets. The multimodal fusion models (`xgboost_best.json`, `vqc_weights.npz`) are strictly designated as **`EXPERIMENTAL_ONLY`** research prototypes.
