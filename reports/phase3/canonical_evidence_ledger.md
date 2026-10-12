# Canonical Experimental Evidence Ledger

**Generated:** October 13, 2026  
**Auditor:** Senior Research Reproducibility Auditor & Scientific Writer  
**Target Submission:** ICICNIS 2026 (Paper ID: ICICNIS-945)  
**Software Revision:** `fe98236595ecc2a73cde6ae4dadeaed78db8648f`  

---

## 1. Perception & Perceptual Classification Baselines

### 1.1 YOLO11 Object Detector
* **Task:** Multi-class bounding box detection (`wound`, `bruise`, `burn`, `abrasion`, `cut`, `laceration`).
* **Canonical Checkpoint:** `ml/models/vision/yolo11_injury_best.pt`
* **SHA-256 Digest:** `857880192ebfbc4b6934a033cf7df78310885291ef00a5dd691576c6fbc4bff9`
* **Dataset & Provenance:** `data/datasets/yolo_expanded_v1` (2,813 public real photographic trauma images; 1,935 train, 442 val, 436 test).
* **Canonical Evaluated Test Set:** 436 held-out test images.
* **Verified Test Metrics:**
  * **mAP@50:** **34.54%**
  * **mAP@50-95:** **16.74%**
  * **Precision:** 51.59%
  * **Recall:** 39.42%
  * **Per-Class AP50:** `abrasion`: 61.03%, `bruise`: 60.08%, `cut`: 50.67%, `wound`: 19.36%, `burn`: 16.07%, `laceration`: 0.00%
* **Evidence Classification:** **Independently Reproduced** (`data/results/phase2_baseline_evaluation.json`).
* **Manuscript Permitted Wording:** Report 34.54% mAP@50 on the 436-image test set (and 36.47% on the localized 123-image baseline). Frame YOLO11 as an initial localization and ROI-cropping stage; explicitly disclose that lacerations achieve 0% AP@50 due to class sparsity (122 instances vs. 1,999 wounds).

### 1.2 EfficientNetV2-S Photo Classifier
* **Task:** 8-class injury classification (`wound`, `bruise`, `burn`, `abrasion`, `cut`, `laceration`, `normal`, `ood_reject`).
* **Canonical Checkpoint:** `ml/models/vision/efficientnetv2_injury_best.pt`
* **SHA-256 Digest:** `8cb4263e70b1e020ecc6324dca9b760f38074fd7778a6b814bcb10018d58a57a`
* **Dataset & Provenance:** `data/datasets/efficientnet_kaggle_v1` (3,039 photos; 2,104 train, 462 val, 473 test).
* **Canonical Evaluated Test Set:** 473 held-out test photos.
* **Verified Test Metrics:**
  * **Accuracy:** **96.62%** (457 / 473 correct)
  * **Macro F1:** **94.34%**
  * **Macro Precision:** 94.26%
  * **Macro Recall:** 95.01%
  * **OOD Rejection Accuracy:** 100% (6 / 6)
* **Evidence Classification:** **Independently Reproduced** (`data/results/phase2_baseline_evaluation.json`).

### 1.3 ResNet34-U-Net Wound Segmenter
* **Task:** Binary pixel wound segmentation.
* **Canonical Checkpoint:** `ml/models/vision/unet_injury_best.pt`
* **SHA-256 Digest:** `3c7f3f39196d71b9d8d58d1fcfc7438b4ae23d75fbd8195ae7da7b1fcb9660d1`
* **Dataset & Provenance:** `data/datasets/unet_deduped_subject` (464 pairs, 463 unique subjects: 326 train, 69 val, 69 test).
* **Canonical Evaluated Test Set:** 69 held-out patient pairs.
* **Verified Test Metrics (Fixed Threshold $\tau = 0.50$):**
  * **Mean Dice:** **0.6418**
  * **Mean IoU:** **0.5178**
  * **Precision:** 0.6930
  * **Recall:** 0.7233
  * **Mean False Positive Area:** 2.93%
* **Evidence Classification:** **Independently Reproduced** (`data/results/phase2_baseline_evaluation.json`).
* **Manuscript Permitted Wording:** Report 64.18% Dice / 51.78% IoU as canonical test metrics on 69 pairs. Contextualize with literature benchmarks (62–72% typical on AZH/Medetec wound photography). State that U-Net estimates relative wound area ratio ($F_5$) to inform triage heuristics.

---

## 2. Sensor Kinematic Event Classification

### Sensor XGBoost Classifier
* **Task:** 3-class kinematic event classification (`normal_activity`, `fall`, `impact`).
* **Canonical Checkpoint:** `ml/models/sensor_motion_best.json` + `ml/models/sensor_scaler.pkl`
* **Architecture:** **XGBoost gradient-boosted tree** operating on 7 kinematic summary features (`peak_g_force`, `peak_accel_ms2`, `peak_jerk_gs`, `accel_variance`, `gyro_variance`, `sma`, `stabilization_seconds`).
* **Dataset & Provenance:** `data/datasets/manifests/sensor_real_manifest.csv` (SisFall + UCI HAR; 6,505 windows, 68 subjects: 4,247 train, 1,030 val, 1,228 test).
* **Canonical Evaluated Test Set:** 1,228 held-out test windows.
* **Verified Test Metrics:**
  * **Overall Accuracy:** **96.34%** (1,183 / 1,228 correct)
  * **Macro F1:** **89.85%**
  * **MCC:** 0.9296
  * **Per-Class Metrics:**
    * `normal_activity` ($n=720$): Precision 98.89%, Recall 98.75%, F1 98.82%
    * `fall` ($n=448$): Precision 94.89%, Recall 95.31%, F1 95.10%
    * `impact` ($n=60$): Precision 76.27%, Recall 75.00%, F1 75.63%
* **Evidence Classification:** **Independently Reproduced** (Live script execution).

---

## 3. Multimodal Fusion & Quantum Baselines (Synthetic Benchmark)

### 3.1 Multimodal XGBoost
* **Task:** 3-class triage classification (`LOW`, `MODERATE`, `HIGH`) on 23-feature vectors.
* **Canonical Checkpoint:** `ml/models/xgboost_best.json`
* **Dataset & Provenance:** `data/multimodal_tabular_test.csv` (200 synthetic vectors; 140 train, 30 val, 30 test).
* **Canonical Evaluated Test Set:** 30 held-out test vectors.
* **Verified Test Metrics:**
  * **Accuracy:** **83.33%** (25 / 30 correct)
  * **95% Wilson Score CI:** $[66.44\%, 92.66\%]$
  * **Macro F1:** **78.50%**
  * **MCC:** 0.7027
  * **Confusion Matrix:** `LOW`: 7/10, `MODERATE`: 16/17, `HIGH`: 2/3
* **Evidence Classification:** **Independently Reproduced** (Live script execution).

### 3.2 4-Qubit Variational Quantum Classifier (VQC)
* **Task:** 3-class triage classification on PCA-compressed 4-qubit circuit.
* **Canonical Checkpoint:** `ml/models/vqc/vqc_weights.npz`
* **Simulator:** PennyLane `default.qubit` classical CPU simulator.
* **Verified Test Metrics:**
  * **Grouped Leak-Free Split (`test_idx`):** **56.67% Accuracy** (17 / 30 correct), Macro F1: 42.14%, MCC: 0.1275.
  * **Sequential Slice ($170:200$):** **53.33% Accuracy** (16 / 30 correct), Macro F1: 36.06%, MCC: 0.0173.
* **Inductive Bias Finding:** Predicts MODERATE in 25 of 30 test cases.
* **Evidence Classification:** **Independently Reproduced** (Live script execution).
* **Manuscript Permitted Wording:** Frame VQC as an exploratory research prototype demonstrating hybrid classical-quantum feature integration. Disclose that it achieves 56.67% accuracy without quantum advantage. Confirm it is decoupled from clinical decision logic (`used_in_main_decision: false`).

---

## 4. Modality Ablation Matrix (30 Synthetic Vectors)

| Configuration | Features | Test Accuracy | Macro F1 | MCC | Wilson 95% CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vision Only** | 6 | **40.00%** (12/30) | 0.2886 | -0.0610 | [24.59%, 57.68%] |
| **Questionnaire Only** | 12 | **56.67%** (17/30) | 0.5036 | 0.3340 | [39.20%, 72.62%] |
| **Sensor Only** | 5 | **56.67%** (17/30) | 0.5010 | 0.3325 | [39.20%, 72.62%] |
| **Vision + Questionnaire** | 18 | **53.33%** (16/30) | 0.4732 | 0.2647 | [36.14%, 69.77%] |
| **Vision + Sensor** | 11 | **63.33%** (19/30) | 0.5321 | 0.4079 | [45.51%, 78.13%] |
| **Questionnaire + Sensor** | 17 | **90.00%** (27/30) | **0.8693** | **0.8251** | [74.38%, 96.54%] |
| **Full Multimodal** | 23 | **83.33%** (25/30) | 0.7850 | 0.7027 | [66.44%, 92.66%] |

* **Evidence Classification:** **Independently Reproduced** (`data/results/phase2_modality_ablation.json`).
