# AI-QTriage — Authoritative Model Evaluation Report (`MODEL_EVALUATION_REPORT.md`)

**Project Title:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Conference:** 7th International Conference on IoT based Control Networks and Intelligent Systems (ICICNIS 2026 / IEEE Proceedings)  
**Paper ID:** ICICNIS-945  
**Document Classification:** Authoritative Empirical Benchmark Ledger  
**Date:** October 12, 2026  

---

## 1. Executive Master Results Table

This table represents the authoritative, verified evaluation results across all six model components.

| Component | Task | Tested Dataset & Split | Sample Count ($N$) | Primary Metric | Secondary Metric | Secondary Value | Macro-F1 | Calibration Metric | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11 (Photographic)** | Object Detection | Real photographic wound dataset (`yolo_expanded_v1`) | 23 held-out test images (74 boxes) | **mAP@50: 36.47%** (34.54%) | mAP@50–95 | **18.44%** (16.74%) | — | Recall: 39.42% Prec: 51.59% | **Authoritative Real Baseline** |
| **YOLO11 (Drawing Pretrain)** | Object Detection | Synthetic drawing dataset (`yolo_retrain_v2`) | 19 held-out test sketches (18 boxes) | mAP@50: 83.58% | mAP@50–95 | 70.80% | — | Pretrain only (0 wound labels) | **Synthetic Artifact Only** |
| **Sensor Motion Event** | Event Classification | SisFall + UCI HAR (`sisfall_uci_har_real`) | 1,228 held-out 2.5s IMU windows | **Accuracy: 96.34%** (1,183 / 1,228) | MCC | **0.9296** | **89.85%** | FPR: 1.25% FNR: 1.57% | **Authoritative Real Benchmark** |
| **ResNet34-U-Net** | Lesion Segmentation | AZH + WSEG + Medetec (`unet_deduped_subject`) | 69 held-out photographic wound pairs | **Mean Dice: 64.18%** (0.641808) | Mean IoU | **51.78%** (0.517783) | — | Threshold: $\tau = 0.50$ Prec: 69.30%, Rec: 72.33% | **Authoritative Real Benchmark** |
| **EfficientNetV2-S** | Image Classification | Kaggle multi-class wound photos + controls | 473 held-out photographic images | **Accuracy: 96.62%** | Macro Precision | 94.26% | **94.34%** (Rep: 93.75%) | Macro Recall: 95.01% OOD Collapse: 0 | **Authoritative Real Benchmark** |
| **Multimodal XGBoost** | Severity Triage | Synthetic multimodal dataset (leak-free split) | 30 held-out synthetic vectors | **Accuracy: 83.33%** (25 / 30) | MCC | **0.7027** | **78.50%** | Brier: 0.2901 ECE: 0.1761 | **Authoritative Synthetic Fusion** |
| **4-Qubit VQC** | Severity Benchmark | Synthetic multimodal dataset (same 30 vectors) | 30 held-out synthetic vectors | **Accuracy: 56.67%** (17 / 30) | MCC | **0.1758** | **46.67%** | Brier: 0.4897 ECE: 0.1420 | **Exploratory Simulation Only** |

---

## 2. Component Deep-Dives

### 2.1 YOLO11 Detection
- **Authoritative Checkpoint:** `ml/models/vision/yolo11_injury_best.pt` (SHA-256: `857880192ebfbc4b6934a033cf7df78310885291ef00a5dd691576c6fbc4bff9`).
- **Real-Data Photographic Evaluation:**
  - Test set size: 23 images (74 bounding boxes across cuts, bruises, wounds).
  - $mAP@50 = 36.47\%$ (sweep baseline 34.54%)
  - $mAP@50\text{--}95 = 18.44\%$ (sweep baseline 16.74%)
  - Precision = $51.59\%$, Recall = $39.42\%$
  - Per-Class $AP@50$: Abrasion: $61.03\%$, Bruise: $60.08\%$, Cut: $50.67\%$, Wound: $19.36\%$, Burn: $16.07\%$, Laceration: $0.00\%$.
- **Error Analysis:** Extreme variation in lighting, hair occlusion, and small superficial skin cuts account for lower detection rates compared to controlled benchmarks.

### 2.2 Sensor Motion Classifier
- **Authoritative Checkpoint:** `ml/models/sensor_motion_best.json` (SHA-256: `4980d2150f379dadf1fef4d8ed61eaf298e67cff8ec495f053d80e9b13e6569d`).
- **Class-Wise Breakdown ($N = 1,228$ Held-Out Windows):**

| Motion Class | Support ($N$) | True Positives | False Positives | False Negatives | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal Activity** | 720 | 711 | 8 | 1 | **98.89%** | **98.75%** | **98.82%** |
| **Fall** | 448 | 427 | 23 | 21 | **94.89%** | **95.31%** | **95.10%** |
| **Impact** | 60 | 45 | 14 | 15 | **76.27%** | **75.00%** | **75.63%** |
| **Macro Average** | **1,228** | **1,183** | **45** | **45** | **90.02%** | **89.69%** | **89.85%** |

- **Confusion Matrix:**
  $$\begin{pmatrix} 711 & 8 & 1 \\ 8 & 427 & 13 \\ 0 & 15 & 45 \end{pmatrix} \quad \begin{matrix} \leftarrow \text{True Normal} \\ \leftarrow \text{True Fall} \\ \leftarrow \text{True Impact} \end{matrix}$$

### 2.3 ResNet34-U-Net Segmentation
- **Authoritative Checkpoint:** `ml/models/vision/unet_injury_best.pt` (SHA-256: `3c7f3f39196d71b9d8d58d1fcfc7438b4ae23d75fbd8195ae7da7b1fcb9660d1`).
- **Held-Out Test Set Metrics ($N = 69$ Image-Mask Pairs):**
  - **Mean Dice:** **64.1808%** ($0.6418$)
  - **Mean IoU:** **51.7783%** ($0.5178$)
  - **Pixel Precision:** **69.30%** ($0.692967$)
  - **Pixel Recall:** **72.33%** ($0.723307$)
  - **Probability Binarization Threshold:** $\mathbf{\tau = 0.50}$
  - **OOD Collapse Instances:** 0

### 2.4 EfficientNetV2-S Classification
- **Authoritative Checkpoint:** `ml/models/vision/efficientnetv2_injury_best.pt`.
- **Held-Out Test Metrics ($N = 473$ Photographic Images):**
  - **Accuracy:** **96.62%** ($457 / 473$)
  - **Macro-F1:** **94.34%** ($0.943364$)
  - **Macro-Precision:** **94.26%**
  - **Macro-Recall:** **95.01%**
  - **Per-Class Breakdown:**
    - Abrasion ($n=23$): F1 = $93.33\%$
    - Bruise ($n=46$): F1 = $93.33\%$
    - Burn ($n=22$): F1 = $86.27\%$
    - Cut ($n=17$): F1 = $94.12\%$
    - Laceration ($n=14$): F1 = $92.31\%$
    - Wound ($n=299$): F1 = $98.48\%$
    - Normal ($n=46$): F1 = $96.84\%$
    - OOD Reject ($n=6$): F1 = $100.00\%$

### 2.5 XGBoost Multimodal vs. 4-Qubit VQC Head-to-Head
Evaluated on the exact same held-out test split of 30 synthetic vectors (`data/results/canonical_held_out_evaluation_corrected.json`):

| Metric | Classical XGBoost | Experimental 4-Qubit VQC | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **Correct Predictions** | **25 / 30** | **17 / 30** | +8 correct |
| **Accuracy** | **83.33%** | **56.67%** | +26.67% |
| **95% Wilson Score CI** | **[66.4%, 92.7%]** | **[39.2%, 72.6%]** | — |
| **Macro Precision** | **0.822222** | **0.508081** | +0.3141 |
| **Macro Recall** | **0.769281** | **0.466013** | +0.3033 |
| **Macro-F1** | **0.785020** | **0.466667** | +0.3184 |
| **Matthews Corr. (MCC)** | **0.702692** | **0.175836** | +0.5269 |
| **Brier Score** (lower is better) | **0.290138** | **0.489704** | -0.1996 (better) |
| **Expected Calibration Error (ECE)** | **0.176072** | **0.141996** | +0.0341 |
| **Execution Hardware** | Classical CPU/GPU | PennyLane `default.qubit` (Classical CPU Sim) | — |
| **Triage Role** | Primary Decision Model | Isolated Exploratory Benchmark | — |

### 2.6 Modality Ablation Ledger (Independently Trained Models on Leak-Free Split)
Evaluated on the exact same held-out test split of 30 synthetic vectors (data/results/independent_ablation_results.json):

| Configuration | Permitted Input Dimensions | Independent Model Accuracy | 95% Wilson CI | Independent Model Macro-F1 | Independent Model MCC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vision Only** | \text{--}F_5$ (=6$) | **40.00%** (12/30) | [24.6%, 57.7%] | **0.2886** | **-0.0610** |
| **Questionnaire Only** | \text{--}F_{17}$ (=12$) | **56.67%** (17/30) | [39.2%, 72.6%] | **0.5036** | **+0.2343** |
| **Sensor Only** | {18}\text{--}F_{22}$ (=5$) | **56.67%** (17/30) | [39.2%, 72.6%] | **0.5010** | **+0.3743** |
| **Vision + Questionnaire** | \text{--}F_{17}$ (=18$) | **53.33%** (16/30) | [36.1%, 69.8%] | **0.4732** | **+0.1982** |
| **Vision + Sensor** | \text{--}F_5, F_{18}\text{--}F_{22}$ (=11$) | **63.33%** (19/30) | [45.5%, 78.1%] | **0.5321** | **+0.3383** |
| **Questionnaire + Sensor** | \text{--}F_{22}$ (=17$) | **90.00%** (27/30) | [74.4%, 96.5%] | **0.8693** | **+0.8246** |
| **Full Multimodal** | \text{--}F_{22}$ (=23$) | **83.33%** (25/30) | [66.4%, 92.7%] | **0.7850** | **+0.7027** |

*Finding:* Fusing Questionnaire and Sensor inputs yields dominant predictive power (+33.33% gain over unimodal baselines). The Full Multimodal architecture achieves strong classification accuracy (83.33%) while contributing spatial lesion bounding boxes (YOLO11 .47\%$ mAP50) and wound surface area measurements (U-Net .18\%$ Dice) for clinician inspection.
