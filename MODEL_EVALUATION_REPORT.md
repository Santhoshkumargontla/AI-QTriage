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
| **Multimodal XGBoost** | Severity Triage | Synthetic multimodal dataset (`train_xgboost.py`) | 30 held-out synthetic vectors | **Accuracy: 83.33%** (25 / 30) | MCC | **0.7027** | **78.50%** | Brier: 0.3045 ECE: 0.1183 | **Authoritative Synthetic Fusion** |
| **4-Qubit VQC** | Severity Benchmark | Synthetic multimodal dataset (same 30 vectors) | 30 held-out synthetic vectors | **Accuracy: 53.33%** (16 / 30) | MCC | **0.0960** | **42.30%** | Brier: 0.5123 ECE: 0.1365 | **Exploratory Simulation Only** |

---

## 2. Component Deep-Dives

### 2.1 YOLO11 Detection
- **Authoritative Checkpoint:** `ml/models/vision/yolo11_injury_best.pt` (SHA-256: `4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879`).
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
Evaluated on the exact same held-out test split of 30 synthetic vectors (`data/results/canonical_held_out_evaluation.json`):

| Metric | Classical XGBoost | Experimental 4-Qubit VQC | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **Correct Predictions** | **25 / 30** | **16 / 30** | +9 correct |
| **Accuracy** | **83.33%** | **53.33%** | +30.00% |
| **95% Wilson Score CI** | **[65.3%, 94.4%]** | **[36.1%, 69.8%]** | — |
| **Macro Precision** | **0.822222** | **0.466184** | +0.3560 |
| **Macro Recall** | **0.769281** | **0.432680** | +0.3366 |
| **Macro-F1** | **0.785020** | **0.423016** | +0.3620 |
| **Matthews Corr. (MCC)** | **0.702692** | **0.095978** | +0.6067 |
| **Brier Score** (lower is better) | **0.304532** | **0.512306** | -0.2078 (better) |
| **Expected Calibration Error (ECE)** | **0.118284** | **0.136455** | -0.0182 (better) |
| **Execution Hardware** | Classical CPU/GPU | PennyLane `default.qubit` (Classical CPU Sim) | — |
| **Triage Role** | Primary Decision Model | Isolated Exploratory Benchmark | — |

### 2.6 Modality Ablation Ledger (Independently Trained Models vs. Zero-Mask Baseline)
Evaluated on the exact same held-out test split of 30 synthetic vectors (`data/results/independent_ablation_results.json`):

| Configuration | Permitted Input Dimensions | Independent Model Accuracy | Independent Model Macro-F1 | Independent Model MCC | Zero-Mask Baseline Accuracy | Zero-Mask Baseline MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vision Only** | $F_0\text{--}F_5$ ($D=6$) | **33.33%** (10/30) | **0.2414** | **-0.1632** | 46.67% | -0.1020 |
| **Questionnaire Only** | $F_6\text{--}F_{17}$ ($D=12$) | **56.67%** (17/30) | **0.5036** | **+0.2343** | 56.67% | +0.1906 |
| **Sensor Only** | $F_{18}\text{--}F_{22}$ ($D=5$) | **56.67%** (17/30) | **0.5010** | **+0.3743** | 46.67% | -0.0565 |
| **Vision + Questionnaire** | $F_0\text{--}F_{17}$ ($D=18$) | **43.33%** (13/30) | **0.3588** | **-0.0464** | 53.33% | +0.1406 |
| **Vision + Sensor** | $F_0\text{--}F_5, F_{18}\text{--}F_{22}$ ($D=11$) | **60.00%** (18/30) | **0.5080** | **+0.2968** | 36.67% | +0.0573 |
| **Questionnaire + Sensor** | $F_6\text{--}F_{22}$ ($D=17$) | **86.67%** (26/30) | **0.8148** | **+0.7609** | 83.33% | +0.7135 |
| **Full Multimodal** | $F_0\text{--}F_{22}$ ($D=23$) | **83.33%** (25/30) | **0.7850** | **+0.7027** | 83.33% | +0.7027 |

*Finding:* Fusing Questionnaire and Sensor inputs yields dominant predictive power (+30.00% gain over unimodal baselines). The Full Multimodal architecture achieves comparable classification accuracy (83.33%) while contributing spatial lesion bounding boxes (YOLO11 $36.47\%$ mAP50) and wound surface area measurements (U-Net $64.18\%$ Dice) for clinician inspection.
