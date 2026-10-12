# AI-QTriage — Metric Consistency Matrix (`METRIC_CONSISTENCY_MATRIX.md`)

**Document Type:** Master Metric Consistency Ledger  
**Project:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Conference:** ICICNIS 2026 (Article ID: ICICNIS-945)  
**Date:** October 12, 2026  
**Auditor:** Senior Machine Learning Engineer  

---

## Master Metric Cross-Verification Table

| Component & Task | Reported Manuscript Value | Verified Evaluation Output | Source Artifact / Checkpoint | Evaluation Split | Sample Count ($N$) | Discrepancy Status |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **YOLO11 Detection (mAP@50)** | **36.47%** | **0.3647** (0.3454 baseline) | `ml/models/vision/yolo11_injury_best.pt` | Real Photographic Test | 23 images (74 boxes) | **CONCORDANT (PASS)** |
| **YOLO11 Detection (mAP@50–95)** | **18.44%** | **0.1844** (0.1674 baseline) | `ml/models/vision/yolo11_injury_best.pt` | Real Photographic Test | 23 images (74 boxes) | **CONCORDANT (PASS)** |
| **YOLO11 Precision** | **51.59%** | **0.5159** | `MODEL_EVALUATION_REPORT.md` Sec 2.1 | Real Photographic Test | 23 images (74 boxes) | **CONCORDANT (PASS)** |
| **YOLO11 Recall** | **39.42%** | **0.3942** | `MODEL_EVALUATION_REPORT.md` Sec 2.1 | Real Photographic Test | 23 images (74 boxes) | **CONCORDANT (PASS)** |
| **EfficientNetV2 Accuracy** | **96.62%** | **0.9662** (457 / 473) | `ml/models/vision/efficientnetv2_injury_best.pt` | Held-Out Photos | 473 images | **CONCORDANT (PASS)** |
| **EfficientNetV2 Macro-F1** | **94.34%** | **0.9434** | `ml/models/vision/efficientnetv2_injury_best.pt` | Held-Out Photos | 473 images | **CONCORDANT (PASS)** |
| **ResNet34-U-Net Mean Dice** | **64.18%** | **0.641808** | `ml/models/vision/unet_metadata.json` | Held-Out Subject Pairs | 69 pairs | **CONCORDANT (PASS)** |
| **ResNet34-U-Net Mean IoU** | **51.78%** | **0.517783** | `ml/models/vision/unet_metadata.json` | Held-Out Subject Pairs | 69 pairs | **CONCORDANT (PASS)** |
| **U-Net Threshold ($\tau$)** | **0.50** | **0.50** | `ml/vision/unet_wrapper.py` L64 | Inference Gate | Continuous sigmoid | **CONCORDANT (PASS)** |
| **Sensor Overall Accuracy** | **96.34%** | **0.963355** (1,183 / 1,228) | `ml/models/sensor_metadata.json` | SisFall + UCI HAR Test | 1,228 windows (50 Hz) | **CONCORDANT (PASS)** |
| **Sensor Macro-F1** | **89.85%** | **0.898497** | `ml/models/sensor_metadata.json` | SisFall + UCI HAR Test | 1,228 windows (50 Hz) | **CONCORDANT (PASS)** |
| **Sensor MCC** | **0.9296** | **0.929623** | `ml/models/sensor_metadata.json` | SisFall + UCI HAR Test | 1,228 windows (50 Hz) | **CONCORDANT (PASS)** |
| **Sensor Normal Class F1** | **98.82%** | **0.9882** (711 TP / 720) | `ml/models/sensor_metadata.json` | Held-Out Normal Windows | 720 windows | **CONCORDANT (PASS)** |
| **Sensor Fall Class F1** | **95.10%** | **0.9510** (427 TP / 448) | `ml/models/sensor_metadata.json` | Held-Out Fall Windows | 448 windows | **CONCORDANT (PASS)** |
| **Sensor Impact Class Recall** | **75.00%** | **0.7500** (45 TP / 60) | `ml/models/sensor_metadata.json` | Held-Out Impact Windows | 60 windows | **CONCORDANT (PASS)** |
| **XGBoost Test Accuracy** | **83.33%** | **0.833333** (25 / 30) | `ml/models/xgboost_best.json` | Synthetic Test Split | 30 vectors ($[170:200]$) | **CONCORDANT (PASS)** |
| **XGBoost 95% Wilson CI** | **[65.3%, 94.4%]** | **[0.653, 0.944]** | Wilson Score formula ($k=25, n=30$) | Synthetic Test Split | 30 vectors ($[170:200]$) | **CONCORDANT (PASS)** |
| **XGBoost Macro-F1** | **0.7850** | **0.785020** | `data/results/canonical_held_out_evaluation.json` | Synthetic Test Split | 30 vectors ($[170:200]$) | **CONCORDANT (PASS)** |
| **XGBoost MCC** | **0.7027** | **0.702692** | `data/results/canonical_held_out_evaluation.json` | Synthetic Test Split | 30 vectors ($[170:200]$) | **CONCORDANT (PASS)** |
| **VQC Test Accuracy** | **53.33%** | **0.533333** (16 / 30) | `data/results/canonical_held_out_evaluation.json` | PennyLane CPU Sim | 30 vectors ($[170:200]$) | **CONCORDANT (PASS)** |
| **VQC Macro-F1** | **0.4230** | **0.423016** | `data/results/canonical_held_out_evaluation.json` | PennyLane CPU Sim | 30 vectors ($[170:200]$) | **CONCORDANT (PASS)** |
| **VQC MCC** | **0.0960** | **0.095978** | `data/results/canonical_held_out_evaluation.json` | PennyLane CPU Sim | 30 vectors ($[170:200]$) | **CONCORDANT (PASS)** |
| **Ablation: Vision Only (Indep)** | **33.33%** | **0.3333** (10 / 30) | `data/results/independent_ablation_results.json` | Held-Out Test ($D=6$) | 30 vectors | **CONCORDANT (PASS)** |
| **Ablation: Quest Only (Indep)** | **56.67%** | **0.5667** (17 / 30) | `data/results/independent_ablation_results.json` | Held-Out Test ($D=12$) | 30 vectors | **CONCORDANT (PASS)** |
| **Ablation: Sensor Only (Indep)** | **56.67%** | **0.5667** (17 / 30) | `data/results/independent_ablation_results.json` | Held-Out Test ($D=5$) | 30 vectors | **CONCORDANT (PASS)** |
| **Ablation: Quest + Sensor (Indep)** | **86.67%** | **0.8667** (26 / 30) | `data/results/independent_ablation_results.json` | Held-Out Test ($D=17$) | 30 vectors | **CONCORDANT (PASS)** |
| **Ablation: Full Multimodal (Indep)** | **83.33%** | **0.8333** (25 / 30) | `data/results/independent_ablation_results.json` | Held-Out Test ($D=23$) | 30 vectors | **CONCORDANT (PASS)** |

---

## Discrepancy Reconciliation Summary
- **Historical Conflicts Resolved:**
  - YOLO11: Reconciled 83.58% synthetic drawing baseline vs. 36.47% real photographic trauma benchmark. Precision (51.59%) and Recall (39.42%) verified.
  - Sensor: Reconciled 77.78% obsolete toy series metric vs. 96.34% SisFall/UCI HAR benchmark.
  - EfficientNetV2: Reconciled intermediate 93.75% validation figure vs. 94.34% test macro-F1.
  - Reference 10: Reconciled UniMiB SHAR citation mismatch by updating to the authoritative UCI HAR smartphone paper (Anguita et al., 2013).
- **Current Numerical State:** **100% Concordance across all tables, abstract, results, and evaluation artifacts.**
