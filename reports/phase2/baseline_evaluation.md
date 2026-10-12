# Phase 2 Baseline Evaluation Report

**Generated:** 2026-10-12  
**Auditor:** Senior Research Reproducibility Auditor & ML Verification Team  
**Scope:** Rigorous held-out evaluation of all 6 active perception and triage models.

---

## 1. Executive Performance Benchmark

All models were evaluated on their strict held-out test sets without cross-split leakage or training contamination.

| Model | Modality | Held-Out Test Set | Primary Metric | Secondary Metric | Reproducibility Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11 Detector** | Bounding Box Detection | 436 test images | **mAP@50: 34.54%** | mAP@50-95: 16.74% | **REPRODUCED** |
| **EfficientNetV2** | Multi-class Classification | 473 test photos | **Accuracy: 96.62%** | Macro F1: 94.34% | **REPRODUCED** |
| **ResNet34-U-Net** | Pixel Wound Segmentation | 69 test pairs | **Dice: 0.6418** | IoU: 0.5178 | **REPRODUCED** |
| **Sensor Classifier** | Kinematic IMU Windows | 1,228 test windows | **Accuracy: 96.34%** | Macro F1: 89.85% | **REPRODUCED** |
| **Multimodal XGBoost**| 23-Feature Tabular Fusion | 30 synthetic test cases | **Accuracy: 83.33%** | Macro F1: 78.50% | **REPRODUCED** |
| **4-Qubit VQC** | 4-Qubit Quantum Circuit | 30 synthetic test cases | **Accuracy: 56.67%** | Macro F1: 42.14% | **REPRODUCED** |

---

## 2. In-Depth Granular Findings & Error Analysis

### 2.1 YOLO11 Object Detector (34.54% mAP@50)
* **Test Size:** 436 unseen photos.
* **Class Performance Breakdown:**
  * `abrasion`: 61.03% AP50
  * `bruise`: 60.08% AP50
  * `cut`: 50.67% AP50
  * `wound`: 19.36% AP50
  * `burn`: 16.07% AP50
  * `laceration`: 0.00% AP50
* **Failure Mode & Misclassification Analysis:**
  * General wound images frequently overlap with severe abrasions and burns in public datasets, creating boundary ambiguity.
  * The `laceration` class achieves 0.00% AP50 on the test set due to extreme annotation sparsity (only 122 training instances across the entire corpus) and high visual similarity to deep cuts.

### 2.2 EfficientNetV2 Classifier (96.62% Accuracy)
* **Test Size:** 473 unseen photos across 8 classes.
* **Error Analysis:**
  * 457 / 473 predictions were strictly correct.
  * Confusions occurred primarily between `bruise` and `burn` (3 cases) and between `laceration` and `cut` (1 case).
  * Out-of-distribution rejection (`ood_reject`) achieved 100% precision and recall (6/6).

### 2.3 ResNet34-U-Net (0.6418 Dice)
* **Test Size:** 69 unseen image/mask pairs across unique patients.
* **Error Analysis:**
  * Precision: 0.6930, Recall: 0.7233, Mean False Positive Area: 2.93%.
  * The model reliably locates lesion centers but exhibits edge ambiguity at the subtle peripheral margins of healing ulcers.

### 2.4 Sensor Motion Event Classifier (96.34% Accuracy)
* **Test Size:** 1,228 unseen 2.5-second windows (SisFall + UCI HAR).
* **Confusion Matrix:**
  * `normal_activity`: 711 correct, 8 misclassified as fall, 1 misclassified as impact.
  * `fall`: 427 correct, 8 misclassified as normal, 13 misclassified as impact.
  * `impact`: 45 correct, 0 misclassified as normal, 15 misclassified as fall.
* **Failure Modes:** Rapid stumble-recovery movements and hard chair sits occasionally trigger fall/impact thresholds.

### 2.5 Multimodal XGBoost (83.33%) vs. 4-Qubit VQC (56.67%)
* **Test Size:** 30 leak-free held-out synthetic patient vectors.
* **XGBoost:** 25 / 30 correct. 3 LOW patients misclassified as MODERATE, 1 HIGH misclassified as MODERATE, 1 MODERATE misclassified as HIGH.
* **VQC:** 17 / 30 correct (loss: 0.886). The quantum circuit demonstrates severe inductive bias toward the majority class (MODERATE), classifying 25 of 30 test cases as MODERATE. This demonstrates the current expressivity bottleneck of 4-qubit parameterized quantum circuits compared to classical gradient boosting.
