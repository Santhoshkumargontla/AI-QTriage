# AI-QTriage — Dataset and Split Report (`DATASET_AND_SPLIT_REPORT.md`)

**Project Title:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Conference:** 7th International Conference on IoT based Control Networks and Intelligent Systems (ICICNIS 2026)  
**Paper ID:** ICICNIS-945  
**Document Classification:** Empirical Dataset & Partition Audit  
**Date:** October 12, 2026  

---

## 1. Master Dataset Summary Table

| Model Component | Exact Dataset Source | Modality / Domain | Total Raw Samples | Classes | Partition (Train / Val / Test) | Split Method & Subject Independence |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| **YOLO11 (Baseline Drawing)** | `data/datasets/yolo_retrain_v2` | Synthetic digitized wound sketches | 126 images | `cut`, `bruise`, `wound` | 87 / 20 / 19 images | Random split. Wound class has 0 boxes in train. |
| **YOLO11 (Photographic Candidate)** | `yolo_candidate_training` (`yolo_expanded_v1`) | Real photographic wound repository | 123 images | `cut`, `bruise`, `wound` | 83 / 17 / 23 images (412 total instances) | Image-level split. $83 + 17 + 23 = 123$ images. |
| **YOLO11 (Expanded Skin)** | `data/datasets/yolo_real_skin_v2` | Merged Roboflow, WSEG, Medetec | 1,790 images | 6 classes (`cut`, `bruise`, `abrasion`, `burn`, `wound`, `laceration`) | 1,246 / 270 / 274 images | Photographic split with empty skin controls. |
| **EfficientNetV2-S** | Kaggle multi-class wound photos + controls | Photographic skin trauma images | 2,000+ photos | 8 classes (`abrasion`, `bruise`, `burn`, `cut`, `laceration`, `wound`, `normal`, `ood_reject`) | ~1,600 / ~300 / 473 test | Grouped partition. Swelling removed for OOD safety. |
| **ResNet34-U-Net** | AZH, WSEG, and Medetec wound databases | Paired RGB wound photographs & binary masks | 414 pairs | Binary `wound_mask` | 276 / 69 / 69 pairs | **Subject-aware deduplicated split.** Exact/near duplicates purged via perceptual hash (`pHash`). |
| **Sensor Motion Classifier** | SisFall + UCI HAR smartphones (`sisfall_uci_har_real`) | 50Hz tri-axial accelerometer streams | 6,505 windows (2.5s duration) | `normal_activity`, `fall`, `impact` | 4,247 / 1,030 / 1,228 windows | **Subject-level split.** SisFall subject IDs segregated across splits; zero window overlap across partitions. |
| **XGBoost & VQC** | Synthetic multimodal generator (`train_xgboost.py`) | 23-dimensional numerical vectors | 200 vectors | `LOW`, `MODERATE`, `HIGH` | 140 / 30 / 30 vectors | Deterministic slice: $[:140]$, $[140:170]$, $[170:]$. Rule-derived synthetic research labels. |

---

## 2. Dataset Reconciliation & Discrepancy Resolution

### 2.1 YOLO11 Dataset Counts (Reviewer 2 Comment 1)
- **Reviewer Critique:** *"The YOLO11 dataset contains 123 images, but the reported training, validation, and test split counts are not clearly reconciled."*
- **Reconciliation:**
  The project developed two distinct YOLO datasets that were conflated in early drafts:
  1. **Photographic Trauma Dataset ($N = 123$):**
     - **Train:** 83 images (288 wound, 12 bruise, 15 cut boxes)
     - **Validation:** 17 images (70 wound, 1 bruise, 2 cut boxes)
     - **Held-Out Test:** 23 images (65 wound, 4 bruise, 5 cut boxes)
     - **Sum:** $83 + 17 + 23 = \mathbf{123\text{ images}}$ (412 total bounding boxes).
  2. **Synthetic Drawing Dataset ($N = 126$):**
     - **Train:** 87 images (47 cut, 40 bruise, 0 wound boxes)
     - **Validation:** 20 images (6 cut, 13 bruise, 0 wound boxes)
     - **Held-Out Test:** 19 images (8 cut, 10 bruise, 0 wound boxes)
     - **Sum:** $87 + 20 + 19 = \mathbf{126\text{ images}}$.
  - **Resolution:** In the camera-ready manuscript, Section 3.1 explicitly distinguishes the 123-image photographic dataset ($83/17/23$) from the 126-image drawing baseline.

### 2.2 Sensor Dataset Counts (Reviewer 2 Comment 4)
- **Reviewer Critique:** *"The sensor classifier reports 96.34% accuracy on 1,228 samples, but the corresponding class-wise performance metrics are not reported."*
- **Reconciliation:**
  - The manuscript's earlier claim of $200$ series and $28/36 = 77.78\%$ accuracy was from an obsolete synthetic test file (`synthetic_sensor_v1.2.0`).
  - The authoritative evaluation used real IMU recordings from SisFall (4,505 windows) and UCI HAR (2,000 windows). Segmenting into 2.5-second windows (125 time steps at 50Hz) produced 6,505 windows:
    - **Train Split:** 4,247 windows (3,057 normal, 1,050 fall, 140 impact)
    - **Validation Split:** 1,030 windows
    - **Held-Out Test Split:** 1,228 windows (720 normal, 448 fall, 60 impact)
  - **Per-Class Metrics on Untouched Test Split ($N = 1,228$):**
    - **Normal Activity ($n = 720$):** Precision = $98.89\%$, Recall = $98.75\%$, F1 = $98.82\%$
    - **Fall ($n = 448$):** Precision = $94.89\%$, Recall = $95.31\%$, F1 = $95.10\%$
    - **Impact ($n = 60$):** Precision = $76.27\%$, Recall = $75.00\%$, F1 = $75.63\%$
    - **Overall Accuracy:** $\frac{1,183}{1,228} = \mathbf{96.3355\% \approx 96.34\%}$, Macro-F1 = $\mathbf{89.85\%}$.

---

## 3. Multimodal 23-Feature Vector Schema & Missingness Protocol

The 23 canonical features in [`ml/fusion/feature_fusion.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/fusion/feature_fusion.py#L4-L16) are defined as follows:

| ID | Feature Name | Modality | Value Range | Normalization / Extraction Formula | Meaning of Zero Value |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **F0** | `vision_present` | Vision | $\{0.0, 1.0\}$ | 1.0 if image analyzed; 0.0 if missing/failed quality gate. | Vision branch absent. |
| **F1** | `prob_cut` | Vision | $[0.0, 1.0]$ | EfficientNetV2 Softmax probability for Cut. | Zero probability or vision absent. |
| **F2** | `prob_bruise` | Vision | $[0.0, 1.0]$ | EfficientNetV2 Softmax probability for Bruise. | Zero probability or vision absent. |
| **F3** | `prob_swelling` | Vision | $[0.0, 1.0]$ | Hardcoded to $0.0$ in current runtime (class eliminated for OOD safety). | Deactivated feature. |
| **F4** | `prob_other` | Vision | $[0.0, 1.0]$ | $\max(\text{Other}, \text{Normal}, \text{OOD}, \text{Burn}, \text{Wound}, \dots)$ | Zero probability or vision absent. |
| **F5** | `affected_ratio` | Vision | $[0.0, 1.0]$ | $\frac{\text{U-Net Mask Foreground Area}}{\text{Total Image Area}}$ (or YOLO BBox ratio fallback). | No visible lesion segmented. |
| **F6** | `questionnaire_present`| Survey | $\{0.0, 1.0\}$ | 1.0 if survey completed; 0.0 if omitted. | Questionnaire absent. |
| **F7** | `pain_level` | Survey | $[0.0, 1.0]$ | $\mathbf{F_7 = \frac{\text{Pain}_{\text{raw}}}{10.0}}$ for integer $\text{Pain}_{\text{raw}} \in [0, 10]$. | No pain reported ($0/10$). |
| **F8** | `mech_fall` | Survey | $\{0.0, 1.0\}$ | 1.0 if mechanism is "fall" or "accident"; else 0.0. | Not a fall mechanism. |
| **F9** | `mech_impact` | Survey | $\{0.0, 1.0\}$ | 1.0 if mechanism is "direct_impact" or "blow"; else 0.0. | Not a direct impact. |
| **F10** | `mech_sports` | Survey | $\{0.0, 1.0\}$ | 1.0 if mechanism is "sports" or "twist"; else 0.0. | Not a sports injury. |
| **F11** | `mech_sharp` | Survey | $\{0.0, 1.0\}$ | 1.0 if mechanism is "sharp_object" or "cut"; else 0.0. | Not a sharp object injury. |
| **F12** | `mech_other` | Survey | $\{0.0, 1.0\}$ | 1.0 if mechanism is specified but not F8–F11; else 0.0. | No other mechanism. |
| **F13** | `direct_impact` | Survey | $\{0.0, 1.0\}$ | 1.0 if user checked "Direct Impact"; else 0.0. | No direct impact reported. |
| **F14** | `visible_bleeding` | Survey | $\{0.0, 1.0\}$ | 1.0 if bleeding reported ("yes", "mild", "heavy"); else 0.0. | No bleeding reported. |
| **F15** | `movement_limitation`| Survey | $\{0.0, 0.5, 1.0\}$| None = 0.0; Mild/Limited = 0.5; Severe/Cannot move = 1.0. | Normal range of motion. |
| **F16** | `weight_bearing` | Survey | $\{0.0, 0.5, 1.0\}$| Full = 0.0; Partial/Painful = 0.5; Unable = 1.0. | Full weight bearing intact. |
| **F17** | `crack_pop` | Survey | $\{0.0, 1.0\}$ | 1.0 if crack or pop felt/heard; else 0.0. | No structural pop/crack. |
| **F18** | `sensor_present` | Sensor | $\{0.0, 1.0\}$ | 1.0 if IMU log provided; 0.0 if omitted. | Sensor absent. |
| **F19** | `peak_g_force` | Sensor | $\ge 0.0$ | $\max_t \left(\frac{\|a(t)\|}{9.80665}\right)$ in g-force units. | Baseline/absent. |
| **F20** | `delta_v` | Sensor | $\ge 0.0$ | Trapezoidal pre-impact velocity delta $\int_{t_p-1}^{t_p} \|a\| dt$ in m/s. | Zero velocity delta. |
| **F21** | `stabilization_time` | Sensor | $\ge 0.0$ | Post-impact duration for acceleration to remain $< 1.5g$. | Zero stabilization time. |
| **F22** | `lux_drop` | Sensor | $\{0.0, 1.0\}$ | 1.0 if ambient optical lux drops $< 10.0\text{ lux}$ at impact; else 0.0. | Lux did not drop $< 10.0$. |

### 3.1 Mapping of YOLO Detection vs. Multimodal Class Probabilities
- YOLO11 outputs spatial bounding boxes with classes `cut`, `bruise`, `wound` (and in expanded models: `abrasion`, `burn`, `laceration`).
- **Critical Architectural Fact:** YOLO detections do **not** directly populate F1–F4. Instead:
  1. YOLO bounding boxes crop the Region of Interest (ROI).
  2. EfficientNetV2 classifies the cropped ROI to produce class probabilities F1–F4 (`prob_cut`, `prob_bruise`, `prob_swelling`, `prob_other`).
  3. YOLO's bounding box area serves as an alternative fallback for F5 (`affected_ratio`) when U-Net segmentation fails.

---

## 4. Train/Test Leakage & Hash Overlap Verification

An exhaustive vector hash comparison of all 200 synthetic vectors in `train_xgboost.py` revealed:
- **Test Row Index 21 ($X_{\text{test}}[21]$) == Training Row Index 84 ($X_{\text{train}}[84]$)**
- **Root Cause:** Modality dropout randomly zeroed out vision and sensor features for both rows, leaving only questionnaire features. Due to the small combinatorial space of discrete survey options, two independent random draws produced an identical survey vector.
- **Impact on Evaluation:** $1$ of the $30$ test vectors ($3.33\%$) is identical to a training sample. When re-evaluating XGBoost strictly on the remaining $29$ unique test samples, $24 / 29 = 82.76\%$ accuracy is achieved (compared to $25 / 30 = 83.33\%$).

---

## 5. Label Heuristics & Clinical Limitations

The multimodal labels `LOW` ($0$), `MODERATE` ($1$), and `HIGH` ($2$) are **not real clinical ground-truth diagnoses**. They were generated via:
$$\text{Severity} = 0.4 \times \text{pain} + 0.3 \times \text{peak\_g} + 0.3 \times \text{affected\_area}$$
- $\text{Severity} < 0.32 \implies \text{LOW}$
- $0.32 \le \text{Severity} < 0.58 \implies \text{MODERATE}$
- $\text{Severity} \ge 0.58 \implies \text{HIGH}$

**Clinical Reality:** The multimodal XGBoost model demonstrates software integration and multi-input fusion on synthetic research data. It **has not been clinically validated** on human patient cohorts. All manuscript sections must maintain this explicit disclaimer.
