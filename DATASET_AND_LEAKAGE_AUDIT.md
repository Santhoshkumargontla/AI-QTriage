# AI-QTriage — Dataset Provenance & Data Leakage Audit (`DATASET_AND_LEAKAGE_AUDIT.md`)

**Document Type:** Dataset Provenance, Partitioning & Leakage Audit  
**Project:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Conference:** ICICNIS 2026 (Article ID: ICICNIS-945)  
**Date:** October 12, 2026  
**Auditor:** Senior Machine Learning Reproducibility Reviewer  

---

## 1. Dataset Provenance & Partitioning Inventory

| Dataset Identifier | Task & Modality | Raw Sources & Repositories | Total Samples | Partitioning Scheme | Split Counts (Train / Val / Test) | Deduplication & Isolation Methods |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| **Photographic Trauma Set** (`yolo_expanded_v1`) | Bounding-Box Detection (YOLO11) | Roboflow Trauma Universe / Kaggle clinical photos | 123 photos (412 boxes) | Stratified random split by wound severity | **83 / 17 / 23** images | Exact image hashing; photographic wounds isolated from synthetic drawings. |
| **Wound Classification Set** (`kaggle_wound_8class`) | Whole-Image Classification (EfficientNetV2-S) | Kaggle Multiclass Skin Lesion / Clinical wound archives | 2,000+ photos across 8 classes | Class-stratified split with OOD controls | **1,500+ / 300+ / 473** test photos | MD5/SHA-256 deduplication; explicit OOD reject class trained. |
| **Wound Segmentation Set** (`unet_deduped_subject`) | Lesion Mask Segmentation (ResNet34-U-Net) | AZH, WSEG, Medetec clinical repositories | 414 image-mask pairs | Subject-level cluster isolation | **276 / 69 / 69** pairs | Perceptual hashing (pHash) removed near-duplicate wounds across patients. |
| **Sensor Telemetry Set** (`sisfall_uci_har_real`) | Motion Event Classification (XGBoost) | SisFall (4,505 windows) + UCI HAR (2,000 windows) | 6,505 2.5s windows (50 Hz) | Subject-level participant segregation | **4,247 / 1,030 / 1,228** windows | Participants strictly partitioned; continuous windows non-overlapping. |
| **Synthetic Multimodal Set** (`multimodal_synthetic_23d`) | Trauma Severity Triage (XGBoost & VQC) | Python parametric simulation (`train_xgboost.py`) | 200 23-dim feature vectors | Index-based deterministic split (seed 42) | **140 / 30 / 30** vectors | Zero test leakage; preprocessing (StandardScaler/PCA) fitted on train only. |

---

## 2. Leakage Audit & Isolation Verification

### 2.1 Multimodal Tabular Leakage Checks
- **Preprocessing Isolation:**
  - `StandardScaler` and `PCA` fitted strictly on $X_{\text{train}}[:140]$.
  - Saved into `ml/models/vqc_scaler.pkl` and `ml/models/vqc_pca.pkl` for runtime inference.
  - XGBoost trees operate on raw unscaled features directly; test features are never passed during scaler fitting.
- **Hash Overlap Findings:**
  - Vector comparison across the 200 synthetic vectors revealed **one vector collision**:
    - Test Index 21 ($X[191]$) has identical values to Training Index 54 ($X[54]$):
      $$\mathbf{x} = [0, 0, 0, 0, 0, 0, 1, 0.1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]$$
    - **Etiology:** Discrete modality dropout zeroed out Vision and Sensor; a mild fall with `pain_score = 0.1` zeroed secondary questionnaire flags.
    - **Labels differ:** Training 54 has label 1 (MODERATE due to underlying unobserved random impact), whereas Test 21 has label 0 (LOW). This is a natural zero-sparse baseline survey state under partial observation rather than an illicit test leak.
  - Zero duplicate feature vectors exist within the 30 test records.

### 2.2 Computer Vision Leakage Isolation
- **U-Net Segmentation:** Perceptual hashing (pHash) was applied across all candidate images from AZH, WSEG, and Medetec prior to partitioning. Near-identical crops of the same ulcer were assigned to the same split.
- **YOLO11 Detection:** Reconciled partitions confirm that 126 synthetic drawings were completely separated from the 123 authentic clinical photographs, preventing synthetic drawing leakage into the real evaluation split.

### 2.3 Sensor Telemetry Subject Isolation
- **Windowing Parameters:** 50 Hz tri-axial acceleration signals divided into 2.5s windows (125 samples).
- **Participant Segregation:** Training, validation, and test windows were partitioned by volunteer participant ID from the SisFall (young and elderly cohorts) and UCI HAR datasets, avoiding temporal leakage from adjacent time-series slices.

---

## 3. Label Provenance & Target Construction Analysis

### 3.1 Severity Ground Truth Construction
In `ml/training/train_xgboost.py`:
$$\text{severity\_score} = 0.4 \times \text{pain} + 0.3 \times \text{peak\_g} + 0.3 \times \text{affected\_area}$$
- $\text{severity\_score} < 0.32 \implies \mathbf{LOW} (0)$
- $0.32 \le \text{severity\_score} < 0.58 \implies \mathbf{MODERATE} (1)$
- $\text{severity\_score} \ge 0.58 \implies \mathbf{HIGH} (2)$

### 3.2 Scientific Implications & Boundaries
- The labels are **rule-derived research heuristics**.
- The model learns to solve a composite physical state estimation problem under noisy, incomplete observation.
- **Scientific Limitation:** While valid as an algorithmic feasibility benchmark, this experiment **does not establish clinical diagnostic effectiveness on human patients**.
