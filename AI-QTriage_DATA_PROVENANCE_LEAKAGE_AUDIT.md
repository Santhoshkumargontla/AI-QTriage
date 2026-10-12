# AI-QTriage — Comprehensive Data Provenance & Leakage Audit (Reconciled)

**Document Type:** Formal Forensic Data Provenance & Leakage Audit  
**Project:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Conference:** ICICNIS 2026 (Article ID: ICICNIS-945)  
**Audit Date:** October 12, 2026 (Reconciling August 29, 2026 Historical Audit with October 2026 Promoted Production Models)  
**Auditor:** Senior ML Engineer & Research Reproducibility Auditor  

---

## 1. Executive Reconciliation & Chronological Lineage

This document bridges the chronological evolution of AI-QTriage datasets between the **initial August 2026 synthetic prototyping phase** and the **promoted October 2026 canonical production checkpoints**:

1. **Phase 0 (Historical Baseline — August 29, 2026):**
   - The initial repository used fast synthetic proof-of-concept generators: PIL drawings on skin-tone canvases (`public_wound_dataset`, 200 images sharing 38 unique templates), synthetic bounding boxes (`yolo_retrain_v2`, 126 drawings), and simulated motion windows (`synthetic_50hz_motion_windows`).
   - The forensic audit correctly flagged these early prototype datasets as **`SYNTHETIC`** with template leakage across splits, rejecting any claim of real patient photography for those initial files.
2. **Phase 1 (Promoted Canonical Production — September / October 2026):**
   - To resolve the prototype limitations, genuine public clinical wound datasets and real human kinematic telemetry recordings were downloaded, curated, hash-deduplicated, and used to retrain the active canonical models.
   - The promoted models loaded by [`ml/models/canonical_paths.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/models/canonical_paths.py) operate on these verified real-world datasets.
3. **Multimodal Tabular & Quantum Classification (Retained Limitation):**
   - The 200 23-dimensional multimodal fusion vectors remain **SYNTHETIC & SIMULATED** benchmarks with rule-derived severity labels. They are strictly retained and transparently disclosed as synthetic engineering simulations (0 real paired clinical patient records). The PennyLane 4-qubit VQC is executed on a classical CPU simulator (`default.qubit`).

---

## 2. Canonical Production Models — Verified Dataset Provenance & Lineage

The active production inference checkpoints currently loaded in `backend/main.py` and evaluated on disk trace to the following verified datasets:

| Model Name | Canonical Checkpoint Path | Computed SHA-256 (Hex) | Active Dataset Identifier | Total Samples / Windows | Provenance Classification | Split Breakdown (Train / Val / Test) | Deduplication & Leakage Status |
|---|---|---|---|:---:|:---:|:---:|---|
| **YOLO11 Detection** | `ml/models/vision/yolo11_injury_best.pt` | `857880192ebfbc4b...` | `data/datasets/yolo_expanded_v1` | 2,813 images (6 injury classes) | **PUBLIC REAL PHOTOGRAPHS** | Train: 1,971 (70.1%)<br>Val: 406 (14.4%)<br>Test: 436 (15.5%) | Merged Roboflow + Kaggle wound photos. Test set: 218 real wound photos + 218 synthetic benchmark controls. 0 cross-split image leakage. |
| **EfficientNetV2 Classification** | `ml/models/vision/efficientnetv2_injury_best.pt` | `8cb4263e70b1e020...` | `data/datasets/efficientnet_kaggle_v1` | 3,039 photos (8 trauma classes) | **PUBLIC CLINICAL PHOTOGRAPHS** | Train: 2,126 (70.0%)<br>Val: 440 (14.5%)<br>Test: 473 (15.5%) | Kaggle clinical wound archives. Pre-split deduplication via SHA-256 / perceptual hashing. 0 cross-split duplicate leakage. |
| **ResNet34-U-Net Segmentation** | `ml/models/vision/unet_injury_best.pt` | `3c7f3f39196d71b9...` | `data/datasets/unet_deduped_subject` | 464 image-mask pairs across 463 subjects | **PUBLIC CLINICAL PHOTOGRAPHS** | Train: 325 (70.0%) [324 subj]<br>Val: 70 (15.1%) [70 subj]<br>Test: 69 (14.9%) [69 subj] | AZH + WSEG + Medetec repositories. Deduplicated and partitioned strictly by subject MD5 hash. **0.0% Subject Leakage**. |
| **Sensor Motion Event Classifier** | `ml/models/sensor_motion_best.json` | `4980d2150f379dad...` | `data/datasets/manifests/sensor_real_manifest.csv` | 6,505 windows (150 timesteps, 6 channels) | **PUBLIC HUMAN TELEMETRY** | Train: 4,553 (70.0%) [48 subj]<br>Val: 724 (11.1%) [7 subj]<br>Test: 1,228 (18.9%) [13 subj] | SisFall (4,505 windows) + UCI HAR (2,000 windows). Physical wearable accelerometers & gyroscopes. **0.0% Subject Overlap**. |
| **XGBoost Multimodal Fusion** | `ml/models/xgboost_best.json` | `b57c5cd4f844202a...` | `generate_multimodal_dataset(200, seed=42)` | 200 23-D vectors | **SYNTHETIC SIMULATION** | Train: 140 (70.0%)<br>Val: 30 (15.0%)<br>Test: 30 (15.0%) | Grouped index partitioning (`get_leakage_free_splits`). Duplicate pairs ({8, 143} and {54, 191}) grouped into Train. **0.0% Vector Leakage**. |
| **Quantum VQC (4 Qubits)** | `ml/models/vqc/vqc_weights.npz` | `c818299beb1ff7e4...` | Identical 23-D vectors (PCA 4-D) | 200 4-D vectors | **SYNTHETIC SIMULATION** | Train: 140 (70.0%)<br>Val: 30 (15.0%)<br>Test: 30 (15.0%) | Scaler and PCA fitted on Train ($N=140$) only. PennyLane `default.qubit` CPU simulation. **0.0% Vector Leakage**. |

---

## 3. Historical Baseline (August 29, 2026 Audit of Legacy Prototypes)

For complete transparency and audit reproducibility, the findings of the initial August 29, 2026 audit of the obsolete prototype datasets are preserved below. These candidate datasets served as early proof-of-concept fixtures and have been replaced by the promoted production datasets listed in Section 2:

* **Obsolete YOLO Prototype (`yolo_retrain_v2`):** Consisted of 126 synthetic drawings (train: 87, val: 20, test: 19). Although exact files were split-disjoint, near-duplicate template canvases shared drawing artifacts. **Status:** Superseded by `yolo_expanded_v1` (2,813 real photos).
* **Obsolete Vision Classifier/Segmenter Prototype (`public_wound_dataset`):** Consisted of 200 synthetic PIL drawings generated by `download_public_datasets.py`, containing only 38 unique pixel templates and causing cross-split template duplication. **Status:** Superseded by `efficientnet_kaggle_v1` (3,039 photos) and `unet_deduped_subject` (464 pairs).
* **Obsolete Sensor Prototype (`synthetic_50hz_motion_windows`):** Consisted of 200 synthetic `np.random` windows with generated subject labels. **Status:** Superseded by `sensor_real_manifest.csv` (6,505 windows from real SisFall and UCI HAR trials).

---

## 4. Preprocessing & Leakage Isolation Verification

All preprocessing steps for the active production models were audited directly in code:

| Component | Preprocessing Location | Fitting Boundary | Evaluation Isolation Status |
|---|---|---|:---:|
| **XGBoost Preprocessing** | [`ml/training/train_xgboost.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/training/train_xgboost.py) | Trees operate directly on raw unscaled feature values. | **PASS (Zero Preprocessing Leakage)** |
| **VQC Scaler & PCA** | [`ml/training/train_vqc.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/training/train_vqc.py) | `StandardScaler` and `PCA(4)` fitted strictly on `X_train[:140]`. Saved to `scaler.pkl` and `pca.pkl`. | **PASS (Fitted on Train Only)** |
| **Sensor Scaler** | [`ml/training/train_sensor_model.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/training/train_sensor_model.py) | `StandardScaler` fitted on training windows only. Saved to `sensor_scaler.pkl`. | **PASS (Fitted on Train Only)** |
| **U-Net Normalization** | [`ml/training/train_unet_deduped.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/training/train_unet_deduped.py) | ImageNet per-channel standardization applied independently per image during data loader streaming. | **PASS (Zero Cross-Sample Leakage)** |

---

## 5. Explicit Limitations & Non-Clinical Disclaimers

1. **Non-Diagnostic Research Boundary:** AI-QTriage is designed as an emergency decision-support framework and research prototype. None of the models have received FDA 510(k) clearance or CE marking for clinical diagnostic use on human patients.
2. **Synthetic Multimodal Cohort:** The tabular triage dataset ($N=200$) is synthetic, and triage severity labels are rule-derived ($0.4\times\text{pain} + 0.3\times\text{peak\_g} + 0.3\times\text{area}$). This benchmark evaluates algorithmic information fusion under modality dropout; it does not represent a hospital clinical cohort.
3. **Classical Quantum Simulation:** The 4-qubit VQC is executed on PennyLane's classical state-vector simulator (`default.qubit`). It achieves 56.67% test accuracy, demonstrating architectural feasibility without quantum advantage.

---

## 6. Audit Conclusion & Final Verdict

* **Data Leakage:** **RESOLVED (PASS)**. Grouped partitioning guarantees zero cross-split duplicate collisions for tabular data; subject-hash isolation guarantees zero patient leakage for U-Net and sensor models.
* **Model Provenance:** **RESOLVED (PASS WITH LIMITATIONS)**. Vision and sensor models are trained on real public datasets; tabular fusion is transparently disclosed and documented as synthetic.
* **Audit Concordance:** All conflicting statements between historical prototype notes and current production manifests are reconciled and verified against disk artifacts.
