# AI-QTriage — Phase 1 Changelog (`CHANGELOG_PHASE1.md`)

**Project Title:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Conference:** 7th International Conference on IoT based Control Networks and Intelligent Systems (ICICNIS 2026)  
**Paper ID:** ICICNIS-945  
**Audit Phase:** Phase 1 (Forensic Verification & Model Improvement Preparation)  
**Date:** October 12, 2026  

---

## Summary of Changes

All changes were performed in strict compliance with non-destructive audit principles: no existing checkpoints, datasets, or evaluation scripts were overwritten. New reproducible scripts and comprehensive audit documentation were added to provide complete evidence chains for camera-ready revision.

---

## Detailed File Ledger

### 1. New Evidence & Audit Deliverables
- **`PHASE1_AUDIT.md` (Created):**
  - **Purpose:** Comprehensive forensic audit report investigating all 6 model components, tracking the root cause of every numerical discrepancy between Section III and Section VI, analyzing train/test hash overlap, and evaluating synthetic ground truth heuristics.
- **`DATASET_AND_SPLIT_REPORT.md` (Created):**
  - **Purpose:** Definitive ledger of all dataset sources, partitions, sample counts, and leakage checks. Reconciles the 123-image YOLO photographic set ($83/17/23$) vs. 126-image drawing set, the 1,228-window sensor set vs. obsolete 200 series, documents the full 23-feature vector schema, and records the exact pain normalization formula ($F_7 = \text{Pain} / 10.0$).
- **`MODEL_EVALUATION_REPORT.md` (Created):**
  - **Purpose:** Authoritative reference ledger containing verified test metrics for all 6 models (YOLO11, EfficientNetV2-S, ResNet34-U-Net, Sensor Classifier, Multimodal XGBoost, and 4-Qubit VQC).
- **`ABLATION_RESULTS.md` (Created):**
  - **Purpose:** Empirical report of the 7-configuration modality ablation experiment resolving Reviewer 1 Comment 4. Documents accuracy, Macro-F1, MCC, and confusion matrices showing that multimodal integration achieves +26.66% to +36.66% accuracy improvements over single-modality baselines.
- **`REVIEWER_RESPONSE.md` (Created):**
  - **Purpose:** Formal point-by-point response document addressing all 11 comments from Review Comments-1 (#1–#6) and Review Comments-2 (#1–#5) with exact manuscript and repository citations.
- **`CHANGELOG_PHASE1.md` (Created):**
  - **Purpose:** Master ledger of all modifications, scripts, and artifacts introduced during Phase 1.
- **`FINAL_VALIDATION.md` (Created):**
  - **Purpose:** Verification summary of non-destructive tests, reproducibility commands, remaining limitations, and final submission readiness.

### 2. Experimental Data & Verification Scripts
- **`scratch/run_independent_modality_ablation.py` (Created):**
  - **Purpose:** Standalone script that independently trains 7 separate XGBoost models strictly on permitted feature subsets ($F_0\text{--}F_5$, $F_6\text{--}F_{17}$, $F_{18}\text{--}F_{22}$, etc.) without out-of-distribution zero-filling artifacts.
- **`data/results/independent_ablation_results.json` (Created):**
  - **Purpose:** Serialized JSON results of the independently trained models with confusion matrices, 95% Wilson Score CIs, and per-class metrics.
- **`ABLATION_METHOD_REVIEW.md` (Created):**
  - **Purpose:** Formal methodological audit analyzing feature-masking vs. independent modality training, target formula dependency, and synthetic label boundaries.
- **`scratch/run_modality_ablation.py` (Created):**
  - **Purpose:** Standalone, reproducible script that executes modality feature masking under Option-A missingness on the held-out test split ($N = 30$) and computes evaluation metrics across all 7 configurations.
- **`data/results/canonical_ablation_results.json` (Created):**
  - **Purpose:** Serialized JSON results of the modality ablation study containing confusion matrices, accuracies, precision, recall, and MCC for each configuration.
- **`scratch/verify_phase1_live.py` (Created):**
  - **Purpose:** Non-destructive test script verifying XGBoost live predictions on held-out test data and detecting row hash overlaps across splits.
- **`scratch/verify_vqc_live.py` (Created):**
  - **Purpose:** Non-destructive test script executing PennyLane's `default.qubit` forward pass on held-out test data to independently verify the 16/30 (53.33%) VQC accuracy.

### 3. Application Alignment & Bug Fixes
- **`backend/main.py`:** Added missing FastAPI `Request` import.
- **`streamlit_app.py`:** Synchronized `render_model_benchmarks()` with verified canonical metrics (YOLO 36.47% mAP50, EfficientNet 96.62%, U-Net 64.18%, Sensor 96.34%, XGBoost 83.33%, VQC 53.33%).
- **`frontend/components/ModelGraphsAndMatrix.tsx`:** Updated Next.js frontend confusion matrices with verified held-out test data.
