# Phase 3 Initial Inventory & Audit Report

**Date:** October 13, 2026  
**Auditor:** Senior Research Reproducibility Auditor & Scientific Writer  
**Target Conference:** 7th International Conference on IoT based Control Networks and Intelligent Systems (ICICNIS 2026 / IEEE Proceedings)  
**Paper ID:** ICICNIS-945  
**Current Synchronized Git Revision:** `fe98236595ecc2a73cde6ae4dadeaed78db8648f`  

---

## 1. Manuscript & Source Documents Inventory

| Artifact Category | File Location | Format / Details | Status |
| :--- | :--- | :--- | :--- |
| **Existing Paper Source** | `AI-QTriage_REVISED_CAMERA_READY_ICICNIS.docx` | Microsoft Word (DOCX), formatted for conference submission | **LOCATED & AUDITED** |
| **Markdown Manuscript Text** | `scratch/extracted_paper.md` | Full plaintext markdown extraction of manuscript | **LOCATED & AUDITED** |
| **Official Reviewer Feedback** | `REVIEWER_RESPONSE.md` | 11 Verbatim Reviewer Comments across Review Comments-1 and Review Comments-2 | **VERIFIED & EXTRACTED** |
| **Phase 2 Audit Deliverables** | `reports/phase2/` (16 documents) | Verified metrics, leak-free audits, baseline JSONs, ablation JSONs | **VERIFIED & CROSS-CHECKED** |
| **Canonical Metrics Artifact** | `data/results/phase2_baseline_evaluation.json` | Live re-evaluated perception and triage metrics | **VERIFIED (SHA-256 confirmed)** |
| **Canonical Ablation Artifact**| `data/results/phase2_modality_ablation.json` | 7-configuration independent modality ablation benchmark | **VERIFIED (SHA-256 confirmed)** |

---

## 2. Reviewer Comments Summary Verification

The original review letter contains **exactly 11 reviewer comments** (6 from Reviewer 1 and 5 from Reviewer 2):
* **Reviewer 1:**
  1. *Comment 1.1:* Academic tone, grammar, and avoidance of personal pronouns ("we", "our").
  2. *Comment 1.2:* YOLO11 detection performance (reporting consistency and low mAP explanation).
  3. *Comment 1.3:* U-Net segmentation Dice score (64.18%) and biomedical context.
  4. *Comment 1.4:* Missing modality ablation study across perception and tabular inputs.
  5. *Comment 1.5:* XGBoost evaluation sample size (30 cases) and confidence intervals.
  6. *Comment 1.6:* Quantum classifier performance (56.67% / 53.33%) relative to classical baselines.
* **Reviewer 2:**
  1. *Comment 2.1:* YOLO dataset inventory, image counts, and split reconciliation.
  2. *Comment 2.2:* Mathematical definitions for LOW, MODERATE, and HIGH triage categories.
  3. *Comment 2.3:* Pain score normalization formula from $[0, 10]$ to $[0, 1]$.
  4. *Comment 2.4:* Class-wise performance metrics for the sensor classifier ($N=1,228$).
  5. *Comment 2.5:* Exact binarization threshold ($\tau = 0.50$) for U-Net segmentation masks.

---

## 3. Discrepancy & Reconciliation Register

During initial audit, three critical numerical discrepancies between the submitted draft and the Phase 2 verified code base were identified:
1. **YOLO Dataset & Performance:**
   * *Submitted Text:* Reported $N=123$ images with $mAP@50 = 36.47\%$.
   * *Verified Phase 2 Dataset:* Expanded real photographic corpus is $N=2,813$ images ($1,935$ train, $442$ val, $436$ test) achieving **$mAP@50 = 34.54\%$** and $mAP@50\text{--}95 = 16.74\%$.
   * *Reconciliation Action:* Update manuscript to present the comprehensive 2,813-image benchmark as canonical, while documenting the 123-image subset as an initial localized exploration.
2. **Quantum Classifier (VQC) Accuracy:**
   * *Submitted Text:* Reported $53.33\%$ (16/30) on sequential slice $170:200$.
   * *Verified Phase 2 Split:* Evaluated on grouped leak-free test partition `test_idx` yielding **$56.67\%$ (17/30)**.
   * *Reconciliation Action:* Report 56.67% on the leak-free grouped split and transparently explain the 53.33% sequential slice variant.
3. **Sensor Architecture Lineage:**
   * *Submitted Text:* Reconciled to an **XGBoost gradient-boosted tree** operating on 7 kinematic features ($96.34\%$ accuracy on $1,228$ windows). Historical references to "1D-CNN-LSTM" must remain strictly archived.
