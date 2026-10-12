# AI-QTriage — Phase 3 Final Report: Research Paper Revision, Reviewer Response, and Submission Readiness

**Project:** AI-QTriage — A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Conference:** ICICNIS 2026  
**Article ID:** ICICNIS-945  
**Evaluation Phase:** Phase 3 — Research Paper Revision, Reviewer Response, and Camera-Ready Preparation  
**Execution Date:** October 13, 2026  
**Final Status:** **PHASE 3 COMPLETE WITH DISCLOSED NON-BLOCKING LIMITATIONS**  

---

## 1. Executive Summary

Phase 3 of the AI-QTriage research lifecycle has been executed end to end. The research manuscript for the 10th International Conference on IoT in Social, Mobile, Analytics, and Cloud (ICICNIS 2026) has been systematically revised to resolve all eleven reviewer comments (six from Reviewer 1 and five from Reviewer 2). 

Every claim, metric, formula, and architectural description in the revised paper is strictly reconciled against the verified repository implementation, live evaluation benchmarks, and the Phase 2 empirical ledger. The manuscript eliminates all colloquial first-person prose, details the explicit mathematical formulas for triage severity and pain normalization, transparently documents the 7-configuration modality ablation study, discloses the exact class-wise metrics for motion and wound detection, and frankly contextualizes the performance of both classical and quantum subsystems.

---

## 2. Reviewer Feedback Resolution Summary

All eleven reviewer comments were systematically cataloged, mapped to empirical evidence, and fully addressed in the revised text and the formal response document:

| ID | Topic | Reviewer Comment Summary | Resolution in Revised Manuscript & Response | Status |
| :--- | :--- | :--- | :--- | :--- |
| **R1.1** | Academic Tone | Colloquial tone and excessive use of first-person pronouns ("we", "our"). | Eliminated all first-person pronouns across all sections. Converted to standard objective passive/third-person academic voice. | **CLOSED** |
| **R1.2** | YOLO11 mAP | Low mAP (34.54% / 36.47%) and poor laceration detection. | Positioned YOLO11 as an initial spatial ROI localization filter rather than an autonomous diagnostic classifier. Disclosed extreme class sparsity constraint (122 laceration samples vs. 1,999 wounds $\rightarrow$ 0.0% AP50) and real photographic skin artifacts. | **CLOSED** |
| **R1.3** | U-Net Dice | U-Net Dice score of 64.18% needs contextualization and role clarification. | Contextualized Dice 64.18% (IoU 51.78%) against published chronic wound benchmarks (AZH/Medetec: 62%–72% typical). Clarified its functional role in computing feature $F_5$ (lesion area ratio) for continuous severity indexing. | **CLOSED** |
| **R1.4** | Modality Ablation | Lack of clear modality contribution ablation. | Added complete 7-configuration ablation matrix (Table III) using independent models: Vision Only (40.00%), Questionnaire Only (56.67%), Sensor Only (56.67%), Vision+Quest (53.33%), Vision+Sensor (63.33%), Quest+Sensor (90.00%), Full Multimodal (83.33%). | **CLOSED** |
| **R1.5** | Sample Size | Small test set size ($N=30$) and lack of confidence intervals. | Disclosed exact sample sizes ($N=30$, 200 synthetic vectors). Computed 95% Wilson Score confidence intervals for all configurations ($[66.44\%, 92.66\%]$ for full multimodal). Explicitly disclosed absence of clinical trial validation. | **CLOSED** |
| **R1.6** | Quantum VQC | VQC accuracy (56.67% / 53.33%) offers no quantum advantage. | Explicitly disclosed that 4-qubit VQC simulated on PennyLane `default.qubit` yields no computational advantage over XGBoost (83.33%). Highlighted majority-class inductive bias ($25/30$ MODERATE) and confirmed complete isolation from clinical decisions. | **CLOSED** |
| **R2.1** | Dataset Splits | Unclear YOLO11 dataset splits and counts. | Reconciled both the primary expanded production corpus (2,813 images: 1,935 train / 442 val / 436 test) and the localized photographic baseline (123 images: 83 train / 17 val / 23 test, 412 boxes). Documented split rationale in Section 3.1. | **CLOSED** |
| **R2.2** | Triage Category Rules | Ambiguous triage severity category assignment rules. | Documented exact continuous Severity Index formula: $S = 0.4 \times F_7 + 0.3 \times \min(F_{19}/10, 1.0) + 0.3 \times F_5$. Stated exact cutoffs: LOW ($S < 0.32$), MODERATE ($0.32 \le S < 0.58$), HIGH ($S \ge 0.58$) and clinical red-flag overrides in Section 3.7. | **CLOSED** |
| **R2.3** | Pain Normalization | Normalization of subjective integer pain scale. | Formally defined normalization equation $F_7 = \text{Pain}_{\text{raw}} / 10.0$ ($0 \le \text{Pain}_{\text{raw}} \le 10 \rightarrow 0.0 \le F_7 \le 1.0$) in Section 3.5. | **CLOSED** |
| **R2.4** | Sensor Metrics | Missing class-wise sensor classifier performance. | Added complete per-class performance Table IV ($N=1,228$ windows): Normal Activity (F1: 98.82%), Fall (F1: 95.10%), Impact (F1: 75.63%), Overall Accuracy (96.34%, Macro-F1: 89.85%) with confusion matrix rows. | **CLOSED** |
| **R2.5** | U-Net Threshold | Missing U-Net probability binarization threshold. | Formally documented the fixed operational binarization threshold $\tau = 0.50$ in Section 3.4. | **CLOSED** |

---

## 3. Verified Phase 3 Deliverables

The following authoritative Phase 3 deliverables are fully generated, verified, and committed to the repository:

1. **Revised Camera-Ready Manuscript (DOCX):**
   - Path: `AI-QTriage_REVISED_CAMERA_READY_ICICNIS.docx`
   - Formatting: IEEE / Conference standard format, Letter, 0.75 in margins, Times New Roman typography, 4 styled empirical tables (Tables I–IV).
2. **Revised Camera-Ready Manuscript (Markdown):**
   - Path: `reports/phase3/AI-QTriage_REVISED_MANUSCRIPT.md`
   - Complete verbatim text of the revised paper with GitHub Flavored Markdown and LaTeX equations.
3. **Formal Point-by-Point Rebuttal:**
   - Path: `reports/phase3/RESPONSE_TO_REVIEWERS.md`
   - Comprehensive academic rebuttal responding to each reviewer item with exact excerpts from the revised text and empirical comparison tables.
4. **Reviewer Comment Matrix:**
   - Structured CSV: `reports/phase3/reviewer_comment_matrix.csv`
   - Markdown Table: `reports/phase3/reviewer_comment_matrix.md`
5. **Canonical Evidence Ledger:**
   - Path: `reports/phase3/canonical_evidence_ledger.md`
   - Authoritative benchmark ledger documenting exact metrics, sample sizes, confidence intervals, and split procedures for all six project models.
6. **Manuscript Claim Audit:**
   - Path: `reports/phase3/manuscript_claim_audit.csv`
   - Audit trail of 16 key claims from the original manuscript, showing previous state, verified evidence, revised claim text, and verification status.
7. **Initial Inventory & Audit Record:**
   - Path: `reports/phase3/PHASE3_INITIAL_INVENTORY.md`

---

## 4. Canonical Model Benchmark Summary

The revised manuscript aligns 100% with the empirical benchmark ledger:

| Component / Subsystem | Architecture | Canonical Test Split | Primary Metric | Macro F1 / Dice | CI or Std Dev | Evidence Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lesion Detection** | YOLO11n | 436 test images (2,813 corpus) | **34.54% mAP@50** | Prec: 44.57%, Rec: 35.88% | mAP50-95: 16.74% | Independently Verified |
| **Wound Classification** | EfficientNetV2-S | 473 photographic photos | **96.62% Accuracy** | **94.34% Macro F1** | Prec: 94.26%, Rec: 95.01% | Independently Verified |
| **Wound Segmentation** | ResNet34-U-Net | 69 image-mask pairs | **64.18% Mean Dice** | **51.78% Mean IoU** | $\tau = 0.50$ threshold | Independently Verified |
| **Motion Event Classifier** | Gradient-Boosted Trees | 1,228 test windows (SisFall/UCI) | **96.34% Accuracy** | **89.85% Macro F1** | MCC: 0.9296 | Independently Verified |
| **Multimodal Triage** | XGBoost (23 features) | 30 synthetic vectors | **83.33% Accuracy** | **78.50% Macro F1** | 95% CI: [66.44%, 92.66%] | Independently Verified |
| **Quantum Classifier** | 4-Qubit VQC (PennyLane) | 30 synthetic vectors | **56.67% / 53.33% Acc** | **42.14% / 36.06% F1** | MCC: 0.1275 / 0.0173 | Independently Verified |

---

## 5. Disclosed Non-Blocking Research Limitations

To uphold the highest standards of scientific integrity, the manuscript explicitly documents the following boundaries:
1. **Academic Proof-of-Concept:** The multimodal benchmark operates on 200 synthetic feature vectors with rule-derived research labels. No autonomous clinical diagnostic authority is claimed, and the tool is strictly designated as pre-hospital decision support.
2. **Quantum Simulator Execution:** The 4-qubit VQC is executed on a classical CPU state-vector simulator (`default.qubit`). Current noisy intermediate-scale quantum circuits show no quantum advantage over classical gradient boosting for tabular triage.
3. **Unconstrained Photographic Variability:** YOLO11 performance reflects the natural difficulty of amateur smartphone wound imaging without calibrated lighting or macro lenses; its role is strictly bounded as an initial Region-of-Interest locator.

---

## 6. Final Quality Gate & Phase 3 Closure Decision

- [x] All 11 reviewer comments fully cataloged, mapped, and addressed.
- [x] All first-person pronouns removed; academic tone verified.
- [x] All mathematical equations formally stated (pain normalization, continuous severity index, binarization threshold).
- [x] Full 7-configuration modality ablation matrix incorporated with Wilson Score confidence intervals.
- [x] Complete per-class sensor motion metrics and confusion matrix rows included.
- [x] YOLO11 performance, sample sizes, and class sparsity constraints fully contextualized.
- [x] U-Net Dice score contextualized against biomedical wound segmentation literature.
- [x] VQC performance disclosed without quantum supremacy claims; clinical isolation confirmed.
- [x] Word document (`AI-QTriage_REVISED_CAMERA_READY_ICICNIS.docx`) and Markdown manuscript generated.
- [x] Claim audit (`manuscript_claim_audit.csv`) and canonical ledger created.
- [x] Response to reviewers (`RESPONSE_TO_REVIEWERS.md`) complete and submission-ready.

**FINAL VERDICT:**
# PHASE 3 COMPLETE WITH DISCLOSED NON-BLOCKING LIMITATIONS
*The AI-QTriage ICICNIS 2026 revision package is complete, verifiable, fully reconciled with repository evidence, and ready for camera-ready submission.*
