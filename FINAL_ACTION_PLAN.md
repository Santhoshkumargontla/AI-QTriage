# AI-QTriage — Final Action Plan & Submission Runbook (`FINAL_ACTION_PLAN.md`)

**Document Type:** Final Action Plan & Conference Submission Runbook  
**Project:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Conference:** 7th International Conference on IoT based Control Networks and Intelligent Systems (ICICNIS 2026 / IEEE Proceedings)  
**Article ID:** ICICNIS-945  
**Date:** October 12, 2026  
**Auditor:** Senior Machine Learning Engineer  

---

## 1. Prioritized Action Checklist

### Phase 1 Technical Corrections: **100% COMPLETED**
- [x] **YOLO11 Metric Reconciliation**: Standardized on real photographic wound benchmark ($36.47\%$ mAP50, $18.44\%$ mAP50-95, $51.59\%$ Precision, $39.42\%$ Recall on 23 photos).
- [x] **U-Net Metrics & Threshold**: Documented $64.18\%$ Mean Dice, $51.78\%$ IoU, and $\tau = 0.50$ binarization threshold.
- [x] **Sensor Classifier Ledgers**: Documented $96.34\%$ overall accuracy on 1,228 windows with class-wise table (Normal 98.8%, Fall 95.1%, Impact 75.6%).
- [x] **Bibliographic Correction**: Aligned Reference [10] with UCI HAR (Anguita et al., 2013).
- [x] **Citation Integrity**: Ensured all 15 references are cited in body text.
- [x] **Independent Modality Ablation**: Trained 7 separate models strictly on permitted feature subsets; serialized results to `data/results/independent_ablation_results.json`.
- [x] **Camera-Ready DOCX Compiled**: Built `AI-QTriage_REVISED_CAMERA_READY_ICICNIS.docx` (2,848 words, 4 IEEE styled tables).
- [x] **Backend & Frontend Test Suite**: 235 / 235 backend pytest tests pass cleanly; UI values synchronized.

---

## 2. Immediate Author / Supervisor Actions (Administrative Runbook)

### Action 1: Upload the Revised Camera-Ready DOCX
- **File to Upload:** `AI-QTriage_REVISED_CAMERA_READY_ICICNIS.docx`
- **Location:** Open Microsoft CMT / EasyChair / Conference Portal for Paper ID **ICICNIS-945**.
- **Action:** Replace previous submission with `AI-QTriage_REVISED_CAMERA_READY_ICICNIS.docx`.

### Action 2: Submit the Reviewer Rebuttal Document
- **File to Submit:** `REVIEWER_RESPONSE.md` (or convert to PDF via Word/Pandoc).
- **Action:** Upload as *"Response to Reviewers / Author Rebuttal"*. Point reviewers to Table I (Feature Schema), Table II (XGBoost vs VQC), Table III (Modality Ablation), and Table IV (Sensor Class-Wise Metrics).

### Action 3: Administrative Conference Extension Inquiry
- **Context:** The acceptance letter or initial notification may have noted October 10, 2026 as an initial camera-ready deadline.
- **Action for Corresponding Author:** Send a confirmation email to the conference secretariat:
  > **To:** icicnis.conference@gmail.com (or official contact)  
  > **Subject:** Paper ID ICICNIS-945 Camera-Ready Submission Confirmation  
  > **Body:**  
  > *"Dear ICICNIS 2026 Organizing Committee,*  
  > *We have completed all technical revisions addressing Reviewer Comments 1 and 2 for Paper ID ICICNIS-945 (AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support).*  
  > *We are uploading our revised camera-ready manuscript (.docx) and detailed point-by-point response document. Kindly confirm receipt and acceptance of the revised files for IEEE proceedings publication.*  
  > *Thank you.*  
  > *Best regards,*  
  > *Gontla Santhosh Kumar / Dr. K. Venkatesh (Kalasalingam Academy of Research and Education)"*

---

## 3. Git Repository Final State
- **Branch:** `main`
- **Local Modified Files:**
  - `backend/main.py` (FastAPI `Request` import)
  - `streamlit_app.py` (canonical benchmark display)
  - `frontend/components/ModelGraphsAndMatrix.tsx` (canonical confusion matrix)
- **Status:** Untracked audit markdown deliverables preserved for institutional review. Zero unapproved commits or pushes executed.
