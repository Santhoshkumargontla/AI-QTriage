# AI-QTriage — Remaining Limitations & Research Risk Register (`REMAINING_LIMITATIONS.md`)

**Document Type:** Formal Research Risk & Model Limitations Register  
**Project:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Conference:** 7th International Conference on IoT based Control Networks and Intelligent Systems (ICICNIS 2026)  
**Article ID:** ICICNIS-945  
**Date:** October 12, 2026  
**Auditor:** Senior Machine Learning Engineer & Research Reproducibility Reviewer  

---

## Executive Status Classification

| # | Limitation Title | Subsystem / Model | Classification | Current Impact on Paper & System | Required Action / Path to Resolution |
| :-: | :--- | :--- | :---: | :--- | :--- |
| **1** | **Synthetic Multimodal Tabular Dataset** | Multimodal Triage (XGBoost & VQC) | **PARTIALLY RESOLVED** | 200 synthetic vectors ($140/30/30$); 0 real patient trauma cohorts. System is strictly an engineering prototype. | Fully disclosed in Abstract, Section 3.1, Section V, and Section VI. Requires hospital IRB real trauma patient trial for clinical deployment. |
| **2** | **Rule-Derived Severity Ground Truth** | Triage Labels | **PARTIALLY RESOLVED** | Severity labels derive from $0.4\times\text{pain} + 0.3\times\text{peak\_g} + 0.3\times\text{affected\_area}$. Model learns formula recovery. | Documented exact mathematical formulas in Section 3.1 and Section 3.9. Cannot replace with fabricated human labels. |
| **3** | **Small Held-Out Test Set ($N=30$)** | Multimodal Triage | **RESOLVED IN DOCS** | Small sample size creates broad 95% Wilson CIs ($[70.3\%, 94.7\%]$ for Quest+Sensor; $[66.4\%, 92.7\%]$ for Full Model). | Computed exact Wilson CIs, executed 5-fold CV on train split ($N=140$), and eliminated statistical dominance claims. |
| **4** | **YOLO11 Real Photographic Wound mAP** | Vision Detection (`yolo11_injury_best.pt`) | **RESOLVED IN DOCS** | Real wound mAP@50 is 36.47% (18.44% mAP50-95). Laceration AP is 0.00% due to lighting/hair occlusion and small sample count ($N=23$). | Reconciled real wound benchmark vs. 83.58% drawing set. Added error analysis in Section 3.2. Retraining requires expanded clinical photography. |
| **5** | **U-Net Lesion Dice Score Ceiling** | Lesion Segmentation (`unet_injury_best.pt`) | **RESOLVED IN DOCS** | Mean Dice is 64.18% (51.78% IoU) across 69 test pairs at binarization threshold $\tau = 0.50$. | Verified Dice calculation; contextualized against clinical benchmark literature ($62\%\text{--}72\%$ typical on chronic ulcer photos). |
| **6** | **Sensor Impact Minority Class Imbalance** | Motion Classifier (`sensor_motion_best.json`) | **RESOLVED IN DOCS** | Overall accuracy is 96.34% ($1,183/1,228$), but Impact recall is 75.00% (F1 75.63%) due to class imbalance (60 impact vs 720 normal, 448 falls). | Added class-wise breakdown table (Table IV) in manuscript and model report. External real-world fall validation needed before commercial wear. |
| **7** | **PennyLane VQC Quantum Positioning** | Quantum Simulation (`vqc_weights.npz`) | **RESOLVED IN DOCS** | 4-qubit VQC achieves 53.33% accuracy (MCC 0.0960) on PennyLane CPU simulator. Significantly underperforms XGBoost (83.33%). | Strictly positioned as an exploratory simulation baseline; explicitly disclaimed quantum speedup or quantum advantage. |
| **8** | **Emergency Dispatch Carrier Reliance** | Emergency Support Workflow (`backend/services/sos_service.py`) | **RESOLVED IN CODE** | Production SMS dispatch requires live Twilio carrier credentials. Default setup operates in simulated demo mode. | Verified local simulated dispatch mode; safety guardrails prevent accidental emergency service dialing during testing. |

---

## Detailed Audit & Resolution Breakdown

### 1. Synthetic Multimodal Dataset & Rule-Derived Labels
- **Evidence:** `ml/training/train_xgboost.py` lines 19–60 and `data/results/canonical_held_out_evaluation.json`.
- **Finding:** Target labels directly depend on linear combinations of input features ($pain\_level \times 0.4 + peak\_g \times 0.3 + affected\_area \times 0.3$).
- **Resolution:**
  - Fully disclosed in all reports (`PHASE1_AUDIT.md`, `ABLATION_METHOD_REVIEW.md`, `MODEL_EVALUATION_REPORT.md`, `REVIEWER_RESPONSE.md`).
  - The manuscript prominently features the disclaimer: *"SYNTHETIC HELD-OUT EVALUATION — RESEARCH PROTOTYPE. Has not undergone clinical validation on real-patient trauma cohorts."*
  - **Status:** **PARTIALLY RESOLVED** (methodologically disclosed; authentic resolution requires real hospital partner datasets).

### 2. Modality Ablation Claim Bounds (Quest + Sensor vs Full Model)
- **Evidence:** `data/results/independent_ablation_results.json` and `ABLATION_RESULTS.md`.
- **Finding:** Questionnaire + Sensor achieves 86.67% accuracy (MCC +0.7609) vs. Full Multimodal 83.33% accuracy (MCC +0.7027).
- **Resolution:**
  - Corrected claims: The paper no longer claims that the Full Model is superior based on discrete severity accuracy.
  - The clinical role of Vision is explicitly documented: Vision provides spatial lesion localization ($mAP@50 = 36.47\%$), wound area measurement ($Dice = 64.18\%$), and tele-triage visual confirmation for emergency physicians.
  - **Status:** **RESOLVED**.

### 3. Sensor Motion Classifier Class Breakdown
- **Evidence:** `ml/models/sensor_metadata.json` and `MODEL_EVALUATION_REPORT.md`.
- **Finding:** Impact class has lower sensitivity (75.00% recall, 45/60 true positives) due to severe class imbalance.
- **Resolution:**
  - Table IV in `AI-QTriage_REVISED_CAMERA_READY_ICICNIS.docx` reports per-class metrics: Normal Activity (98.8% F1), Fall (95.1% F1), and Impact (75.6% F1).
  - **Status:** **RESOLVED**.

### 4. Application UI Alignment & Safety
- **Evidence:** `backend/main.py`, `streamlit_app.py`, and `frontend/components/ModelGraphsAndMatrix.tsx`.
- **Finding:** UI previously contained stale placeholder benchmark values (e.g., YOLO 91.4%).
- **Resolution:**
  - Updated Streamlit `render_model_benchmarks()` with verified test numbers (YOLO 36.47% mAP50, EfficientNet 96.62%, U-Net 64.18% Dice, Sensor 96.34%, XGBoost 83.33%, VQC 53.33%).
  - Updated Next.js confusion matrices with canonical test outputs.
  - Backend pytest suite passes 100% (235/235 tests in 124.15s).
  - **Status:** **RESOLVED**.

---

## Summary of Unresolved Items & Recommended Future Work

1. **Unresolved Item 1: Real-Patient Trauma Dataset**
   - **Reason:** Requires institutional clinical ethics approval and clinical partner hospital integration.
   - **Action:** Clearly state as future work in Section V of the paper.
2. **Unresolved Item 2: Quantum Hardware Execution**
   - **Reason:** Physical quantum hardware access (e.g. IBM Quantum / Rigetti) was outside the scope of the simulation study.
   - **Action:** Retain PennyLane CPU simulation baseline description without quantum advantage claims.
