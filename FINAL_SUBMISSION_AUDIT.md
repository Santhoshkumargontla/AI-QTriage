# AI-QTriage — Final Pre-Submission Research & Reproducibility Audit (`FINAL_SUBMISSION_AUDIT.md`)

**Document Type:** Final Pre-Submission Audit Report & Readiness Certificate  
**Project:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Conference:** 7th International Conference on IoT based Control Networks and Intelligent Systems (ICICNIS 2026 / IEEE Proceedings)  
**Article ID:** ICICNIS-945  
**Audit Date:** October 12, 2026  
**Auditor:** Senior Machine Learning Engineer & Research Methodology Auditor  

---

## 1. Executive Pre-Submission Verdict

### **FINAL VERDICT: READY FOR SUBMISSION**

**Audit Determination:**
All 11 official reviewer comments (Review Comments-1 #1–#6, Review Comments-2 #1–#5) have been systematically resolved with verified empirical evidence. All reported numerical metrics in the revised camera-ready manuscript (`AI-QTriage_REVISED_CAMERA_READY_ICICNIS.docx`) match the underlying repository checkpoints, evaluation JSON artifacts, and test logs. Every research limitation (synthetic tabular data, lack of human clinical trials, classical quantum simulation) is explicitly disclosed.

---

## 2. Check-by-Check Audit & Classification Matrix

| Check # | Audit Item | Classification | File / Section Evidence | Audit Finding & Resolution Status |
| :-: | :--- | :---: | :--- | :--- |
| **1** | **Numerical Concordance** | **PASS** | `MODEL_EVALUATION_REPORT.md` vs DOCX Tables I–IV | Every reported metric matches saved JSON logs: YOLO11 (36.47% mAP50, 18.44% mAP50-95), EfficientNetV2 (94.34% F1, 96.62% Acc), U-Net (64.18% Dice, 51.78% IoU), Sensor (96.34% Acc), XGBoost (83.33% Acc), VQC (53.33% Acc). |
| **2** | **YOLO Precision / Recall Concordance** | **PASS** | Abstract vs. Section IV.4 | **Resolved:** Reconciled Abstract and Section IV.4 to report the verified photographic test metrics: $\text{mAP@50} = 36.47\%$, $\text{mAP@50--95} = 18.44\%$, $\text{Precision} = 51.59\%$, $\text{Recall} = 39.42\%$ on $N = 23$ held-out photographic images. |
| **3** | **Sensor Dataset & Reference Alignment** | **PASS** | Section II & Section VI [Ref 10] | **Resolved:** Corrected Ref [10] citation from UniMiB SHAR to Anguita et al. (ESANN 2013) to match the actual UCI HAR + SisFall datasets extracted in `ml/models/sensor_metadata.json`. |
| **4** | **Modality Ablation Phrasing** | **PASS** | Abstract, Section IV.2, Table III | **Resolved:** Expressed improvement as **percentage points** (+30.00 percentage points gain over unimodal baselines: 86.67% vs 56.67%). Avoided ambiguous relative percent multipliers. |
| **5** | **Dataset Sizes & Split Integrity** | **PASS** | `DATASET_AND_SPLIT_REPORT.md` | Reconciled 123-image trauma set ($83/17/23$), 473-photo classification set, 414-pair U-Net set ($276/69/69$), 6,505-window IMU set ($4,247/1,030/1,228$), and 200 synthetic vectors ($140/30/30$). |
| **6** | **Bibliographic & Reference Accuracy** | **PASS** | `build_revised_manuscript.py` L450-467 | All 15 references verified for bibliographic existence, author listings, and relevance to healthcare AI, deep learning backbones, IMU fall detection, and QML. |
| **7** | **Disclosure of Synthetic Data & Limitations** | **PASS** | Abstract, Section I, Section 3.1, Section V | Explicitly states: *"Multimodal tabular dataset relies on 200 synthetic feature vectors with rule-derived research labels. Presented strictly as an academic research prototype rather than a clinically validated medical device."* |
| **8** | **No Quantum Advantage Claims** | **PASS** | Abstract, Section 3.9, Table II, Section V | Strictly positioned as an exploratory simulation on PennyLane CPU simulator (`default.qubit`). Shows XGBoost (83.33%, MCC 0.7027) outperforms VQC (53.33%, MCC 0.0960). Zero quantum speedup claimed. |
| **9** | **DOCX Formatting & Page Budget** | **PASS** | `AI-QTriage_REVISED_CAMERA_READY_ICICNIS.docx` | Total word count is **2,848 words** with 4 styled tables, meeting standard IEEE 6–8 page double-column conference constraints. Times New Roman typography and IEEE heading styles applied. |
| **10** | **Backend Test Suite Validation** | **PASS** | `pytest backend/tests/ -q` | **235 passed out of 235 tests in 124.15s (100% pass rate)**. |
| **11** | **Git Working Tree Safety** | **PASS** | `git status` & `git diff` | Only 3 files modified in working tree (`backend/main.py`, `streamlit_app.py`, `frontend/.../ModelGraphsAndMatrix.tsx`). Zero git commits or pushes executed. |

---

## 3. Detailed Verification Deep-Dives

### 3.1 YOLO11 Photographic Baseline Reconciliation
- **Historical Conflict:** Prior drafts exhibited a conflict between a synthetic sketch baseline ($mAP@50 = 83.58\%$) and real photographic wounds ($mAP@50 = 36.47\%$).
- **Audit Finding:** The 83.58% score was derived from `data/datasets/yolo_retrain_v2` (19 sketch drawings containing 0 wound annotations). The authentic performance on unconstrained clinical photographs ($N=123$ split $83/17/23$, 412 boxes) is:
  - $\text{mAP@50} = \mathbf{36.47\%}$
  - $\text{mAP@50--95} = \mathbf{18.44\%}$
  - $\text{Precision} = \mathbf{51.59\%}$
  - $\text{Recall} = \mathbf{39.42\%}$
- **Concordance:** Abstract, Section 3.1, Section IV.4, and Table I now report these identical verified metrics.

### 3.2 Modality Ablation Improvement Metrics
- **Independent Model Evaluation (`data/results/independent_ablation_results.json`):**
  - Unimodal Baselines: Vision ($33.33\%$, F1 $0.2414$), Questionnaire ($56.67\%$, F1 $0.5036$), Sensor ($56.67\%$, F1 $0.5010$).
  - Bimodal Synergy: Questionnaire + Sensor reaches **86.67% accuracy** (Macro-F1 $0.8148$, MCC $+0.7609$).
  - Full Multimodal Fusion: Reaches **83.33% accuracy** (Macro-F1 $0.7850$, MCC $+0.7027$).
- **Improvement Metric:**
  - Questionnaire + Sensor achieves a **+30.00 percentage point** improvement over unimodal Questionnaire and Sensor baselines ($86.67\% - 56.67\% = 30.00\text{ pp}$).
  - Full Multimodal achieves a **+26.66 percentage point** improvement over unimodal baselines ($83.33\% - 56.67\% = 26.66\text{ pp}$).
- **Precision in Text:** The manuscript strictly quotes percentage point gains rather than relative percentage multipliers to prevent exaggeration.

### 3.3 Sensor Telemetry Ground Truth & Reference Alignment
- **Dataset Provenance:** `ml/models/sensor_metadata.json` confirms 6,505 windows extracted from:
  - **SisFall**: 4,505 windows.
  - **UCI HAR Smartphone Accelerometer**: 2,000 windows.
- **Reference Correction:** In `build_revised_manuscript.py`, Reference [10] was corrected from UniMiB SHAR to the authoritative UCI HAR paper:
  > *[10] D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A public domain dataset for human activity recognition using smartphones," in Proc. 21th Int. European Symp. Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), 2013, pp. 437–442.*
- **Per-Class Metrics:**
  - Normal Activity ($n=720$): F1 = $98.82\%$
  - Fall ($n=448$): F1 = $95.10\%$
  - Impact ($n=60$): Precision = $76.27\%$, Recall = $75.00\%$, F1 = $75.63\%$
  - Overall Accuracy: $96.34\%$ ($1,183 / 1,228$).

---

## 4. Software & Test Suite Status

```
pytest backend/tests/ -q
============================= 235 passed in 124.15s =============================
```

- **All 235 backend tests pass cleanly.**
- **FastAPI backend** (`http://127.0.0.1:8000`), **Streamlit frontend** (`http://127.0.0.1:8501`), and **Next.js web portal** (`http://localhost:3000`) are verified operational with synchronized metrics.

---

## 5. Pre-Submission Checklist & Final Sign-Off

- [x] All 11 reviewer comments answered in `REVIEWER_RESPONSE.md`.
- [x] Camera-ready manuscript generated: `AI-QTriage_REVISED_CAMERA_READY_ICICNIS.docx`.
- [x] Modality ablation table reflects independently trained models alongside zero-masking baselines.
- [x] Academic research prototype disclaimers present in Abstract, Section I, and Section V.
- [x] Sensor dataset reference aligned with UCI HAR (Ref [10]).
- [x] YOLO precision ($51.59\%$) and recall ($39.42\%$) consistent across Abstract and Section IV.4.
- [x] Zero Git commits or pushes made.

**Conclusion:** The manuscript and codebase are fully reconciled, scientifically grounded, reproducible, and **READY FOR SUBMISSION**.
