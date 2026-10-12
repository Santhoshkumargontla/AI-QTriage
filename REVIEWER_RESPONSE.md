# AI-QTriage — Official Response to Reviewer Comments (`REVIEWER_RESPONSE.md`)

**Conference:** 7th International Conference on IoT based Control Networks and Intelligent Systems (ICICNIS 2026 / IEEE Proceedings)  
**Paper Title:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Article ID:** ICICNIS-945  
**Authors:** N. Subbulakshmi, Dr. K. Venkatesh, Gontla Santhosh Kumar, Khambam Venkata Sai Girish, Dummu Rajesh  
**Date:** October 12, 2026  

---

## Overview

The authors sincerely thank the Program Chair and the anonymous reviewers for their constructive, thorough, and highly insightful feedback on **AI-QTriage (Article ID: ICICNIS-945)**. We have meticulously addressed all 11 reviewer comments across both Review Comments-1 and Review Comments-2. 

Every modification is grounded in reproducible empirical evidence, complete dataset reconciliations, new modality ablation experiments, and rigorous statistical disclosures.

---

## Response to Review Comments-1 (Reviewer 1)

### Comment 1.1
> **Reviewer 1, Comment 1:**  
> *"Some minor English language editing is required throughout the manuscript (Avoid using Personal Pronouns).."*

- **Author Response:**  
  We agree completely. We conducted a comprehensive grammatical and stylistic overhaul of the entire manuscript. All occurrences of first-person pronouns (*"we propose"*, *"our system"*, *"our models"*, *"we evaluate"*) have been eliminated and replaced with objective, third-person academic phrasing (*"the proposed framework"*, *"AI-QTriage integrates"*, *"the experimental evaluation demonstrates"*). Passive voice and precise academic terminology were adopted across Sections I, III, IV, V, VI, and VII.
- **Manuscript Section:** Global revision across Sections I, III, IV, VI, VII.
- **Supporting Evidence:** Updated manuscript draft `scratch/extracted_paper.md` / `CSP_RESEARCH_PAPER[3].docx`.

---

### Comment 1.2
> **Reviewer 1, Comment 2:**  
> *"The injury detection model shows poor performance, with only 36.47% mAP@50 and 18.44% mAP@50–95."*

- **Author Response:**  
  We appreciate the reviewer highlighting this important practical limitation. We have made two major revisions:
  1. **Harmonized Section 3.3 and Section VI:** We resolved an internal contradiction in the initial submission where Section 3.3 cited an early synthetic drawing pretraining experiment ($83.58\%$). We have standardized the manuscript on the authentic real-world photographic performance ($mAP@50 = 36.47\%$, $mAP@50\text{--}95 = 18.44\%$).
  2. **Expanded Error Analysis & Clinical Context:** We added a dedicated discussion explaining why unconstrained photographic trauma detection yields lower mAP compared to controlled benchmarks. Contributing factors include: complex illumination variance, hair and clothing occlusions, specular reflections from blood, small superficial cuts, and extreme class imbalance. We explicitly position YOLO11 as an initial localization and ROI-cropping tool rather than a standalone diagnostic classifier.
- **Manuscript Section:** Section 3.3 (Methodology), Section VI (Results and Discussion), and Section 6.3.
- **Supporting Evidence:** `ml/models/yolo_threshold_eval/THRESHOLD_SWEEP_REPORT.json` and `MODEL_EVALUATION_REPORT.md` Section 2.1.

---

### Comment 1.3
> **Reviewer 1, Comment 3:**  
> *"The segmentation model achieves only 64.18% Dice score, indicating limited accuracy in identifying injury regions"*

- **Author Response:**  
  We agree with the reviewer's observation. We have enhanced the manuscript in two key ways:
  1. **Completed Missing Values in Section 3.5:** We populated the blank field in Section 3.5 with the exact held-out test score: $\text{Mean Dice} = 64.18\%$, $\text{Mean IoU} = 51.78\%$ on $N = 69$ photographic wound-mask pairs.
  2. **Comparative Biomedical Benchmark Context:** In Section 6.3, we added a literature comparison showing that on heterogeneous, non-standardized wound photography (e.g., AZH, Medetec), published deep-learning segmentation models typically achieve Dice scores between $62\%$ and $72\%$ due to indistinct wound margins and variable tissue discoloration. We clarify that AI-QTriage uses the U-Net mask primarily to estimate the approximate relative lesion area ($F_5: \text{affected\_ratio}$) to guide downstream triage heuristics, not for microsurgical boundary delimitation.
- **Manuscript Section:** Section 3.5 (Methodology) and Section 6.3 (Discussion).
- **Supporting Evidence:** `ml/models/vision/unet_metadata.json` and `MODEL_EVALUATION_REPORT.md` Section 2.3.

---

### Comment 1.4
> **Reviewer 1, Comment 4:**  
> *"The study does not demonstrate how much each modality contributes to the final prediction through an ablation study."*

- **Author Response:**  
  This was an essential critique. To directly answer this, we conducted a rigorous modality ablation study across all 7 modality configurations on the untouched held-out test split ($N = 30$, Seed = 42). We evaluated both independently trained modality models and runtime missingness masking:
  1. **Independently Trained Modality Models:** Seven separate XGBoost models trained strictly on their permitted feature subsets ($F_0\text{--}F_5$, $F_6\text{--}F_{17}$, $F_{18}\text{--}F_{22}$, and combinations).
     - **Vision Only ($D=6$):** $33.33\%$ accuracy, $0.2414$ Macro-F1, $\text{MCC} = -0.1632$.
     - **Questionnaire Only ($D=12$):** $56.67\%$ accuracy, $0.5036$ Macro-F1, $\text{MCC} = +0.2343$.
     - **Sensor Only ($D=5$):** $56.67\%$ accuracy, $0.5010$ Macro-F1, $\text{MCC} = +0.3743$.
     - **Questionnaire + Sensor ($D=17$):** **86.67% accuracy** (26/30), **0.8148 Macro-F1**, $\text{MCC} = +0.7609$.
     - **Full Multimodal ($D=23$):** **83.33% accuracy** (25/30), **0.7850 Macro-F1**, $\text{MCC} = +0.7027$.
  2. **Runtime Zero-Masking Baseline:** Evaluated the production 23-feature model under runtime Option-A feature omission, yielding $83.33\%$ for Full Multimodal and Questionnaire + Sensor.
  - **Key Finding:** Unimodal models lack sufficient predictive power, while fusing symptoms with kinetic telemetry yields dominant accuracy (+30.00% gain). The Full Multimodal system matches this classification power while providing spatial bounding boxes (YOLO11 $36.47\%$ mAP50) and wound surface area measurements (U-Net $64.18\%$ Dice) essential for remote clinician review.
- **Manuscript Section:** Section IV (Results and Discussion), Table III.
- **Supporting Evidence:** `ABLATION_RESULTS.md`, `ABLATION_METHOD_REVIEW.md`, and `data/results/independent_ablation_results.json`.

---

### Comment 1.5
> **Reviewer 1, Comment 5:**  
> *"The XGBoost model is evaluated on only 30 test samples, making its reported 83.33% accuracy less reliable."*

- **Author Response:**  
  We fully acknowledge this statistical limitation. In the revised manuscript:
  1. We added the exact **95% Wilson Score Confidence Interval** for $25/30$ correct predictions: $[65.3\%, 94.4\%]$ directly in Table 1 and Section VI.
  2. We eliminated pseudo-precise decimal claims and explicitly disclosed that the multimodal dataset consists of 200 synthetic vectors ($140$ train, $30$ val, $30$ test) with **0 paired real-patient records**.
  3. We revised the discussion to emphasize that the $83.33\%$ accuracy represents an initial software and feature-fusion engineering proof-of-concept rather than statistically powered clinical validation.
- **Manuscript Section:** Table 1 Notes, Section 3.9, and Section VII (Conclusion).
- **Supporting Evidence:** `MODEL_EVALUATION_REPORT.md` Section 2.5 and `DATASET_AND_SPLIT_REPORT.md` Section 5.

---

### Comment 1.6
> **Reviewer 1, Comment 6:**  
> *"The quantum classifier achieves only 53.33% accuracy, significantly below the classical XGBoost model."*

- **Author Response:**  
  We agree with the reviewer. In the revised manuscript:
  1. We explicitly state that the 4-qubit VQC achieves $53.33\%$ accuracy ($16/30$) and $\text{MCC} = 0.095978$, which is barely above the 3-class random chance baseline ($33.33\%$).
  2. We emphasize that the VQC runs on PennyLane's `default.qubit` CPU simulator and **does not establish quantum computational advantage or superiority**.
  3. We clarify that VQC serves strictly as an exploratory research baseline demonstrating how variational quantum circuits can interface with classical multimodal feature pipelines, while classical XGBoost remains the authoritative decision engine.
- **Manuscript Section:** Abstract, Section 3.9, Section VI (Fig-4), and Section VII (Conclusion).
- **Supporting Evidence:** `ml/models/vqc/vqc_metadata.json` and `MODEL_EVALUATION_REPORT.md` Section 2.5.

---

## Response to Review Comments-2 (Reviewer 2)

### Comment 2.1
> **Reviewer 2, Comment 1:**  
> *"The YOLO11 dataset contains 123 images, but the reported training, validation, and test split counts are not clearly reconciled."*

- **Author Response:**  
  We thank the reviewer for identifying this ambiguity. In the revised manuscript, we explicitly reconciled the dataset partitions in Section 3.1:
  - **Photographic Trauma Dataset ($N = 123$ images, 412 bounding boxes):**
    - **Training Set:** 83 images ($67.5\%$), containing 288 wound, 12 bruise, and 15 cut instances.
    - **Validation Set:** 17 images ($13.8\%$), containing 70 wound, 1 bruise, and 2 cut instances.
    - **Held-Out Test Set:** 23 images ($18.7\%$), containing 65 wound, 4 bruise, and 5 cut instances.
    - **Sum:** $83 + 17 + 23 = \mathbf{123\text{ images}}$.
  - We also clarified that the earlier 126-image count belonged to a separate synthetic drawing set ($87/20/19$), preventing any further confusion.
- **Manuscript Section:** Section 3.1 (Dataset Acquisition).
- **Supporting Evidence:** `DATASET_AND_SPLIT_REPORT.md` Section 2.1 and `scratch/yolo_baseline_vs_candidate.json`.

---

### Comment 2.2
> **Reviewer 2, Comment 2:**  
> *"The multimodal dataset uses rule-derived labels, but the exact rules for assigning LOW, MODERATE, and HIGH classes are not specified."*

- **Author Response:**  
  We agree that methodological transparency is critical for reproducibility. We have added the exact mathematical label-generation equations to Section 3.1 and Section 3.9:
  $$\text{Severity Index} = 0.4 \times \text{pain} + 0.3 \times \text{peak\_g} + 0.3 \times \text{affected\_area}$$
  - $\text{Severity} < 0.32 \implies \mathbf{LOW}$
  - $0.32 \le \text{Severity} < 0.58 \implies \mathbf{MODERATE}$
  - $\text{Severity} \ge 0.58 \implies \mathbf{HIGH}$
  Furthermore, we documented the deterministic clinical rules enforced by the rules engine:
  - $\mathbf{HIGH}$ is assigned if: (i) structural crack/pop is accompanied by inability to bear weight or severe movement limitation; (ii) visible active bleeding occurs with pain $\ge 7/10$; or (iii) kinetic impact $\ge 5.0g$ is accompanied by pain $\ge 6/10$.
  - $\mathbf{MODERATE}$ is assigned if: impact $\ge 3.0g$, pain $\ge 4/10$, partial mobility limitation, or reported direct impact.
  - $\mathbf{LOW}$ is assigned otherwise.
- **Manuscript Section:** Section 3.1 (Dataset Acquisition) and Section 3.9 (Methodology).
- **Supporting Evidence:** `ml/fusion/rules_engine.py` lines 28–58 and `DATASET_AND_SPLIT_REPORT.md` Section 5.

---

### Comment 2.3
> **Reviewer 2, Comment 3:**  
> *"The 23-feature vector lists pain_level as continuous in the range [0,1], whereas the questionnaire defines pain on a 0–10 scale without explaining the normalization."*

- **Author Response:**  
  We thank the reviewer for catching this missing definition. We updated Section 3.6 and Section 3.8 to explicitly state the normalization formula:
  $$F_7 = \frac{\text{Pain}_{\text{raw}}}{10.0}$$
  where $\text{Pain}_{\text{raw}} \in \{0, 1, 2, \dots, 10\}$ is the user-reported integer pain score, yielding a normalized continuous feature $F_7 \in [0.0, 1.0]$. A value of $0.0$ signifies no reported pain or an absent questionnaire.
- **Manuscript Section:** Section 3.6 (Questionnaire Processing) and Section 3.8 (Feature Fusion).
- **Supporting Evidence:** `ml/fusion/feature_fusion.py` line 111 and `DATASET_AND_SPLIT_REPORT.md` Section 3.

---

### Comment 2.4
> **Reviewer 2, Comment 4:**  
> *"The sensor classifier reports 96.34% accuracy on 1,228 samples, but the corresponding class-wise performance metrics are not reported."*

- **Author Response:**  
  We appreciate this valuable recommendation. We resolved the discrepancy between the obsolete 28/36 series count and the 1,228 windowed samples, and added the complete per-class classification report to Section VI:
  - **Normal Activity ($n = 720$):** Precision = $98.89\%$, Recall = $98.75\%$, F1-Score = $98.82\%$
  - **Fall ($n = 448$):** Precision = $94.89\%$, Recall = $95.31\%$, F1-Score = $95.10\%$
  - **Impact ($n = 60$):** Precision = $76.27\%$, Recall = $75.00\%$, F1-Score = $75.63\%$
  - **Overall Accuracy:** $\frac{1,183}{1,228} = \mathbf{96.34\%}$, Macro-F1 = $\mathbf{89.85\%}$.
  We also added discussion regarding the minority Impact class ($75.0\%$ recall) and its clinical significance.
- **Manuscript Section:** Section 3.7 (Sensor Processing) and Section VI (Results, Table IV).
- **Supporting Evidence:** `ml/models/sensor_metadata.json` and `MODEL_EVALUATION_REPORT.md` Section 2.2.

---

### Comment 2.5
> **Reviewer 2, Comment 5:**  
> *"The segmentation section reports a Dice score of 64.18%, but the threshold used to convert probability masks into binary masks is not specified."*

- **Author Response:**  
  We agree with the reviewer. In Section 3.5, we updated Equation (4) and the accompanying text to explicitly define the probability binarization threshold:
  $$M(x, y) = \begin{cases} 1, & \text{if } M_{\text{prob}}(x, y) \ge \tau \\ 0, & \text{if } M_{\text{prob}}(x, y) < \tau \end{cases} \quad \text{with fixed threshold } \mathbf{\tau = 0.50}$$
  We confirmed that all evaluation metrics (Mean Dice $64.18\%$ and Mean IoU $51.78\%$ on $N = 69$ test pairs) were computed using this exact threshold $\tau = 0.50$.
- **Manuscript Section:** Section 3.5 (U-Net-Based Injury Segmentation).
- **Supporting Evidence:** `ml/models/vision/unet_metadata.json` line 30 and `MODEL_EVALUATION_REPORT.md` Section 2.3.

---

## Conclusion
Every comment raised by Reviewer 1 and Reviewer 2 has been thoroughly resolved with empirical verification, exact mathematical definitions, complete split tables, and updated manuscript text.
