# AI-QTriage — Official Response to Reviewer Comments

**Conference:** 7th International Conference on IoT based Control Networks and Intelligent Systems (ICICNIS 2026 / IEEE Proceedings)  
**Paper Title:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Preliminary Injury Assessment and Emergency Support  
**Article ID:** ICICNIS-945  
**Authors:** N. Subbulakshmi, Dr. K. Venkatesh, Gontla Santhosh Kumar, Khambam Venkata Sai Girish, Dummu Rajesh  
**Date:** October 13, 2026  

---

## 1. Overview & General Statement

The authors express their gratitude to the Program Committee, Technical Chairs, and anonymous Reviewers of ICICNIS 2026 for their constructive, thorough, and scientifically rigorous evaluations of **AI-QTriage (Article ID: ICICNIS-945)**.

In response to the reviewers' recommendations, we have completed a comprehensive revision of the manuscript:
1. **Academic Style & Style Overhaul:** Completely eliminated informal personal pronouns (*"we"*, *"our"*) throughout the paper, adopting an objective third-person and formal passive academic style.
2. **Empirical Grounding of Computer Vision:** Harmonized all detection metrics on the expanded 2,813-image real photographic trauma corpus (**34.54% mAP@50**, 16.74% mAP@50-95), provided an in-depth error analysis of laceration sparsity (0% AP@50), contextualized U-Net segmentation (**64.18% Dice**, 51.78% IoU) within biomedical wound benchmarks, and explicitly defined the mask binarization threshold ($\tau = 0.50$).
3. **Comprehensive Modality Ablation:** Integrated a full 7-configuration modality ablation benchmark demonstrating that unimodal inputs achieve limited accuracy, while fusing Questionnaire + Sensor yields **90.00% accuracy** ($F_1 = 0.8693$) and Full Multimodal achieves **83.33% accuracy** ($F_1 = 0.7850$) alongside spatial lesion localization.
4. **Statistical Rigor & Provenance Transparency:** Disclosed the exact 95% Wilson Score Confidence Interval ($[66.44\%, 92.66\%]$) for the multimodal test set, explicitly designated multimodal fusion as a synthetic research benchmark (200 vectors, 0 paired clinical hospital records), reported full per-class sensor metrics (96.34% accuracy), and transparently positioned the 4-qubit VQC (**56.67% accuracy**) as an exploratory classical simulation on PennyLane without claiming quantum computational advantage.
5. **Algorithmic Clarity:** Provided the exact mathematical formulas for pain score normalization ($F_7 = \text{Pain}_{\text{raw}} / 10.0$), the continuous severity index calculation, and deterministic clinical red-flag triage cutoffs.

Below is our detailed, point-by-point response to every reviewer comment.

---

## 2. Point-by-Point Response to Review Comments-1 (Reviewer 1)

### Comment 1.1
> **Reviewer 1, Comment 1:**  
> *"Some minor English language editing is required throughout the manuscript (Avoid using Personal Pronouns).."*

* **Author Response:**  
  The authors fully agree with the reviewer's stylistic guidance. A systematic, global language overhaul was conducted across the entire manuscript. Every occurrence of first-person pronouns (*"we propose"*, *"our system"*, *"our models"*, *"we evaluate"*) was removed and replaced with objective, third-person academic phrasing (*"the proposed framework"*, *"AI-QTriage integrates"*, *"the experimental evaluation demonstrates"*). Formal passive voice and standardized medical-informatics terminology were adopted across Sections I through VII.
* **Manuscript Modifications:**  
  - **Abstract:** Rephrased from *"We present AI-QTriage..."* to *"An experimental research prototype, AI-QTriage, is presented..."*
  - **Section I (Introduction):** Replaced narrative first-person formulations with third-person system-level descriptions.
  - **Sections III, IV, VI, VII:** Reformulated methodology and conclusion discussions into formal passive voice.
* **Supporting Evidence:** Complete revised manuscript draft in `reports/phase3/AI-QTriage_REVISED_MANUSCRIPT.md`.

---

### Comment 1.2
> **Reviewer 1, Comment 2:**  
> *"The injury detection model shows poor performance, with only 36.47% mAP@50 and 18.44% mAP@50–95."*

* **Author Response:**  
  The authors appreciate this essential observation regarding real-world computer vision performance. We have resolved internal inconsistencies and provided an extensive empirical error analysis:
  1. **Metric Standardization & Corpus Scale:** In earlier drafts, Section 3.3 cited a localized 123-image experiment ($36.47\%$), while an obsolete synthetic drawing pretraining experiment was cited elsewhere. The revised manuscript standardizes on the canonical, audited 2,813-image real photographic trauma corpus (`data/datasets/yolo_expanded_v1`), reporting **34.54% mAP@50** and **16.74% mAP@50–95** on 436 unseen held-out test photos.
  2. **Error Analysis & Laceration Class Sparsity:** We added a detailed discussion in Section 6.3 analyzing the underlying factors behind open-world trauma detection scores:
     - Unconstrained photographic variance: complex skin illumination, shadows, specular reflections from moisture and blood, hair, and clothing occlusions.
     - Annotation imbalance: The `laceration` class has severe representation sparsity (only 122 training instances vs. 1,999 general wound boxes) and visual ambiguity with deep cuts, resulting in 0.00% test AP@50. In contrast, well-represented classes achieve strong detection: `abrasion` (61.03%), `bruise` (60.08%), and `cut` (50.67%).
  3. **Role within the System Architecture:** We explicitly clarify that YOLO11 is deployed as an initial localization and region-of-interest (ROI) cropping filter to extract bounding box coordinates ($F_0\text{--}F_3$) and bounding confidence ($F_4$), rather than serving as a definitive standalone diagnostic authority.
* **Manuscript Modifications:** Section 3.3 (Methodology), Section VI (Results, Table II), and Section 6.3 (Error Analysis & Discussion).
* **Supporting Evidence:** Checkpoint `ml/models/vision/yolo11_injury_best.pt` (SHA-256: `85788019...`), `data/results/phase2_baseline_evaluation.json`, and `reports/phase2/targeted_vision_investigation.md`.

---

### Comment 1.3
> **Reviewer 1, Comment 3:**  
> *"The segmentation model achieves only 64.18% Dice score, indicating limited accuracy in identifying injury regions"*

* **Author Response:**  
  The authors agree with the reviewer's assessment. We have addressed this critique through complete numerical reconciliation and biomedical literature contextualization:
  1. **Canonical Test Score Reporting:** Missing placeholder values in Section 3.5 were populated with the exact held-out test score: **Mean Dice = 64.18%**, **Mean IoU = 51.78%**, Precision = 69.30%, Recall = 72.33% across $N=69$ unseen patient wound-mask pairs (`data/datasets/unet_deduped_subject`). We also reconciled this with the raw unweighted slice score ($0.6441$), confirming 0.6418 as the authoritative tensor benchmark.
  2. **Biomedical Literature Context:** In Section 6.3, we added a comparative benchmark showing that on unstandardized clinical wound photography (AZH Wound Database, Medetec), published deep-learning segmentation models typically achieve Dice scores between 62% and 72% due to irregular wound borders, necrotic slough, erythema, and subtle tissue transition zones.
  3. **Clinical Role in Triage:** We clarify that AI-QTriage uses U-Net segmentation strictly to estimate the relative lesion surface area ratio ($F_5 = \text{affected\_pixels} / \text{total\_roi\_pixels}$) to inform downstream triage heuristics, not to delineate precise micro-margins for surgical intervention.
* **Manuscript Modifications:** Section 3.5 (Methodology) and Section 6.3 (Segmentation Discussion).
* **Supporting Evidence:** Checkpoint `ml/models/vision/unet_injury_best.pt` (SHA-256: `3c7f3f39...`), `data/results/phase2_baseline_evaluation.json`, and `reports/phase2/targeted_vision_investigation.md`.

---

### Comment 1.4
> **Reviewer 1, Comment 4:**  
> *"The study does not demonstrate how much each modality contributes to the final prediction through an ablation study."*

* **Author Response:**  
  This was an outstanding critique that has been completely resolved. We designed, executed, and integrated a systematic 7-configuration modality ablation benchmark across all combinations of vision, questionnaire, and sensor inputs on the identical 30-sample leak-free test partition (`data/results/phase2_modality_ablation.json`):
  
  | Configuration | Modalities Included | Active Features | Test Accuracy | Macro F1 | MCC | Wilson 95% CI |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | **Vision Only** | Vision (`F0–F5`) | 6 | **40.00%** (12/30) | 0.2886 | -0.0610 | [24.59%, 57.68%] |
  | **Questionnaire Only** | Questionnaire (`F6–F17`) | 12 | **56.67%** (17/30) | 0.5036 | 0.3340 | [39.20%, 72.62%] |
  | **Sensor Only** | Kinematic IMU (`F18–F22`) | 5 | **56.67%** (17/30) | 0.5010 | 0.3325 | [39.20%, 72.62%] |
  | **Vision + Questionnaire** | Vision + Questionnaire | 18 | **53.33%** (16/30) | 0.4732 | 0.2647 | [36.14%, 69.77%] |
  | **Vision + Sensor** | Vision + Sensor | 11 | **63.33%** (19/30) | 0.5321 | 0.4079 | [45.51%, 78.13%] |
  | **Questionnaire + Sensor** | Questionnaire + Sensor | 17 | **90.00%** (27/30) | **0.8693** | **0.8251** | [74.38%, 96.54%] |
  | **Full Multimodal** | All Modalities (`F0–F22`) | 23 | **83.33%** (25/30) | 0.7850 | 0.7027 | [66.44%, 92.66%] |

  * **Key Insights:** Unimodal inputs achieve limited classification performance ($\le 56.67\%$). Fusing Questionnaire symptoms with Sensor biomechanics yields dominant accuracy (**90.00%**), while the Full Multimodal model achieves **83.33% accuracy** while additionally supplying spatial bounding boxes and lesion surface area overlays essential for remote clinician review.
* **Manuscript Modifications:** Section IV (Results), Table III (Ablation Matrix), and Section 6.2 (Modality Contribution Discussion).
* **Supporting Evidence:** `data/results/phase2_modality_ablation.json` and `reports/phase2/modality_ablation.md`.

---

### Comment 1.5
> **Reviewer 1, Comment 5:**  
> *"The XGBoost model is evaluated on only 30 test samples, making its reported 83.33% accuracy less reliable."*

* **Author Response:**  
  The authors fully acknowledge this statistical limitation and have introduced explicit statistical safeguards and provenance disclosures:
  1. **Wilson Score Confidence Intervals:** Directly in Table 1 and Section VI, we reported the exact **95% Wilson Score Confidence Interval** for $25/30$ correct predictions: **$[66.44\%, 92.66\%]$**.
  2. **Data Provenance Disclosure:** We eliminated misleading claims of real-world patient validation. The manuscript states transparently that the multimodal dataset consists of 200 synthetically generated vectors ($140$ train, $30$ val, $30$ test) with **0 paired hospital patient records**, owing to the absence of public open-access clinical trauma cohorts combining synchronized photos, questionnaires, and IMU telemetry.
  3. **Research Prototype Framing:** We reframed the 83.33% result as a technical proof-of-concept demonstrating feature fusion feasibility rather than clinically validated efficacy.
* **Manuscript Modifications:** Section 3.9 (Methodology), Table 1 Notes, Section VI (Results), and Section VII (Limitations).
* **Supporting Evidence:** `reports/phase2/baseline_evaluation.md` and `data/results/phase2_baseline_evaluation.json`.

---

### Comment 1.6
> **Reviewer 1, Comment 6:**  
> *"The quantum classifier achieves only 53.33% accuracy, significantly below the classical XGBoost model."*

* **Author Response:**  
  The authors agree completely with the reviewer. We have revised all descriptions of the quantum component to ensure scientific realism:
  1. **Accurate Empirical Reporting:** We disclose that the 4-qubit VQC achieves **56.67% accuracy** (17/30 correct, Macro F1: 42.14%, MCC: 0.1275) on the leak-free grouped test split, and 53.33% (16/30) on sequential indexing.
  2. **Classical Simulation & Lack of Quantum Advantage:** We explicitly clarify that the VQC executes on PennyLane's `default.qubit` classical CPU simulator and **does not demonstrate quantum computational advantage or superiority**.
  3. **Architectural Role:** We explain that VQC's performance is constrained by Hilbert space dimensionality limits (compressing 23 features into 4 qubits via PCA). It is included strictly as an exploratory research component demonstrating hybrid classical-quantum data interfaces, while classical XGBoost (83.33%) and deterministic rules drive all triage decisions (`used_in_main_decision: false`).
* **Manuscript Modifications:** Abstract, Section 3.9, Section VI (Quantum Discussion & Fig-4), and Section VII (Conclusion).
* **Supporting Evidence:** Checkpoint `ml/models/vqc/vqc_weights.npz` (SHA-256: `c818299b...`), `data/results/phase2_baseline_evaluation.json`, and `backend/main.py`.

---

## 3. Point-by-Point Response to Review Comments-2 (Reviewer 2)

### Comment 2.1
> **Reviewer 2, Comment 1:**  
> *"The YOLO11 dataset contains 123 images, but the reported training, validation, and test split counts are not clearly reconciled."*

* **Author Response:**  
  We thank the reviewer for highlighting this data reporting ambiguity. In Section 3.1, we have explicitly reconciled both the localized dataset baseline and the expanded production corpus:
  1. **Localized Baseline Dataset ($N = 123$ images, 412 bounding boxes):**
     - **Training Set:** 83 images ($67.5\%$), containing 288 wound, 12 bruise, 15 cut boxes.
     - **Validation Set:** 17 images ($13.8\%$), containing 70 wound, 1 bruise, 2 cut boxes.
     - **Held-Out Test Set:** 23 images ($18.7\%$), containing 65 wound, 4 bruise, 5 cut boxes.
     - **Total:** $83 + 17 + 23 = \mathbf{123\text{ images}}$.
  2. **Expanded Production Corpus ($N = 2,813$ images):**
     - **Training Set:** 1,935 images.
     - **Validation Set:** 442 images.
     - **Held-Out Test Set:** 436 images.
     - **Total:** $1,935 + 442 + 436 = \mathbf{2,813\text{ images}}$ (0 duplicate hashes across splits).
  Both split breakdowns are now presented in Section 3.1 with complete partition integrity.
* **Manuscript Modifications:** Section 3.1 (Dataset Acquisition).
* **Supporting Evidence:** `reports/phase2/dataset_inventory.csv` and `reports/phase2/leakage_audit.json`.

---

### Comment 2.2
> **Reviewer 2, Comment 2:**  
> *"The multimodal dataset uses rule-derived labels, but the exact rules for assigning LOW, MODERATE, and HIGH classes are not specified."*

* **Author Response:**  
  We agree that full algorithmic transparency is essential for reproducibility. We have added the exact mathematical formula and deterministic escalation criteria to Section 3.1 and Section 3.9:
  1. **Continuous Severity Index Formula:**
     $$\text{Severity Index} = 0.4 \times F_7 (\text{pain}) + 0.3 \times F_{18} (\text{peak\_g}) + 0.3 \times F_5 (\text{affected\_area})$$
     - $\text{Severity} < 0.32 \implies \mathbf{LOW}$
     - $0.32 \le \text{Severity} < 0.58 \implies \mathbf{MODERATE}$
     - $\text{Severity} \ge 0.58 \implies \mathbf{HIGH}$
  2. **Deterministic Clinical Red-Flag Escalation Rules:**
     - $\mathbf{HIGH}$ is assigned if: (i) structural crack/pop is accompanied by inability to bear weight or severe movement limitation; (ii) visible active bleeding occurs with pain $\ge 7/10$; or (iii) kinetic impact $\ge 5.0g$ is accompanied by pain $\ge 6/10$.
     - $\mathbf{MODERATE}$ is assigned if: kinetic impact $\ge 3.0g$, pain $\ge 4/10$, partial mobility limitation, or direct blunt trauma.
     - $\mathbf{LOW}$ is assigned otherwise.
* **Manuscript Modifications:** Section 3.1 (Dataset Acquisition) and Section 3.9 (Decision Logic & Rules Engine).
* **Supporting Evidence:** Implementation in `ml/fusion/rules_engine.py` (lines 28–58) and `reports/phase2/FINAL_PHASE2_AUDIT.md`.

---

### Comment 2.3
> **Reviewer 2, Comment 3:**  
> *"The 23-feature vector lists pain_level as continuous in the range [0,1], whereas the questionnaire defines pain on a 0–10 scale without explaining the normalization."*

* **Author Response:**  
  We thank the reviewer for catching this omission. We updated Section 3.6 and Section 3.8 to explicitly define the normalization transformation:
  $$F_7 = \frac{\text{Pain}_{\text{raw}}}{10.0}$$
  where $\text{Pain}_{\text{raw}} \in \{0, 1, 2, \dots, 10\}$ is the user-reported integer pain score from the questionnaire interface, mapping linearly to a continuous feature $F_7 \in [0.0, 1.0]$. A value of $0.0$ indicates zero pain or an absent questionnaire input.
* **Manuscript Modifications:** Section 3.6 (Questionnaire Processing) and Section 3.8 (Feature Fusion Table).
* **Supporting Evidence:** Implementation in `ml/fusion/feature_fusion.py` (line 111).

---

### Comment 2.4
> **Reviewer 2, Comment 4:**  
> *"The sensor classifier reports 96.34% accuracy on 1,228 samples, but the corresponding class-wise performance metrics are not reported."*

* **Author Response:**  
  The authors appreciate this valuable suggestion. We resolved earlier sample-count ambiguities and incorporated the complete per-class classification report into Section VI (Table IV):
  
  | Class | Support ($n$) | Precision (%) | Recall (%) | F1-Score (%) |
  | :--- | :--- | :--- | :--- | :--- |
  | **Normal Activity** | 720 | 98.89% | 98.75% | 98.82% |
  | **Fall** | 448 | 94.89% | 95.31% | 95.10% |
  | **Impact** | 60 | 76.27% | 75.00% | 75.63% |
  | **Overall / Macro** | **1,228** | **90.02%** | **89.69%** | **89.85%** |

  * **Overall Accuracy:** $1,183 / 1,228 = \mathbf{96.34\%}$, $\text{MCC} = \mathbf{0.9296}$.
  We also added discussion regarding the minority Impact class ($75.0\%$ recall, 60 test windows) and why sudden stumble-recoveries can trigger false positives.
* **Manuscript Modifications:** Section 3.7 (Sensor Processing) and Section VI (Results, Table IV).
* **Supporting Evidence:** Checkpoint `ml/models/sensor_motion_best.json` (SHA-256: `4980d215...`) and `reports/phase2/baseline_evaluation.md`.

---

### Comment 2.5
> **Reviewer 2, Comment 5:**  
> *"The segmentation section reports a Dice score of 64.18%, but the threshold used to convert probability masks into binary masks is not specified."*

* **Author Response:**  
  We agree with the reviewer. In Section 3.5, we updated Equation (4) and the accompanying text to explicitly state the fixed binarization threshold:
  $$M(x, y) = \begin{cases} 1, & \text{if } M_{\text{prob}}(x, y) \ge \tau \\ 0, & \text{if } M_{\text{prob}}(x, y) < \tau \end{cases} \quad \text{with fixed threshold } \mathbf{\tau = 0.50}$$
  We confirm that all reported test evaluation metrics (Mean Dice $64.18\%$ and Mean IoU $51.78\%$ on $N=69$ patient test pairs) were computed using this fixed threshold without post-hoc tuning on the test partition.
* **Manuscript Modifications:** Section 3.5 (U-Net-Based Injury Segmentation).
* **Supporting Evidence:** `ml/models/vision/unet_metadata.json` (line 30) and `reports/phase2/targeted_vision_investigation.md`.

---

## 4. Conclusion
Every comment raised by Reviewer 1 and Reviewer 2 has been thoroughly resolved with empirical verification, exact mathematical definitions, complete split tables, and revised manuscript text. The revised paper represents an honest, reproducible, and scientifically sound contribution.
