# Reviewer Comment Register & Resolution Matrix

**Target Conference:** 7th International Conference on IoT based Control Networks and Intelligent Systems (ICICNIS 2026 / IEEE Proceedings)  
**Paper ID:** ICICNIS-945  
**Paper Title:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Auditor:** Senior Research Reproducibility Auditor & Scientific Writer  

---

## 1. Review Comments-1 (Reviewer 1)

### Comment 1.1: Academic Writing & Personal Pronouns
* **Verbatim Comment:** *"Some minor English language editing is required throughout the manuscript (Avoid using Personal Pronouns).."*
* **Underlying Concern:** The initial manuscript contained informal first-person phrasing (*"we propose"*, *"our models"*), detracting from formal academic rigor.
* **Manuscript Revision:** Conducted a comprehensive stylistic overhaul across Sections I through VII. Replaced all personal pronouns with objective third-person and passive phrasing (*"the proposed framework"*, *"AI-QTriage evaluates"*).
* **Status:** **RESOLVED**

### Comment 1.2: YOLO11 Detection Performance
* **Verbatim Comment:** *"The injury detection model shows poor performance, with only 36.47% mAP@50 and 18.44% mAP@50–95."*
* **Underlying Concern:** Low reported detection accuracy and conflicting numbers between sections.
* **Manuscript Revision:** Harmonized all sections to report the audited 2,813-image corpus performance (**34.54% mAP@50**, 16.74% mAP@50-95) alongside the localized 123-image baseline (36.47%). Added an in-depth error analysis explaining that severe laceration sparsity (122 instances) and unconstrained open-world skin illumination depress bounding box overlap, while positioning YOLO11 as an initial localization and cropping filter rather than a standalone clinical diagnostic.
* **Status:** **RESOLVED**

### Comment 1.3: U-Net Segmentation Dice Score Context
* **Verbatim Comment:** *"The segmentation model achieves only 64.18% Dice score, indicating limited accuracy in identifying injury regions"*
* **Underlying Concern:** Modest segmentation accuracy on wound tissue without biomedical benchmarking.
* **Manuscript Revision:** Populated missing Section 3.5 values with the canonical test score (**64.18% Dice**, 51.78% IoU on $N=69$ pairs). Added clinical literature context showing that deep-learning wound segmentation models typically achieve 62%–72% on non-standardized wound photography (AZH, Medetec). Clarified that U-Net's output is used solely to estimate relative wound area ratio ($F_5$) to inform downstream heuristics, not for microsurgical margin excision.
* **Status:** **RESOLVED**

### Comment 1.4: Modality Ablation Analysis
* **Verbatim Comment:** *"The study does not demonstrate how much each modality contributes to the final prediction through an ablation study."*
* **Underlying Concern:** Absence of empirical proof that combining vision, questionnaire, and sensor inputs outperforms individual modalities.
* **Manuscript Revision:** Executed and integrated a full 7-configuration modality ablation benchmark in Section IV (Table III). Demonstrated that unimodal models achieve limited accuracy (Vision: 40.00%, Questionnaire: 56.67%, Sensor: 56.67%), while fusing Questionnaire + Sensor yields **90.00% accuracy** ($F_1 = 0.8693$) and Full Multimodal achieves **83.33% accuracy** ($F_1 = 0.7850$) with spatial lesion localization.
* **Status:** **RESOLVED**

### Comment 1.5: Small Multimodal Test Sample Size
* **Verbatim Comment:** *"The XGBoost model is evaluated on only 30 test samples, making its reported 83.33% accuracy less reliable."*
* **Underlying Concern:** Small test set size ($N=30$) and lack of statistical uncertainty estimation.
* **Manuscript Revision:** Added the exact **95% Wilson Score Confidence Interval** $[66.44\%, 92.66\%]$ for the 25/30 correct predictions directly in Table 1 and Section VI. Formally disclosed that the multimodal dataset consists of 200 synthetic vectors without paired real-patient hospital records, framing results as an engineering proof-of-concept rather than a clinically powered trial.
* **Status:** **RESOLVED**

### Comment 1.6: Quantum Classifier (VQC) Performance
* **Verbatim Comment:** *"The quantum classifier achieves only 53.33% accuracy, significantly below the classical XGBoost model."*
* **Underlying Concern:** Inferior performance of the quantum component relative to classical boosting, questioning its clinical utility.
* **Manuscript Revision:** Disclosed that the 4-qubit VQC achieves **56.67% accuracy** (17/30 on leak-free grouped split; 53.33% on sequential slice) and $\text{MCC} = 0.1275$. Explicitly clarified that VQC operates on PennyLane's `default.qubit` classical CPU simulator and **does not establish quantum advantage or supremacy**. Confirmed that VQC is isolated as an exploratory prototype (`used_in_main_decision: false`), with classical XGBoost driving patient decisions.
* **Status:** **RESOLVED**

---

## 2. Review Comments-2 (Reviewer 2)

### Comment 2.1: YOLO Dataset & Split Reconciliation
* **Verbatim Comment:** *"The YOLO11 dataset contains 123 images, but the reported training, validation, and test split counts are not clearly reconciled."*
* **Underlying Concern:** Ambiguous image counts across dataset splits.
* **Manuscript Revision:** Reconciled both the localized 123-image baseline ($83$ train, $17$ val, $23$ test) and the expanded 2,813-image production corpus ($1,935$ train, $442$ val, $436$ test) in Section 3.1, confirming zero duplicate-hash leakage across all partitions.
* **Status:** **RESOLVED**

### Comment 2.2: Definitions of LOW, MODERATE, and HIGH Categories
* **Verbatim Comment:** *"The multimodal dataset uses rule-derived labels, but the exact rules for assigning LOW, MODERATE, and HIGH classes are not specified."*
* **Underlying Concern:** Opaque triage class boundaries.
* **Manuscript Revision:** Documented the exact mathematical severity index equation in Section 3.1:
  $$\text{Severity Index} = 0.4 \times \text{pain} + 0.3 \times \text{peak\_g} + 0.3 \times \text{affected\_area}$$
  with cutoffs: $<0.32 \implies \text{LOW}$, $0.32\text{--}0.58 \implies \text{MODERATE}$, $\ge 0.58 \implies \text{HIGH}$. Fully documented the deterministic clinical red-flag escalation rules in Section 3.9.
* **Status:** **RESOLVED**

### Comment 2.3: Pain Score Normalization Formula
* **Verbatim Comment:** *"The 23-feature vector lists pain_level as continuous in the range [0,1], whereas the questionnaire defines pain on a 0–10 scale without explaining the normalization."*
* **Underlying Concern:** Unexplained mathematical transformation of patient pain scores.
* **Manuscript Revision:** Explicitly documented the normalization formula in Section 3.6 and Section 3.8:
  $$F_7 = \frac{\text{Pain}_{\text{raw}}}{10.0}$$
  where $\text{Pain}_{\text{raw}} \in \{0, 1, \dots, 10\}$ integer scale maps linearly to $F_7 \in [0.0, 1.0]$.
* **Status:** **RESOLVED**

### Comment 2.4: Sensor Classifier Class-Wise Performance
* **Verbatim Comment:** *"The sensor classifier reports 96.34% accuracy on 1,228 samples, but the corresponding class-wise performance metrics are not reported."*
* **Underlying Concern:** Aggregate accuracy concealing potential minority class failure.
* **Manuscript Revision:** Inserted the complete per-class classification report into Section VI (Table IV):
  * **Normal Activity ($n=720$):** Precision 98.89%, Recall 98.75%, F1 98.82%
  * **Fall ($n=448$):** Precision 94.89%, Recall 95.31%, F1 95.10%
  * **Impact ($n=60$):** Precision 76.27%, Recall 75.00%, F1 75.63%
  * **Overall:** 96.34% Accuracy (1,183/1,228), Macro-F1 89.85%.
* **Status:** **RESOLVED**

### Comment 2.5: U-Net Binarization Threshold
* **Verbatim Comment:** *"The segmentation section reports a Dice score of 64.18%, but the threshold used to convert probability masks into binary masks is not specified."*
* **Underlying Concern:** Unspecified post-processing decision threshold.
* **Manuscript Revision:** Updated Equation (4) in Section 3.5 to explicitly define the fixed probability threshold:
  $$M(x, y) = \begin{cases} 1, & \text{if } M_{\text{prob}}(x, y) \ge \tau \\ 0, & \text{otherwise} \end{cases} \quad \text{with fixed } \mathbf{\tau = 0.50}$$
  Confirmed that all test metrics were computed with this fixed setting without post-hoc test-set threshold tuning.
* **Status:** **RESOLVED**
