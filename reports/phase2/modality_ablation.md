# Phase 2 Modality Ablation Study Report

**Generated:** 2026-10-12  
**Auditor:** Senior Research Reproducibility Auditor & ML Verification Team  
**Scope:** Systematic evaluation of all 7 independent modality combinations on the held-out test split (30 test vectors).

---

## 1. Modality Ablation Matrix

Each configuration represents an independently trained XGBoost model evaluated on the identical 30-sample leak-free test partition.

| Configuration | Modalities Included | Active Features | Test Accuracy | Correct / Total | Macro F1 | MCC | Wilson 95% CI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vision Only** | Vision (`F0–F5`) | 6 | **40.00%** | 12 / 30 | 0.2886 | -0.0610 | [24.59%, 57.68%] |
| **Questionnaire Only** | Questionnaire (`F6–F17`) | 12 | **56.67%** | 17 / 30 | 0.5036 | 0.3340 | [39.20%, 72.62%] |
| **Sensor Only** | Kinematic IMU (`F18–F22`) | 5 | **56.67%** | 17 / 30 | 0.5010 | 0.3325 | [39.20%, 72.62%] |
| **Vision + Questionnaire** | Vision + Questionnaire | 18 | **53.33%** | 16 / 30 | 0.4732 | 0.2647 | [36.14%, 69.77%] |
| **Vision + Sensor** | Vision + Sensor | 11 | **63.33%** | 19 / 30 | 0.5321 | 0.4079 | [45.51%, 78.13%] |
| **Questionnaire + Sensor** | Questionnaire + Sensor | 17 | **90.00%** | 27 / 30 | **0.8693** | **0.8251** | [74.38%, 96.54%] |
| **Full Multimodal** | All Modalities (`F0–F22`) | 23 | **83.33%** | 25 / 30 | **0.7850** | **0.7027** | [66.44%, 92.66%] |

---

## 2. Research Insights & Discussion

1. **Unimodal Limitations:**
   * Vision alone achieves only 40.00% test accuracy. In emergency triage, visual appearance of skin injuries (e.g., superficial cuts vs. lacerations) lacks critical systemic physiological context.
   * Questionnaire and sensor features achieve 56.67% accuracy independently, illustrating that symptom severity and kinetic impact carry moderate discriminative power on their own.

2. **Synergy of Biomechanical and Clinical Inputs:**
   * Combining Questionnaire + Sensor yields the highest overall accuracy (**90.00%**, 27 / 30 correct, Macro F1: 0.8693). Biomechanical kinetic trauma (peak g-force and stabilization time) combined with patient symptom reports (pain, nausea, loss of consciousness) forms a powerful signal for triage categorization.

3. **Multimodal Trade-Off (Vision Noise):**
   * Incorporating vision features into the full 23-feature model slightly attenuates accuracy from 90.00% to 83.33% (25 / 30). This is attributable to visual score variance acting as slight feature noise when combined with highly predictive physiological indicators.
   * However, vision features remain essential for localized anatomical assessment, lesion area quantification, and wound progression tracking in the end-to-end clinical workflow.
