# AI-QTriage — Modality Ablation Methodology Review (`ABLATION_METHOD_REVIEW.md`)

**Document Type:** Research Methodology & Scientific Validity Audit  
**Project:** AI-QTriage: A Hybrid AI–Quantum Multimodal Framework for Explainable Injury Assessment and Emergency Support  
**Conference:** 7th International Conference on IoT based Control Networks and Intelligent Systems (ICICNIS 2026)  
**Article ID:** ICICNIS-945  
**Audit Date:** October 12, 2026  
**Auditor:** Senior Machine Learning Engineer & Research Reproducibility Reviewer  

---

## 1. Executive Summary & Audit Findings

### **Methodological Finding: Feature-Masking vs. Independent Modality Training**
The existing modality ablation experiment implemented in `scratch/run_modality_ablation.py` was evaluated using **zero-filled feature masking on a single pre-trained 23-feature XGBoost model (`ml/models/xgboost_best.json`)**.

While feature-masking evaluates the single model's runtime robustness to missing inputs under Option-A zero-filling, **it does not constitute an authentic modality contribution ablation study**. An authentic modality ablation must measure whether an architecture trained *specifically* on the permitted subset of modalities can learn predictive representations, versus evaluating an overfitted full-feature tree ensemble confronted with artificially zeroed input vectors.

---

## 2. Forensic Audit Questions & Evidence

### Question 1: Does the current ablation use a single full-feature model with masked test features?
**Evidence:**
- In `scratch/run_modality_ablation.py` (lines 16–17 & 46–48):
  ```python
  xgb = XGBClassifier()
  xgb.load_model(XGB_CANONICAL)  # ml/models/xgboost_best.json (trained on all 23 features F0-F22)
  ...
  for name, mods in configs:
      X_masked = mask_features(X_test, mods)
      preds = xgb.predict(X_masked)
```
- **Conclusion:** **YES**. A single 23-feature model was loaded and evaluated across all 7 configurations by simply replacing excluded feature columns with zeros.

---

### Question 2: Are zero-filled features within the normal feature distributions or out-of-distribution (OOD)?
**Evidence:**
- In `ml/training/train_xgboost.py` (lines 44–60), the synthetic data generator applies **modality dropout during training**:
  - 70% of synthetic samples have all modalities present ($has\_vision = has\_quest = has\_sensor = True$).
  - 30% of synthetic samples randomly drop 1 or 2 modalities (setting their corresponding columns to $0.0$).
- Because modality dropout was present in 30% of the training distribution, zero inputs are technically seen during training.
- **However**, zero-filling creates a significant structural distortion:
  - If Vision is masked, `vision_present = 0.0`, `prob_cut = 0.0`, `affected_area = 0.0`.
  - In a decision tree trained on all 23 features, splits placed on `pain_score` or `peak_g` still expect those features to exist. When evaluating "Vision Only", `pain_score = 0.0` and `peak_g = 0.0` are passed to trees where the upper splits may route low severity based on zero pain, severely degrading performance to 46.67% accuracy and negative MCC ($-0.1020$).
- **Conclusion:** Zero-filling tests **missing-modality fallback performance of the full model**, not the true discriminatory capacity of the modality itself.

---

### Question 3: Does the experiment measure modality contribution or model sensitivity to masking?
**Conclusion:** It measures **sensitivity to input masking**. When evaluating a unimodal model (e.g. Vision Only), an independently trained vision model learns the direct relationship between visual probabilities and injury tier without relying on non-existent symptom and sensor splits. Testing a 23-feature tree with 17 zeros conflates feature absence with zero-valued symptoms.

---

### Question 4: Are labels generated from features that belong to specific modalities (Target Leakage)?
**Evidence:**
- In `ml/training/train_xgboost.py` (lines 32–42):
  ```python
  pain_score = float(np.random.randint(1, 10) / 10.0)      # Questionnaire (F7)
  peak_g = float(np.random.uniform(1.0, 15.0) / 15.0)     # Sensor (F19)
  affected_area = float(np.random.uniform(0.05, 0.40))     # Vision (F5)

  severity_score = (pain_score * 0.4) + (peak_g * 0.3) + (affected_area * 0.3)
  if severity_score < 0.32:
      label = 0  # LOW
  elif severity_score < 0.58:
      label = 1  # MODERATE
  else:
      label = 2  # HIGH
  ```
- **Target Construction Analysis:**
  - Questionnaire ($pain\_score$) contributes **40%** to the severity label.
  - Sensor ($peak\_g$) contributes **30%** to the severity label.
  - Vision ($affected\_area$) contributes **30%** to the severity label.
- **Impact:**
  - The ground truth is a direct linear combination of features from all three modalities.
  - Questionnaire has the largest single weight (0.4), explaining why Questionnaire alone (56.67%) and Questionnaire + Sensor (83.33%) achieve high accuracy.
  - This is a synthetic rule-recovery task. The model is learning to approximate the label assignment formula under noisy and missing feature observations.

---

### Question 5: Do the 200 synthetic samples support clinical claims?
**Conclusion:** **NO**. The 200 synthetic vectors are entirely simulated. The experiment provides proof-of-concept algorithmic verification of multimodal integration, but **cannot and must not claim clinical triage validity**.

---

## 3. Methodological Correction Plan

To establish a scientifically sound modality ablation study:
1. **Implement Independent Model Training (`scratch/run_independent_modality_ablation.py`)**:
   - Train 7 separate XGBoost classifiers independently on only their permitted feature columns:
     - Config 1: Vision Only ($F_0\text{--}F_5$, 6 features)
     - Config 2: Questionnaire Only ($F_6\text{--}F_{17}$, 12 features)
     - Config 3: Sensor Only ($F_{18}\text{--}F_{22}$, 5 features)
     - Config 4: Vision + Questionnaire ($F_0\text{--}F_{17}$, 18 features)
     - Config 5: Vision + Sensor ($F_0\text{--}F_5, F_{18}\text{--}F_{22}$, 11 features)
     - Config 6: Questionnaire + Sensor ($F_6\text{--}F_{22}$, 17 features)
     - Config 7: Full Multimodal ($F_0\text{--}F_{22}$, 23 features)
2. **Partitioning Discipline**:
   - Exactly the same $N = 200$ dataset (seed=42).
   - Training ($N = 140$, [:140]), Validation ($N = 30$, [140:170]), Held-out Test ($N = 30$, [170:200]).
   - Hyperparameters tuned on train/val only. Test set evaluated strictly at inference.
3. **Save Results Separately**:
   - Serialize to `data/results/independent_ablation_results.json`.
   - Preserve historical zero-mask results in `data/results/canonical_ablation_results.json`.
   - Document both protocols transparently in `ABLATION_RESULTS.md`.
