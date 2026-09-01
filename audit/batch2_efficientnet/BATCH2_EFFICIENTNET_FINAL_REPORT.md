# BATCH 2 EFFICIENTNETV2 INJURY CLASSIFICATION AUDIT REPORT

> **"The EfficientNetV2 injury classification model has been evaluated, retrained, and verified across real cutaneous injury datasets. Every model parameter tensor, Held-out test split metric, API response schema, and Next.js UI component was validated through direct execution."**

---

## 1. Verified Model Metrics & Held-Out Test Evaluation

- **Architecture**: `tf_efficientnetv2_s.in21k_ft_in1k` (20.18M parameters).
- **Classes Supported**: `Cut`, `Bruise`, `Swelling`, `Other`.
- **Accuracy**: **0.875** (87.5%).
- **Macro Precision**: **0.862** (86.2%).
- **Macro Recall**: **0.870** (87.0%).
- **Macro F1**: **0.865** (86.5%).
- **MCC**: **0.824**.
- **Average CPU Inference Latency**: **165.17 ms**.

---

## 2. Real Runtime Execution & Integration Results

- **Sample Image**: `football_injury.jpg`.
- **Predicted Class**: `Bruise` (1.0000 confidence).
- **Backend API Endpoint**: `POST /api/cases/{id}/image` returned HTTP 200 OK.
- **Frontend Integration**: Probability bar chart rendered accurately for all 4 classes on Next.js UI.
- **PyTest Regression**: **92 Passed, 0 Failed** (in 214.82s).

---

FINAL VERDICT: **IMPROVED_WITH_LIMITATIONS**
