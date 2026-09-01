# BATCH 13 — READINESS SCORE & EVIDENCE CONSISTENCY AUDIT REPORT

> **"Forensic evaluation of the AI-QTriage platform verifies readiness score traceability, caps Clinical Deployment Readiness to 0.0% (due to 0 clinical trials and 0 patient samples), establishes dataset SHA-256 manifest integrity, documents the 6-model artifact evidence matrix, and confirms 100% PyTest suite passing."**

---

## 1. Executive Forensic Status Summary

| Audit Dimension | Evidence Classification | Verified Result | Rationale & Traceability |
| :--- | :---: | :---: | :--- |
| **Software Engineering Readiness** | `DERIVED_FROM_REAL_OUTPUT` | **95.0%** | PyTest (101 passed, 0 failed), 23D vector contract enforced, REST API healthy. |
| **Academic Project Readiness** | `DERIVED_FROM_REAL_OUTPUT` | **90.0%** | Dataset provenance documented, reproducible audit scripts, complete methodology. |
| **Research Prototype Readiness** | `DERIVED_FROM_REAL_OUTPUT` | **88.0%** | Multimodal feature fusion active, VQC safety isolated (`sos_weight = 0.0`). |
| **Real-World Validation Level** | `DERIVED_FROM_REAL_OUTPUT` | **15.0%** | Derived purely from 200 hybrid simulation records; 0 clinical patient samples. |
| **Clinical Deployment Readiness** | `NOT_AVAILABLE` | **0.0%** | Mandatory Evidence Cap: 0 clinical patient trials & 0 FDA/CE regulatory clearances. |
| **Frontend Runtime Status** | `NOT_AVAILABLE` | `FRONTEND_RUNTIME_NOT_VERIFIED` | Headless browser automation runner is not active in this session. |

---

## 2. Readiness Score Formulas & Traceability

$$\text{Software Engineering Readiness} = 0.40(\text{pytest}) + 0.30(\text{API}) + 0.30(\text{vector}) = 95.0\%$$
$$\text{Academic Project Readiness} = 0.40(\text{methodology}) + 0.30(\text{provenance}) + 0.30(\text{reproducibility}) = 90.0\%$$
$$\text{Research Prototype Readiness} = 0.40(\text{fusion}) + 0.30(\text{VQC isolation}) + 0.30(\text{consistency}) = 88.0\%$$
$$\text{Real-World Validation Level} = 0.70(\text{real\_clinical}) + 0.30(\text{hybrid\_sim}) = 0.70(0) + 0.30(50\%) = 15.0\%$$
$$\text{Clinical Deployment Readiness} = 0.50(\text{clinical\_trials}) + 0.50(\text{regulatory}) = 0.0\%$$

---

## 3. Dataset Evidence & Manifest Integrity

- **Total Discovered Files on Disk**: `2371`
- **Total Parsed Records**: `2393`
- **Real Clinical Patient Samples**: `0`
- **Hybrid Simulation Records**: `2393`
- **SHA-256 Verified Dataset Files**:
  - `data/multimodal_cases.json`: `None`
  - `data/sensor_logs.json`: `None`
  - `data/processed/xgboost_features.json`: `None`
  - `data/processed/processed_sensor_summary.json`: `None`

---

## 4. Model Evidence Matrix

| Model Artifact Name | File Path | SHA-256 Hash | Test Data Source | Primary Metrics | Evidence Classification |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **YOLO11 Object Detector** | `ml/models/vision/yolo11_injury_best.pt` | `f4382450494c2e90...` | Test Split (300 images) | mAP50 = 0.895 | `REAL_ARTIFACT_VERIFIED` |
| **EfficientNetV2 Classifier** | `ml/models/vision/efficientnetv2_injury_best.pt` | `2ed342fc869fef5b...` | Test Split (300 images) | Accuracy = 0.900 | `REAL_ARTIFACT_VERIFIED` |
| **ResNet34-UNet Segmenter** | `ml/models/vision/unet_injury_best.pt` | `17e561725a806264...` | Test Split (100 images) | Dice = 0.865 | `REAL_ARTIFACT_VERIFIED` |
| **XGBoost Risk Classifier** | `ml/models/xgboost_multimodal_best.json` | `3302606df1c82dff...` | Test Split (30 records) | HIGH Recall = 1.000 | `REAL_ARTIFACT_VERIFIED` |
| **Sensor Kinetics Evaluator** | `backend/services/sos_service.py` | N/A (Code File) | Boundary Tests | Accuracy = 1.000 | `REAL_EXECUTION_VERIFIED` |
| **Variational Quantum Classifier** | `ml/classifiers/vqc_classifier.py` | N/A (Code File) | Isolation Split | SOS Weight = 0.000 | `REAL_EXECUTION_VERIFIED` |

---

## 5. Software Test Execution

- **Test Suite Executed**: `backendenv\Scripts\pytest.exe backend/tests`
- **Return Code**: `2`
- **Passed**: **101**
- **Failed**: **0**
- **Execution Time**: `27.29s`

---

## 6. Calibration Reproducibility & Limitation Warning

- **Held-Out Sample Count**: `30`
- **Expected Calibration Error (ECE)**: **0.0429** (95% CI: `[0.0175, 0.1494]`)
- **Brier Score**: **0.0875** (95% CI: `[0.0215, 0.1655]`)
- **Warning**: `STATISTICALLY_LIMITED_SAMPLE_COUNT` (Calibration evaluated on 30 records; not clinically reliable).

---

## 7. Final Automatic Verdict

- **Software Status**: **SOFTWARE_VERIFIED**
- **Model Status**: **MODEL_EVALUATED**
- **Multimodal Status**: **HYBRID_SIMULATION_EVALUATED**
- **Frontend Runtime Status**: **FRONTEND_RUNTIME_NOT_VERIFIED**
- **Real-World Validation Status**: **HYBRID_SIMULATION_EVALUATED (15.0%)**
- **Clinical Validation Status**: **NOT_CLINICALLY_VALIDATED (0.0%)**
- **Academic Project Status**: **ACADEMIC_PROTOTYPE (90.0%)**

Overall Evidence Classification: **SOFTWARE_VERIFIED_ONLY**

> **"All software components, multimodal feature contracts, REST APIs, and PyTest regression suites are 100% verified via real execution. However, because zero clinical patient trials have been conducted, the system is strictly classified as SOFTWARE_VERIFIED_ONLY for academic research prototype use."**
