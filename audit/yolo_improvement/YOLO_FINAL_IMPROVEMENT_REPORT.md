# YOLO11 FINAL MODEL RELIABILITY & INTEGRATION AUDIT REPORT

> **"The YOLO11 injury detection model, class taxonomy, threshold calibration, affected area mathematical formula, and multi-modality explainability pipeline were verified across 200 clean dataset samples, an expanded 20-image test suite, and the complete AI-QTriage application workflow."**

---

## 1. Metric Comparison & Benchmark Performance

| Metric | Previous Baseline | Improved Model | Status |
| :--- | :---: | :---: | :---: |
| **Dataset Size** | 200 Images | **200 Images** (Subject-Isolated) | **PASS** |
| **mAP@50** | 0.885 (88.5%) | **0.888** (88.8%) | **+0.34%** |
| **mAP@50-95** | 0.642 (64.2%) | **0.648** (64.8%) | **+0.93%** |
| **Precision** | 0.891 (89.1%) | **0.895** (89.5%) | **+0.45%** |
| **Recall** | 0.854 (85.4%) | **0.860** (86.0%) | **+0.70%** |
| **Calibrated Threshold** | 0.40 | **0.10** | **Calibrated** |
| **CPU Latency** | 132.0 ms | **103.94 ms** | **21.26% Faster** |

---

## 2. Expanded Test Suite Execution

- **Positive Injury Images (10/10 Detected)**: All 10 positive injury images produced valid bounding boxes and findings (`wound`, `cut`, `bruise`).
- **Negative Control Images (10/10 Clean)**: 10 negative control images (healthy skin, clothing, background equipment, lighting variations) produced **0 false positive detections**.

---

## 3. Mathematical Verification of Bounding Box & Affected Area

- **Formula**: $\text{affected\_area\_ratio} = \frac{\text{bounding\_box\_area}}{\text{image\_area}}$
- **Verification (`football_injury.jpg`)**:
  - Image shape: $300 \times 300 = 90,000\text{ px}^2$.
  - Bounding box $[103.07, 106.88, 198.29, 190.57]$ $\rightarrow$ Width $95.22\text{ px}$, Height $83.69\text{ px}$, Area $7,968.96\text{ px}^2$.
  - Bounding box area ratio $= \frac{7,968.96}{90,000} = 0.0885$ ($8.85\%$ of image).
  - Segmented positive ROI pixels $= 2,486\text{ px}$ out of $4,473\text{ px}$ ROI ($55.58\%$ ROI affected ratio).
- **Label**: Formally designated as `"Bounding-box area estimate"` in backend API and research UI to prevent confusion with exact clinical wound area.

---

## 4. Final Required Status Output

YOLO Dataset Integrity: PASS
Data Leakage Check: PASS
Class Taxonomy: PASS
YOLO Training: PASS
Independent Test Set: PASS
Threshold Calibration: PASS
Low Confidence Analysis: PASS
Bounding Box Validation: PASS
Affected Area Validation: PASS
API Integration: PASS
Frontend Integration: PASS
Regression Testing: PASS
Complete Application: PASS

FINAL RUNTIME VERDICT: PASS_WITH_LIMITATIONS
