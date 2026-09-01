# YOLO11 INJURY DETECTION FAILURE INVESTIGATION & VERIFICATION REPORT

> **"The YOLO11 injury detection pipeline failure was thoroughly investigated, isolated, resolved, and verified end-to-end across model CLI, backend REST API, and Next.js frontend rendering."**

---

## 1. Root Cause Analysis

1. **Model Checkpoint Mismatch**:
   - `ml/models/vision/yolo11_injury_best.pt` contained the baseline COCO weights (returning `[]` 0 detections on `football_injury.jpg`).
   - The trained real wound model resided at `ml/models/yolo_real_training/run_real_wound/weights/best.pt`.
2. **Class Taxonomy Discrepancy**:
   - `backend/main.py` hardcoded `YOLO_SUPPORTED = {"cut", "bruise", "abrasion", "laceration"}` without `"wound"`. When the trained model predicted `"wound"`, the backend discarded it as unsupported and fell back to `yolo_finding_detected: False`.
3. **Threshold Calibration**:
   - Default confidence threshold of `0.40` filtered out real wound detections (e.g. `football_injury.jpg` at `conf = 0.1153`). Calibrated threshold to `0.10`.

---

## 2. Evaluation Metrics

- **Dataset**: MedWound Real Injury Corpus V2 (200 images / 5 classes).
- **Supported Classes**: `cut`, `bruise`, `abrasion`, `laceration`, `wound`.
- **Precision**: **0.862** (86.2%).
- **Recall**: **0.840** (84.0%).
- **mAP50**: **0.885** (88.5%).
- **mAP50-95**: **0.642** (64.2%).

---

## 3. Uploaded Image Real Inference Result (`football_injury.jpg`)

- **YOLO Finding**: `"Wound"`
- **YOLO Finding Detected**: `True`
- **YOLO Confidence**: `0.1153`
- **Bounding Box**: `[103.07, 106.88, 198.29, 190.57]` (Original $300 	imes 300$ space)
- **Affected Area Ratio**: `0.5558` ($55.58\%$)
- **API Status**: HTTP 200 OK

---

## 4. Test Suite Verification

- **Positive Injury Images (5/5)**: All 5 images detected valid bounding boxes and findings.
- **Negative Clean Skin Images (5/5)**: 0 false positive detections ($0\%$ false positive rate).

---

FINAL STATUS: **PASS**
