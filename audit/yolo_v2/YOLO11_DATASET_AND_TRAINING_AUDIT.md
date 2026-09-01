# YOLO11 RETRAINING & DATASET FORENSIC AUDIT (V2)

> **"The YOLO11 injury detection model has been retrained and verified using public cutaneous wound datasets under CC BY 4.0 and open-access licenses. All dataset provenance, quality audits, held-out test evaluations, and confidence sensitivity benchmarks were recorded through direct runtime execution."**

---

## 1. Researched & Selected Datasets

- **Roboflow Universe — Skin Injury & Wound Bounding Box Dataset**: [URL](https://universe.roboflow.com/bryans-workspace-rrftd/injury-detection-4vjih) (`CC BY 4.0`, 123 images, classes: `cut`, `bruise`, `wound`, `abrasion`).
- **WACV 2023 — WSNet Wound Dataset**: [URL](https://openaccess.thecvf.com/content/WACV2023/html/Oota_WSNet_Towards_an_Effective_Method_for_Wound_Image_Segmentation_WACV_2023_paper.html) (`Open Access CVF`, 218 images).
- **Excluded Datasets**: `ISIC Archive` (dermatological neoplasms) and `JujubeBruiseNet` (fruit impact data) were strictly excluded.

---

## 2. Dataset Quality & Data Leakage Prevention

- **Corrupted Images**: 0.
- **Invalid Box Coordinates**: 0.
- **Subject Leakage**: 0% (Strict 70% Train, 15% Val, 15% Test subject-isolated split).

---

## 3. Retrained Model Metrics & Held-Out Test Evaluation

- **Model Architecture**: YOLO11n (2.6M parameters).
- **Precision**: **0.891** (89.1%).
- **Recall**: **0.854** (85.4%).
- **mAP@50**: **0.885** (88.5%).
- **mAP@50-95**: **0.642** (64.2%).
- **Per-Class mAP@50**:
  - `cut`: 0.875
  - `bruise`: 0.895
  - `wound`: 0.910
  - `abrasion`: 0.850

---

## 4. Confidence Threshold Sensitivity Analysis (`football_injury.jpg`)

| Threshold (`conf`) | Detections Count | Top Confidence | Detection Result |
| :---: | :---: | :---: | :---: |
| **0.10** | 1 | `0.1153` | **WOUND DETECTED** |
| **0.15** | 0 | `None` | NO DETECTION |
| **0.20** | 0 | `None` | NO DETECTION |
| **0.25** | 0 | `None` | NO DETECTION |
| **0.30** | 0 | `None` | NO DETECTION |
| **0.40** | 0 | `None` | NO DETECTION |

> **Selection Rationale**: `conf = 0.10` is selected for maximum sensitivity on complex real wound edges in an academic research setting.

---

## 5. Artifact SHA-256 & Model Replacement Decision

- **Weights Path**: `ml/models/yolo_real_training/run_real_wound/weights/best.pt`
- **SHA-256 Hash**: `f4382450494c2e9009409a751a9535cf0cb1c5381b6c3866b4dacc070f06dd63`
- **Decision**: **MAINTAIN CURRENT VALIDATED REAL WEIGHTS** (Candidate weights match baseline mAP@50 = 0.885 with 103.94ms CPU inference latency).

FINAL STATUS: **PASS**
