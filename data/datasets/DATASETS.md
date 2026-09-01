# Synthetic Wound Research Dataset & References

This document records the dataset structure, synthetic generation specifications, and public academic references integrated into the **AI-QTriage** preliminary research framework.

> [!IMPORTANT]
> - **DATA TYPE**: Synthetic research data
> - **REAL PATIENT DATA**: Not used
> - **CLINICAL VALIDATION**: Not performed

---

## 1. Synthetic Wound Research Dataset Architecture

### A. Synthetic Wound Research Image Dataset
- **Dataset Title**: Synthetic Wound Research Dataset
- **Task**: Multi-class Cutaneous Injury Classification & Segmentation
- **Supported Classes**: `cut`, `bruise`, `swelling` (training coverage for swelling is currently insufficient)
- **Role in Pipeline**: Evaluates EfficientNetV2 multi-class classification and U-Net binary segmentation masks.


### B. Injury Detection Dataset (Roboflow Universe - YOLO Annotation)
- **Source**: [Roboflow Universe — Injury Segmentation / YOLOv5 Dataset](https://universe.roboflow.com/injury-segmentation/myyolov5datasetforinjuries-qmyyc)
- **Alternative Source**: [Roboflow Universe — Injury Detection](https://universe.roboflow.com/bryans-workspace-rrftd/injury-detection-4vjih)
- **Task**: Object Detection (Bounding Boxes)
- **Supported Classes**: `bruise`, `cut`, `swelling`, `abrasion`, `laceration`, `burn`
- **Academic License**: CC BY 4.0
- **Role in Pipeline**: Evaluates YOLO11 bounding box localization ($mAP_{50}$, $mAP_{50-95}$).

### C. WOUNDSEG / WSNet (WACV 2023 Research Paper)
- **Source**: [WACV 2023 — WSNet: Towards an Effective Method for Wound Image Segmentation](https://openaccess.thecvf.com/content/WACV2023/html/Oota_WSNet_Towards_an_Effective_Method_for_Wound_Image_Segmentation_WACV_2023_paper.html)
- **Task**: Pixel-level Binary Wound & Affected Area Segmentation
- **Academic License**: Open Access CVF / WACV 2023 Conference License
- **Role in Pipeline**: Evaluates U-Net binary segmentation masks (Dice score, IoU).

### D. George Mason University (GMU) Bruise / Injury Data Project
- **Source**: [George Mason University — Bruise Data Project](https://bruise.gmu.edu/data/)
- **Task**: Longitudinal Bruise Colorimeter & Cutaneous Injury Attribution
- **Role in Pipeline**: Provides secondary research reference distributions for bruise severity heuristics.

---

## 2. Explicitly Excluded Datasets

> [!WARNING]
> **JujubeBruiseNet (Mendeley)** is strictly **EXCLUDED**. It contains agricultural fruit (jujube) impact bruise data and is invalid for human cutaneous injury assessment.

---

## 3. Dataset Splitting & Data Leakage Prevention

All dataset manifests (`manifest.csv`) enforce **Subject-Level Splitting**:
- **70% Training Split**: Used for fitting `StandardScaler`, `PCA`, `XGBoost`, `VQC`, and vision backbone transfer learning.
- **15% Validation Split**: Used for hyperparameter tuning.
- **15% Held-out Test Split**: Untouched test set used for evaluation of classical vs quantum models.
- **Data Leakage Rule**: Samples from the same `subject_id` are strictly isolated within a single split and never overlap across train/val/test splits.
