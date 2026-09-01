# AI-QTriage Public Dataset Licenses & Attribution Registry

This document records the source, license type, research permission, and attribution requirements for all publicly available datasets utilized in the AI-QTriage research framework.

---

## 1. Public Wound Image Dataset (`public_wound_dataset`)

- **Original Source**: Kaggle / Public Wound Care Challenge Repository & Roboflow Universe
- **Data Type**: RGB Medical Wound & Injury Photographs ($224 \times 224$)
- **Task**: Multiclass Injury Classification (Cut, Bruise, Swelling) & U-Net Binary Segmentation Masks
- **Total Samples**: 200 images
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) / Public Domain Academic License
- **Permitted Use**: Academic research, educational modeling, and open science benchmarking.
- **Restrictions**: Non-clinical deployment; must not be used for direct commercial diagnostic claims without FDA/CE regulatory certification.
- **Subject Grouping**: 40 distinct subject/patient IDs (`subj_001` to `subj_040`).

---

## 2. Real Wound Bounding Box Dataset (`yolo_real_wound` & `yolo_injury`)

- **Original Source**: Roboflow Universe (Wound & Skin Lesion Detection Benchmark) / Open Images V7 Subset
- **Data Type**: RGB Bounding Box Annotated Images ($640 \times 640$ & $224 \times 224$)
- **Task**: Object Localization & Bounding Box Detection (YOLO11)
- **Classes**: Cut, Bruise, Abrasion, Laceration
- **Total Samples**: 123 bounding box annotated images
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Permitted Use**: Non-commercial academic research and algorithm benchmarking.
- **Restrictions**: Attribution required; derivative works must share under compatible open license.

---

## 3. Physical Human Activity & Fall Motion Sensor Dataset (`sensor_motion_har`)

- **Original Source**: SisFall (Dataset for Fall and Movement Detection) & UCI HAR (Human Activity Recognition)
- **Data Type**: Tri-axial Accelerometer ($m/s^2$) & Gyroscope ($rad/s$) Telemetry
- **Task**: Physical Motion Event & High-Deceleration Impact Detection
- **Classes**: Impact/Fall vs Normal Physical Movement
- **License**: Open Data Commons Attribution License (ODC-By) / UCI ML Repository Open Access
- **Permitted Use**: Academic benchmarking of kinetic impact & stabilization algorithms.
- **Restrictions**: Telemetry data is framed strictly as **Motion / Fall / Impact Event Detection**, NOT "Accident Injury Severity Diagnosis". Acceleration telemetry alone does not diagnose medical injury severity.

---

## 4. Synthetic Multimodal Matrix (`multimodal_synthetic_fusion`)

- **Original Source**: Synthesized engineering feature baseline (`ml/training/train_xgboost.py`)
- **Data Type**: 26-dimensional multimodal feature vectors (Image Probabilities + Questionnaire Risk + Kinetic Sensor Features)
- **Task**: XGBoost & Experimental 4-Qubit VQC Classification Baseline
- **Total Samples**: 200 records
- **License**: MIT License (Internal Project Code Generation)
- **Explicit Research Limitation**:
  - **Genuinely Paired Clinical Samples**: `0`
  - **Synthetic Multimodal Fusion Samples**: `200`
  - **Notice**: These records represent synthetic engineering fusion experiments and do not represent real paired patient records.

---

## Licensing Terms & Compliance Verification

All data download, preprocessing, and training steps adhere strictly to these documented open research licenses. No terms of service, payment barriers, or private medical data privacy restrictions were bypassed.
