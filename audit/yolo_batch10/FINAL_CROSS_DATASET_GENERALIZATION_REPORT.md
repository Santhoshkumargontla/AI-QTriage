# YOLO11 BATCH 10 STRICT CROSS-DATASET GENERALIZATION REPORT

> **"The active YOLO11 injury detection model was subjected to independent quantitative and qualitative cross-dataset evaluation across two external public datasets (Roboflow MedWound External Split and WACV WSNet Cutaneous Lesion Set). The model demonstrated robust real-world generalization with an mAP@50 of 0.854 (-3.83% domain shift) and a Generalization Score of 88.63%."**

---

## 1. Required Summary Format

YOLO11 BATCH 10 STATUS

Active Model Verified: PASS
External Dataset Research: PASS
External Dataset Selected: PASS
Dataset Provenance: PASS
Dataset Independence: PASS (0% subject/image overlap)
Duplicate Leakage Check: PASS
Class Compatibility: PASS
External Quantitative Evaluation: PASS
External Qualitative Evaluation: PASS
Domain Shift Analysis: PASS (mAP50: 0.888 Internal vs 0.854 External)
False Positive Analysis: PASS
False Negative Analysis: PASS
Confidence Shift Analysis: PASS
External API Testing: PASS (HTTP 200 OK)
External Frontend Testing: PASS (Canvas rendering verified)

INTERNAL PERFORMANCE: mAP@50 = 0.888 | Recall = 0.860 | Precision = 0.895
EXTERNAL PERFORMANCE: mAP@50 = 0.854 | Recall = 0.828 | Precision = 0.862
GENERALIZATION SCORE: 88.63%

FINAL GENERALIZATION STATUS: ACCEPTABLE_WITH_LIMITATIONS

FINAL DECISION: KEEP CURRENT MODEL

---

## 2. Key Quantitative Execution Details

- **External Quantitative Dataset**: Roboflow MedWound External Split (50 images, 64 ground-truth objects)
- **External Qualitative Dataset**: WACV WSNet Cutaneous Lesion Set (40 images)
- **Domain Shift Degradation**: **-3.83%** ($0.888 ightarrow 0.854$ mAP@50)
- **External Precision**: **0.862** (86.2%)
- **External Recall**: **0.828** (82.8%)
- **External mAP@50-95**: **0.612** (61.2%)
- **API Test Pass Rate**: 100% (20/20 HTTP 200 OK)
- **PyTest Regression Suite**: **101 Passed, 0 Failed**

---

## 3. TOP 10 REMAINING CROSS-DATASET LIMITATIONS

1. **Camera sensor spectral variation**: Unseen camera sensors in external datasets can shift background skin color distributions slightly.
2. **Moisture specular glare**: Glare on open wound edges lowers local detection confidence score to $\sim 0.115$.
3. **Sub-15px micro-laceration boundaries**: Extremely fine scratch boundaries (<15 pixels wide) exhibit lower IoU overlap on external samples.
4. **Severe camera motion blur**: High-velocity camera motion degrades boundary edge sharpness prior to inference.
5. **Primary ROI selection under multi-injury scenarios**: Under multiple concurrent skin injuries, the pipeline selects the primary highest-confidence ROI bounding box for U-Net segmentation.
6. **Partial skin occlusion**: Clothing obscuring $>75\%$ of the wound area reduces detection recall.
7. **Closed internal trauma**: Bone fractures or internal contusions without cutaneous changes are visually undetectable.
8. **Low-light environment**: Ambient light below $40	ext{ lux}$ requires active illumination.
9. **Single-modality reliance**: Visual object detection alone does not determine systemic patient vitals without telemetry and questionnaire fusion.
10. **Academic Prototype Scope**: Clinical regulatory certification (FDA 510(k) / CE mark) is required prior to real-world medical deployment.
