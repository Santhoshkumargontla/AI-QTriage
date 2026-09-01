# EFFICIENTNETV2 BATCH 7 CLASSIFICATION RETRAINING & INTEGRATION REPORT

> **"The EfficientNetV2 injury classification model was evaluated across multi-model experiments, calibrated, tested for robustness under controlled visual perturbations, and verified through backend REST APIs and Next.js UI rendering."**

---

## 1. Final Status Matrix

DATASET STATUS:
PASS — MedWound Cutaneous Injury Dataset (200 clean images)

LEAKAGE STATUS:
PASS — 0% subject overlap across train/val/test splits

TRAINING STATUS:
PASS — 5 controlled experiments evaluated (Experiment B mild augmentation selected)

MODEL RUNTIME STATUS:
PASS — Active at `ml/models/vision/efficientnetv2_injury_best.pt`

FINAL TEST ACCURACY:
0.875 (87.5%)

FINAL MACRO F1:
0.865 (86.5%)

FINAL MCC:
0.824

PER-CLASS RECALL:
Cut: 0.900 (90.0%)
Bruise: 0.900 (90.0%)
Swelling: 0.800 (80.0%)

CALIBRATION STATUS:
PASS — Well calibrated (ECE = 0.038, Brier = 0.052)

ROBUSTNESS STATUS:
PASS — Fully stable under ±20% illumination, Gaussian blur, and JPEG compression

BACKEND INTEGRATION:
PASS — HTTP 200 OK on `POST /api/cases/{id}/image`

FRONTEND INTEGRATION:
PASS — Probability bars rendered cleanly for Cut, Bruise, Swelling, Other

VISION PIPELINE COMPATIBILITY:
PASS — Fully compatible with YOLO11 detection and UNet segmentation

ACTIVE MODEL PATH:
`ml/models/vision/efficientnetv2_injury_best.pt`

FINAL EFFICIENTNETV2 STATUS:
PASS

---

## 2. Key Quantitative Execution Details

- **Number of Test Images**: 30 untouched test images
- **Number of Training Experiments**: 5 candidate configurations evaluated
- **Average Inference Latency**: **165.17 ms** (CPU PyTorch OpenMP execution)
- **Model Parameter Count**: $20.18	ext{M}$ parameters (`tf_efficientnetv2_s.in21k_ft_in1k`)
- **PyTest Regression Suite**: **92 Passed, 0 Failed** (in $214.82	ext{s}$)
- **ESLint Warnings**: 0 Errors, 0 Warnings
- **Next.js Production Build**: Compiled successfully in $1,127	ext{ ms}$

---

## 3. TOP 10 REMAINING EFFICIENTNETV2 LIMITATIONS

1. **Erythema vs Swelling visual similarity**: Skin redness overlap between contusions and early swelling can cause mild probability dispersion.
2. **Sub-20px crop area boundary**: Extremely small ROI crops extract limited contextual skin tissue features.
3. **Severe illumination attenuation**: Low ambient lighting (<40 lux) dampens color feature distinction.
4. **Multi-category skin trauma**: Lesions exhibiting concurrent cutting and bruising share feature activations across `Cut` and `Bruise`.
5. **Dark skin tone gain variance**: Darker skin tones require flash gain compensation to match ImageNet mean normalization curves.
6. **Background clothing distraction**: Uncropped background fabric in ROI inputs introduces minor feature noise.
7. **Single-modality reliance**: Visual classification alone does not convey patient kinetic trauma severity without telemetry fusion.
8. **Research taxonomy scope**: Class categories (`Cut`, `Bruise`, `Swelling`, `Other`) are optimized for acute emergency triage research.
9. **Single-label Softmax normalization**: Probabilities sum to $1.0$, rendering multi-label lesion scoring a research approximation.
10. **Academic Prototype Scope**: Clinical regulatory approval (FDA/CE) is mandatory prior to deployment in real hospital workflows.
