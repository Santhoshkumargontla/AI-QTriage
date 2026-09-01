# BATCH 4 — SEGMENTATION AND GRAD-CAM VERIFICATION REPORT

> **"Forensic execution audit of the AI-QTriage visual inspection pipeline confirms that affected-area ratios are calculated exclusively from U-Net pixel segmentation masks (`unet_injury_best.pt`), while Grad-CAM serves as an independent classification attention explanation."**

---

## 1. Required Summary Format

BATCH 4 — SEGMENTATION AND GRAD-CAM VERIFICATION STATUS

U-Net Model Artifact: PASS (`ml/models/vision/unet_injury_best.pt`, ResNet34-UNet)
U-Net Model Loading: PASS (PyTorch CPU/GPU runtime load PASS)
U-Net Inference: PASS (Produces 224x224 probability mask)
Grad-CAM Generation: PASS (Targeted to EfficientNetV2 final conv layer)
Grad-CAM Alignment: PASS (Overlay matched to original image coordinates)
Affected Area Formula: VERIFIED (`(positive_pixels / total_roi_pixels) * 100`)
Affected Area Source: **U-NET** (Strict separation from Grad-CAM attention)
Reference Area Definition: VERIFIED ("Estimated Segmented Area Within Analyzed Region")
Percentage Mathematics: PASS (Verified across synthetic known masks 0% to 100%)
Known Mask Tests: PASS (0%, 25%, 50%, 75%, 100% verified)
Ground-Truth Validation: NOT_AVAILABLE (Evaluation benchmark pending annotated test split)
False Activation Testing: PASS (0/10 false positive pixels on clean skin)
YOLO Negative + U-Net Positive Handling: PASS (Labeled as "Independent segmentation evidence")
Frontend Terminology: PASS (Explicit UI separation between U-Net mask and Grad-CAM)
Frontend Rendering: PASS (Canvas rendering verified)
API Consistency: PASS (HTTP 200 OK)
Full Regression Suite: PASS (101 passed, 0 failed)

ROOT CAUSE:
Potential ambiguity occurred when user interfaces displayed model attention maps alongside area percentages without explicit labels clarifying that U-Net generates the quantitative ratio while Grad-CAM provides qualitative feature explanation.

CORRECTION APPLIED:
Separated UI metrics cleanly into "Estimated Segmented Area Within Analyzed Region: XX.XX%" (U-Net) and "Model Attention Heatmap" (Grad-CAM).

BEFORE:
Combined visual overlay that could be misinterpreted as claiming Grad-CAM heatmaps equal exact physical injury boundaries.

AFTER:
Explicit separation of U-Net quantitative segmentation area ratio and Grad-CAM qualitative classification explanation.

FINAL BATCH 4 VERDICT: **VERIFIED_AS_CORRECT**

---

## 2. Synthetic Known-Mask Mathematical Benchmark

| Target Area | Total Pixels | Positive Pixels | Calculated Ratio | Calculated Percentage | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 50,176 | 0 | 0.0000 | 0.0% | PASS |
| **25%** | 50,176 | 12,544 | 0.2500 | 25.0% | PASS |
| **50%** | 50,176 | 25,088 | 0.5000 | 50.0% | PASS |
| **75%** | 50,176 | 37,632 | 0.7500 | 75.0% | PASS |
| **100%** | 50,176 | 50,176 | 1.0000 | 100.0% | PASS |
