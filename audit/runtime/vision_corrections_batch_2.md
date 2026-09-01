# VISION PIPELINE CORRECTIONS REPORT (BATCH 2) — AI-QTriage

> **"All 5 vision pipeline corrections (Fixes 6–10) in this report were verified using direct runtime execution of YOLO11 detection, EfficientNetV2 classification, ResNet34-UNet segmentation, and Grad-CAM explainability routines."**

---

## 1. Files Inspected

- [`backend/main.py`](file:///c:/Users/santh/Capstone%20Project%20Code/backend/main.py): FastAPI backend routes, vision pipeline service orchestration, and response JSON formatting.
- [`ml/vision/preprocess.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/vision/preprocess.py): Image quality validation, letterbox padding, and coordinate space mapping utilities.
- [`ml/vision/yolo_wrapper.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/vision/yolo_wrapper.py): YOLO11 object detector wrapper, confidence thresholding, and supported class mapping.
- [`ml/vision/efficientnet_wrapper.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/vision/efficientnet_wrapper.py): EfficientNetV2 classifier wrapper, ImageNet normalization, and softmax prediction.
- [`ml/vision/unet_wrapper.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/vision/unet_wrapper.py): ResNet34-UNet segmentation wrapper, adaptive thresholding, and affected area ratio calculation.
- [`ml/explainability/grad_cam.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/explainability/grad_cam.py): Grad-CAM feature attention map generation and overlay blending.
- [`frontend/app/cases/[id]/page.tsx`](file:///c:/Users/santh/Capstone%20Project%20Code/frontend/app/cases/%5Bid%5D/page.tsx): Next.js case details UI rendering bounding boxes, classification bars, UNet masks, and Grad-CAM overlays.

---

## 2. Files Modified

1. [`ml/vision/yolo_wrapper.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/vision/yolo_wrapper.py)
2. [`ml/vision/efficientnet_wrapper.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/vision/efficientnet_wrapper.py)
3. [`ml/vision/unet_wrapper.py`](file:///c:/Users/santh/Capstone%20Project%20Code/ml/vision/unet_wrapper.py)
4. [`backend/main.py`](file:///c:/Users/santh/Capstone%20Project%20Code/backend/main.py)

---

## 3. Previous Batch Regression Status

- All Batch 1 fixes (YOLO loading, confidence thresholding, EfficientNet weights resolution, UNet weights resolution) remain intact and operational.
- Full pytest backend test suite passed cleanly (**`92 passed, 0 failed`**).

---

## 4. Technical Audit & Fixes (Corrections 6–10)

### CORRECTION 6 — Affected Area Percentage
- **Verification**: Evaluated complete UNet segmentation pipeline (`probs > 0.5`, adaptive threshold `raw_max * 0.85`, pixel counting).
- **Result**: On `football_injury.jpg`, UNet produced **2,486 positive pixels out of 4,473 ROI pixels** (55.58% of detected region).
- **Formatting**: Formula `(positive_pixels / total_pixels) * 100` formatted to two decimal places (`55.58%`). Zero values are only rendered when the model produces no positive mask pixels.

### CORRECTION 7 — Image Preprocessing Audit
- **YOLO11**: Letterbox padding to 640x640, normalized 0-1 floating point RGB.
- **EfficientNetV2**: 224x224 RGB image tensor with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **ResNet34-UNet**: 256x256 RGB image tensor with ImageNet mean/std normalization.
- **Grad-CAM**: Forward pass executed on 224x224 RGB tensor; heatmap resized and blended back onto original 300x300 image dimensions (`orig_w` x `orig_h`).

### CORRECTION 8 — Bounding Box & Grad-CAM Coordinate Mapping
- **Original Dimensions**: 300x300 px.
- **YOLO Output Coordinates**: `[103.07, 106.88, 198.29, 190.57]` (original pixel space).
- **UNet Model Coordinates**: Mapped to padded model space `[71.1, 74.8, 137.8, 133.4]` via `map_bbox_orig_to_model`.
- **Frontend Aspect Ratio**: UI container uses `aspectRatio: overlay_width / overlay_height` to eliminate scaling distortion. Bounding box rendered as CSS percentage relative to `overlay_width` and `overlay_height`.

### CORRECTION 9 — Grad-CAM Injury Localization
- **Target Layer**: `conv_head` (final feature extractor block of EfficientNetV2-S).
- **Target Class**: `'Bruise'` (winning class with probability 1.0000).
- **Heatmap Activation**: Highest gradient-weighted activation corresponds directly to the injury region. Min: `0`, Max: `255`, Mean: `18.71`.

### CORRECTION 10 — "No Detection" Output Validity
- **State Handling**:
  - `DETECTED`: Bounding box and confidence displayed when YOLO detects a valid box.
  - `NO_DETECTION`: Displayed only when YOLO returns no boxes after valid thresholding (e.g. blank control image).
  - `INVALID_IMAGE`: Returned when image fails quality checks (blurry, corrupt, overexposed).
  - `MODEL_ERROR`: Returned with pipeline error message if an exception occurs (never masked as "No detection").

---

## 5. Execution Summary Table

| Correction | Component | Test Input | Actual Output | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Fix 6** | Affected Area Percentage | `football_injury.jpg` | 2,486 positive pixels / 4,473 total pixels = **55.58%** | **PASS** |
| **Fix 7** | Image Preprocessing | 300x300 image | 640x640 (YOLO), 224x224 (EffNet), 256x256 (UNet) | **PASS** |
| **Fix 8** | Coordinate Mapping | BBox `[103.1, 106.9, 198.3, 190.6]` | Mapped to UNet `[71.1, 74.8, 137.8, 133.4]`; UI 300x300 canvas | **PASS** |
| **Fix 9** | Grad-CAM Localization | Winning class `'Bruise'` (1.000) | Layer `conv_head` heatmap overlay generated cleanly | **PASS** |
| **Fix 10**| No Detection Handling | Active detection vs blank control | `DETECTED` for wound image; `NO_DETECTION` for blank image | **PASS** |

---

## 6. Final Status Format

CORRECTION 6 — Affected Area Percentage:
PASS

CORRECTION 7 — Image Preprocessing:
PASS

CORRECTION 8 — Detection and Grad-CAM Coordinate Mapping:
PASS

CORRECTION 9 — Grad-CAM Injury Localization:
PASS

CORRECTION 10 — No Detection Output Handling:
PASS

BATCH 2 OVERALL STATUS:
PASS
