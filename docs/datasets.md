# AI-QTriage Dataset Documentation & Provenance

This document records the current canonical training, validation, and held-out test datasets for all production models in AI-QTriage.

> [!IMPORTANT]
> **Research Prototype Notice:** Zero (0) genuinely paired clinical multimodal records exist in the training dataset. XGBoost and VQC models are evaluated on synthetic multimodal feature vectors derived from clinical rule heuristics. All evaluations demonstrate model recovery of synthetic labeling functions, **not** real-world clinical triage accuracy.

---

## 1. Summary of Canonical Runtime Datasets

| Modality / Model | Dataset Name | Provenance Class | Total Samples / Windows | Splits (Train / Val / Test) | Canonical Verified Performance | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO11 Detection** | `yolo_expanded_v1` | Public Real Photographs | 2,813 images (4,456 bounding boxes) | 1,935 train / 442 val / 436 test | mAP@50: **34.54%**<br>mAP@50–95: **16.74%**<br>Precision: **51.59%**, Recall: **39.42%** | `INFERENCE_EXECUTES` (Research Baseline) |
| **EfficientNetV2** | `kaggle-v1` | Public Real Photographs | 1,576 photographic images | Train / Val / 473 test | Accuracy: **96.62%** (457/473)<br>Macro-F1: **0.9434**<br>Blank / OOD collapse: **Passed** | `READY_FOR_RESEARCH_DEMO` |
| **ResNet34-UNet** | `deduped-subject-v1` | Public Wound Photographs | 230 image-mask pairs | Train / 69 val / 69 test | Mean Dice: **64.18%**<br>Mean IoU: **51.78%**<br>Blank-collapse: **Passed** | `READY_FOR_RESEARCH_DEMO` |
| **XGBoost Multimodal** | `synthetic_multimodal_fusion` | Synthetic Multimodal Vectors | 200 tabular feature vectors (23-dim) | 140 train / 30 val / 30 test (leak-free grouped split) | Accuracy: **83.33%** (25/30)<br>Macro-F1: **0.7850**<br>MCC: **+0.7027** | `TRAINED` (Research Feasibility) |
| **4-Qubit VQC** | `synthetic_multimodal_fusion` | Synthetic Multimodal Vectors | 200 tabular feature vectors (23-dim) | 140 train / 30 val / 30 test (same leak-free split) | Accuracy: **56.67%** (17/30)<br>Macro-F1: **0.4214**<br>MCC: **+0.1275** | `EXPERIMENTAL_ONLY` (PennyLane `default.qubit` Sim) |
| **Sensor Motion** | `sisfall_uci_har_real` | Real Human Biomechanics | 6,505 accelerometer windows (68 subjects) | Subject-level split (1,228 test windows) | Accuracy: **96.34%** (1,183/1,228)<br>Macro-F1: **0.8985**<br>MCC: **+0.9296** | `TRAINED` (Real Benchmark) |

---

## 2. Detailed Dataset Specifications

### 2.1 YOLO11 Object Detection (`yolo_expanded_v1`)
- **Metadata Reference:** `ml/models/vision/yolo11_metadata.json`
- **Evaluation Report:** `ml/models/yolo_expanded_skin_candidate/TRAIN_EVAL_REPORT.json`
- **Active Model SHA-256:** `857880192ebfbc4b6934a033cf7df78310885291ef00a5dd691576c6fbc4bff9`
- **Composition:** Merged from Roboflow injury detection v2, wound2, aid, Shubhambaid burn repository, and Hugging Face WSEG bounding boxes.
- **Class Support & AP@50:**
  - `cut`: 50.67% AP@50 (81 train, 24 val, 24 test boxes)
  - `bruise`: 60.08% AP@50 (458 train, 87 val, 101 test boxes)
  - `abrasion`: 61.03% AP@50 (425 train, 97 val, 94 test boxes)
  - `burn`: 16.07% AP@50 (1,503 train, 340 val, 343 test boxes)
  - `wound`: 19.36% AP@50 (194 train, 51 val, 53 test boxes)
  - `laceration`: **0.00% AP@50** (14 train, 3 val, 4 test boxes; class imbalance limitation)
- **Known Limitations:** Research baseline only. Severe class imbalance on laceration results in 0.00 AP@50. Fracture detection requires clinical X-ray imaging and is not supported on visible photographs.

### 2.2 EfficientNetV2 Multi-Class Classifier (`kaggle-v1`)
- **Metadata Reference:** `ml/models/vision/efficientnetv2_metadata.json`
- **Active Model SHA-256:** `8cb4263e70b1e020ecc6324dca9b760f38074fd7778a6b814bcb10018d58a57a`
- **Classes (8):** `abrasion`, `bruise`, `burn`, `cut`, `laceration`, `wound`, `normal`, `ood_reject`.
- **Test Performance ($N=473$):** Accuracy 96.62%, Macro-F1 0.9434. Negative controls (clean skin, non-injury images) explicitly included in evaluation to prevent blank-image prediction collapse.

### 2.3 ResNet34-UNet Segmentation (`deduped-subject-v1`)
- **Metadata Reference:** `ml/models/vision/unet_metadata.json`
- **Active Model SHA-256:** `3c7f3f39196d71b9d8d58d1fcfc7438b4ae23d75fbd8195ae7da7b1fcb9660d1`
- **Provenance:** AZH Wound Dataset + WSEG + Medetec with subject-aware, exact and near-duplicate deduplication.
- **Test Performance ($N=69$ pairs):** Mean Dice 64.18%, Mean IoU 51.78% at $\tau = 0.50$.
- **Reliability Gate:** When an image contains no detectable injury region ($0\text{ positive pixels}$ on an acute photo), the mask is **withheld** as `UNAVAILABLE_WITHHELD (N/A)` rather than reported as $0.0\%$ measured area.

### 2.4 Multimodal Tabular Fusion (`synthetic_multimodal_fusion`)
- **Evaluation Artifact:** `data/results/canonical_held_out_evaluation.json`
- **Active XGBoost SHA-256:** `b57c5cd4f844202aad62feecc8e30e03e46303f2e3e98a3337690107244f9e25`
- **Active VQC SHA-256:** `c818299beb1ff7e44bc1fa729a14befc6fd0ab63474205948b76897960e79412`
- **Generation:** 200 synthetic vectors (seed 42) with 23 features spanning vision probabilities, questionnaire responses, and sensor impact telemetry.
- **Split Hygiene:** Grouped split (`get_leakage_free_splits(200)`) strictly places duplicate synthetic templates (`{8, 143}` and `{54, 191}`) into the training partition:
  - Exact row hash overlap: `train_val = 0`, `train_test = 0`, `val_test = 0`.
- **Labels:** Rule-derived from clinical triage guidelines: LOW ($0$), MODERATE ($1$), HIGH ($2$).

### 2.5 Real Sensor Telemetry (`sisfall_uci_har_real`)
- **Metadata Reference:** `ml/models/sensor_metadata.json`
- **Provenance:** SisFall (4,505 trials across 38 subjects) + UCI HAR (2,000 windows across 30 subjects).
- **Features:** 5-dimensional kinematic feature vector (peak $g$-force, post-impact stabilization time, $\Delta v$, motion category).

---

## 3. Historical vs. Current Benchmark Evolution

| Epoch | Dataset Description | Reported Results | Architectural Status |
| :--- | :--- | :--- | :--- |
| **August 2026 (Initial Prototype)** | Naive sketch drawings (19 images) and synthetic masks (140 pairs). | 100% accuracy, ~0.98 Dice on naive test split. | **NOT_TRUSTWORTHY**: Collapsed to high-confidence injury on blank/white images. |
| **Current Canonical Baseline** | Photographic datasets (`yolo_expanded_v1`, `kaggle-v1`, `deduped-subject-v1`) with negative controls and subject-aware splits. | YOLO: 34.54% mAP@50; EffNet: 96.62% acc; UNet: 64.18% Dice; Sensor: 96.34% acc. | **READY_FOR_RESEARCH_DEMO**: Verified with negative controls, blank gates, and 18-check audit suite. |

---

## 4. Licenses
See `data/datasets/DATASET_LICENSES.md` for individual component licenses. Follow all respective attribution terms.
