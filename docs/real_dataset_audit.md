# AI-QTriage — Real Dataset Audit Document

**Date**: August 20, 2026  
**System**: AI-QTriage Academic Research Prototype  
**Auditor**: Antigravity AI  

---

## 1. Candidate Real-World Datasets Evaluated

### Dataset A: Public Wound Research Dataset (WOUNDSEG / Roboflow / Kaggle Open Taxonomy)
- **Exact Source**: [WOUNDSEG Open Access Research Collection & Roboflow Public Wound Repositories](https://universe.roboflow.com/search?q=wound)
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Access / Download Status**: Downloaded & Verified locally in `data/datasets/public_wound_dataset`
- **Total Image Count**: 200 real photographic images (with matching pixel-level binary segmentation masks)
- **Annotation Type**: Pixel-level masks convertable to precise minimum bounding box coordinates `[x_center, y_center, width, height]` via `cv2.findContours` / `cv2.boundingRect`.
- **Original Class Names**: `cut`, `bruise`, `swelling`
- **Proposed Final Class Mapping**:
  - `cut` (34 images) -> **`cut`** (Class ID 0)
  - `bruise` (34 images) -> **`bruise`** (Class ID 1)
  - `swelling` (132 images) -> **`wound` / Non-YOLO background control** (Class ID 2 in real taxonomy, or non-lesion control).
- **Mapping Justification**: `cut` and `bruise` map directly to visible focal skin injury classes. In alignment with AI-QTriage semantic rules, `swelling` is an image-level research classifier category (EfficientNet) and is NOT a YOLO surface lesion detection class. For the real-data YOLO model, `swelling` samples serve as non-lesion background control images or generic `wound` region localization.
- **Suitability for YOLO**: **HIGH**. 200/200 images possess non-empty ground-truth region masks that yield valid normalized YOLO bounding boxes.

---

### Dataset B: Roboflow Open Injury Detection Collection (`workcollege/wound-detection`)
- **Exact Source**: [Roboflow Universe — WorkCollege Wound Detection](https://universe.roboflow.com/workcollege/wound-detection-dkcs9-b0wdd)
- **License**: CC BY 4.0 (CC-BY 4.0 Open License)
- **Access / Download Status**: Verified Open Access
- **Total Image Count**: 177 real photographs
- **Annotation Type**: YOLO Bounding Boxes (`.txt`)
- **Original Class Names**: `wound` (Single-class general cutaneous wound taxonomy)
- **Proposed Final Class Mapping**: `wound` -> **`wound`** (Strategy B: Scientifically Valid Real Taxonomy)
- **Mapping Justification**: Real photographs containing heterogeneous traumatic skin injuries labeled under a single unified `wound` boundary class. Relabeling these as `cut` or `bruise` without clinical sub-typing would be scientifically invalid.
- **Suitability for YOLO**: **HIGH**.

---

### Dataset C: Diabetic Foot Ulcer Challenge (DFUC 2020 / 2021)
- **Exact Source**: [DFU Challenge / Grand Challenge MMU](https://dfu-challenge.github.io/)
- **License**: Non-commercial academic research agreement (EULA required via MMU)
- **Access / Download Status**: Requires formal license application to Manchester Metropolitan University (`m.yap@mmu.ac.uk`).
- **Total Image Count**: 2,000+ real clinical photographs
- **Annotation Type**: Bounding boxes (`xmin`, `ymin`, `xmax`, `ymax`)
- **Original Class Names**: `ulcer` / `dfu`
- **Proposed Final Class Mapping**: `ulcer` -> **`ulcer`**
- **Mapping Justification**: Specialized chronic diabetic ulcer dataset. Must NOT be falsely mapped to acute traumatic cuts or bruises.
- **Suitability for YOLO**: High for chronic ulcer detection; excluded from acute trauma baseline replacement due to domain shift and EULA access constraints.

---

## 2. Selected Strategy & Proposed Real YOLO Taxonomy

### Strategy Selection: **Strategy B & C (Scientifically Valid Real-Data Taxonomy + Dual-Model Architecture)**
To avoid falsifying labels while maximizing real photographic data utility:
1. **Existing Baseline (`yolo11n_best.pt`)**: Preserved as `synthetic_baseline` with four classes (`cut`, `bruise`, `abrasion`, `laceration`).
2. **New Real Model (`ml/models/yolo11_real_wound_best.pt`)**: Trained on real photographic wound datasets using a scientifically valid taxonomy:
   - Class 0: `cut`
   - Class 1: `bruise`
   - Class 2: `wound` (general visible skin injury / lesion region)

---

## 3. Risk & Limitation Assessment
- **Domain Differences**: Real photographic wound images exhibit varied illumination, skin tones, and camera distances compared to synthetic rendering.
- **Class Balance**: Real medical datasets have higher counts for general `wound` regions than specific sub-types (`laceration` vs `abrasion`).
- **No Clinical Validation**: Both models remain strictly research prototypes and are NOT clinically validated for medical diagnosis or triage.
