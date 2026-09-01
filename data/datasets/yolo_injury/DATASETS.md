# YOLO Injury Object Detection Dataset Documentation

## 1. Dataset Reference Overview
This dataset is a combined, harmonized, and deduplicated merger of injury object detection datasets.

**WARNING: fallback to synthetic cutaneous wound dataset was triggered because the provided Roboflow API key was invalid or revoked.**

> [!IMPORTANT]
> - **DATA TYPE**: Synthetic simulated research data
> - **REAL PATIENT DATA**: Non-clinical source data
> - **CLINICAL VALIDATION**: Not performed
> - **SWELLING LIMITATION**: Swelling is supported as an application category but is not currently covered by this YOLO11 training dataset.

## 2. Dataset Metrics
- **Total Unique Images**: 85
- **Split Distribution**:
  - **Train**: 59 (70%)
  - **Validation**: 12 (15%)
  - **Test**: 14 (15%)

## 3. Class Harmonization Mapping
Legitimate equivalent classes were mapped and scientifically different or unsupported categories were excluded:
- `cut` / `Cut` → **cut** (Class 0)
- `bruise` / `Bruises` / `Bruise` → **bruise** (Class 1)
- `abrasion` / `Abrasion` / `abrasions` / `Abrasions` / `Otarcie` (Polish for abrasion) → **abrasion** (Class 2)
- `Laseration` (spelling variant) / `laceration` / `Laceration` → **laceration** (Class 3)

### Excluded Classes:
- `blister`, `burn`, `rana kluta` (stab wound in Polish), `no_abnormality`, `swelling`

## 4. Leakage Prevention
- **Exact Hash duplicate detection (SHA256)**: Performed to eliminate identical image files.
- **Perceptual Hash duplicate detection (pHash)**: Performed with distance threshold < 8 to eliminate near-duplicate and re-scaled images.
- **Leakage Prevention**: Identical or near-identical images are fully excluded. The held-out test split is strictly isolated.
- **Subject-level split**: Subject-level split could not be guaranteed because the public dataset does not provide sufficient subject identifiers.
