# AI-QTriage strict data provenance and leakage audit

**Date:** 2026-08-29  
**Method:** SHA-256 file bytes, SHA-256 RGB pixels+WxH, 8×8 average-hash Hamming ≤ 5, float32 vector hashes. Training-code inspection for scaler/PCA/feature selection.  
**Machine-readable manifest:** `data/datasets/CANONICAL_DATASET_MANIFEST.json`  
**Per-record hashes:** `data/datasets/canonical_dataset_records.csv`

No dataset in this repository is **REAL** clinical photography or downloaded public recordings. `np.random` and PIL drawings are **SYNTHETIC**. Sensor windows and PennyLane `default.qubit` are **SIMULATED**. `football_injury.jpg`, `blank_skin.jpg`, and `dummy_test.jpg` are **DEMO** / eval-only files. Nothing is labeled **PUBLIC** as a download receipt.

If `subject_id` is missing, or is a generator label (`subj_001`…), status is **`SUBJECT_LEAKAGE_NOT_VERIFIABLE`**. That is not zero subject leakage.

Rejected: `data/datasets/DATASET_PROVENANCE_AUDIT.json` (PIL drawings as Kaggle/MedWound; np.random sensor as SisFall/UCI HAR) and `data/datasets/LEAKAGE_AUDIT.json` (`ZERO_SUBJECT_LEAKAGE`, 0 SHA-256 duplicates). Live hashes contradict both.

---

## FINAL REPORT

| Model | Dataset | Data provenance | Split sizes | Subject counts | Duplicate overlap (exact hash train/val, train/test, val/test) | Subject overlap | Leakage status | Final conclusion |
|---|---|---|---|---|---|---|---|---|
| YOLO11 Detection | `yolo_retrain_v2` (production) | **SYNTHETIC** drawings + 2 DEMO empty-label files | train 87, val 20, test 19 | 0 (no subject_id) | 0, 0, 0 | n/a | No exact-hash split leak. 1153 aHash near-dup pairs across splits. **SUBJECT_LEAKAGE_NOT_VERIFIABLE** | Exact files are split-disjoint. Near-identical canvases still leak template identity. Not real wounds. |
| EfficientNetV2 | `public_wound_dataset` (production) | **SYNTHETIC** PIL drawings (CSV claims Kaggle/Roboflow; generator is `download_public_datasets.py`) | train 140, val 30, test 30 | 40 generator IDs (28/6/6 disjoint) | 5, 5, 5 pixel-hash groups | generator IDs 0/0/0 | **EXACT_HASH_CROSS_SPLIT**. 38 unique pixels / 200 files. 2736 near-dup pairs. **SUBJECT_LEAKAGE_NOT_VERIFIABLE** | Production classifier saw duplicated templates in every split. Do not call this PUBLIC or REAL. Candidate `efficientnet_processed` is hash-disjoint and **not loaded**. |
| ResNet34-UNet | `public_wound_dataset` (production) | **SYNTHETIC** (same as EfficientNet) | train 140, val 30, test 30 | 40 generator IDs | 5, 5, 5 | generator IDs 0/0/0 | Same exact-hash leak as EfficientNet. **SUBJECT_LEAKAGE_NOT_VERIFIABLE** | Live U-Net SHA was trained on the leaking drawing set. Candidate `unet_processed` (41 unique hashes, 29/6/6, 0 exact overlap) is **not loaded**. |
| XGBoost Multimodal | `synthetic_multimodal_fusion` | **SYNTHETIC** (`np.random`, seed 42, 23-d) | train 140, val 30, test 30 | 0 | 1 train/val vector, 1 train/test vector (198 unique / 200) | n/a | Two accidental identical rows across splits. **SUBJECT_LEAKAGE_NOT_VERIFIABLE** | Not real-world. Scaler/PCA fit on train then **unused** by the booster. No feature selection. No HP search. Val unused; test is `[170:]` only. |
| Experimental 4-Qubit VQC | same matrix as XGBoost | **SYNTHETIC** data; circuit **SIMULATED** (`default.qubit`) | train 140, val 30 unused, test 30 | 0 | same two duplicate vectors | n/a | Same vector leak. Scaler/PCA `n_samples_seen_=140` on disk. **SUBJECT_LEAKAGE_NOT_VERIFIABLE** | Adam on train only. Val not materialized, not merged into test. No feature selection. No test-set HP search. Not a QPU. |
| Sensor Motion Event Classifier | `synthetic_50hz_motion_windows` | **SYNTHETIC** + **SIMULATED** (`np.random` 50 Hz windows). CSV `source_dataset` says SisFall/UCI HAR; **raw dir does not exist** | train 138, val 26, test 36 | 38 generator IDs (26/5/7, overlap 0) | 0, 0, 0 | generator IDs 0/0/0 | No exact vector leak. **SUBJECT_LEAKAGE_NOT_VERIFIABLE** | Disjoint fake IDs are not patients. Scaler fit on train; val/test transform only. No feature selection. No HP search. Val not merged into test. |

---

## Image datasets (supporting)

| Dataset | Role | Provenance | n | Unique pixels | Exact cross-split groups | aHash near-dup pairs | Subject status |
|---|---|---|---:|---:|---:|---:|---|
| `yolo_retrain_v2` | YOLO production | SYNTHETIC | 126 | 126 | 0 | 1153 | SUBJECT_LEAKAGE_NOT_VERIFIABLE |
| `public_wound_dataset` | EfficientNet + U-Net production | SYNTHETIC | 200 | 38 | 5 (each spans train+val+test; e.g. one cut template n=34) | 2736 | generator IDs disjoint; **NOT_VERIFIABLE** |
| `injury_dataset` | U-Net candidate source | SYNTHETIC | 30 | 3 | 3 (all splits) | 74 | generator IDs disjoint; **NOT_VERIFIABLE** |
| `efficientnet_processed` | EfficientNet candidate (not production) | SYNTHETIC | 126 | 126 | 0 | 1090 | SUBJECT_LEAKAGE_NOT_VERIFIABLE |
| `unet_processed` | U-Net candidate (not production) | SYNTHETIC | 41 | 41 | 0 | 163 | SUBJECT_LEAKAGE_NOT_VERIFIABLE |

Near-duplicate counts are high because drawings share a skin-tone canvas. Exact pixel-hash overlap is the identity leak. aHash overlap is template similarity.

---

## XGBoost / VQC / Sensor preprocess (code + disk)

| Check | XGBoost | VQC | Sensor |
|---|---|---|---|
| StandardScaler fit on train only | Yes, then discarded | Yes; `scaler.pkl` n=140 | Yes; `sensor_scaler.pkl` |
| PCA fit on train only | Yes, unused at train/infer | Yes; `pca.pkl` n=140 | Not used |
| Feature selection fitted | None | None | None |
| HP tuning on final test | No (hardcoded) | No (hardcoded epochs/lr) | No (hardcoded) |
| Optimizer on train only | Booster on `X_train` | Adam on `X_train` | XGB on scaled train |
| Val merged into test metrics | No | No (val never built) | No |

XGBoost accidental duplicates: `mm_008` (train) = `mm_143` (val); `mm_054` (train) = `mm_191` (test).

---

## DEMO files

| Path | Label | Role |
|---|---|---|
| `data/sample/image/football_injury.jpg` | DEMO | UI / OOD probe |
| `data/datasets/yolo_injury/blank_skin.jpg` | DEMO | YOLO val empty-label; EfficientNet/U-Net eval-only |
| `data/datasets/yolo_injury/dummy_test.jpg` | DEMO | YOLO test empty-label; EfficientNet/U-Net eval-only |

---

## Overall conclusion

Every production model is trained on **SYNTHETIC** (and for sensor/VQC, **SIMULATED**) data. EfficientNet and U-Net production sets have **exact pixel-hash leakage** across train/val/test. YOLO files are hash-disjoint but near-duplicate. XGBoost/VQC have two identical feature rows across splits. No model has verifiable real-subject independence: **SUBJECT_LEAKAGE_NOT_VERIFIABLE**.
