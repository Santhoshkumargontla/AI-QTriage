# AI-QTriage dataset notes

Counts below are what the **current canonical training metadata** records. Older audit reports that list 1,240 YOLO images or 850 U-Net masks are **not** the runtime training sets.

## YOLO11 (`yolo_merged` / processed detect set)

From `ml/models/vision/yolo11_metadata.json`:

| Split | Images | cut boxes | bruise boxes | wound boxes |
| --- | ---: | ---: | ---: | ---: |
| Train | 83 | 15 | 12 | 288 |
| Val | 17 | 2 | 1 | 70 |
| Test | 23 | 5 | 4 | 65 |

`abrasion` and `laceration` were remapped to `wound` in the merged set (not trained as their own classes). `swelling` was never a YOLO class.

Processed drop-remap set: `data/datasets/yolo_processed` (see that folder README).

## EfficientNetV2

From `efficientnetv2_metadata.json`: train 140, val 30, test 30 public labeled images (`cut`, `bruise`, `swelling`). Test accuracy 1.0 on this split **and** blank-image collapse — status `NOT_TRUSTWORTHY`.

## U-Net

From `unet_metadata.json`: train 140, val 30, test 30 public image/mask pairs. Held-out Dice ~0.98 **and** near-full masks on black/white — status `MODEL_OUTPUT_NOT_TRUSTWORTHY`.

## XGBoost / VQC

200 synthetic multimodal fusion vectors. 0 genuinely paired clinical samples. `DATA_PROVENANCE=SYNTHETIC`.

## Sensor

Canonical `v2.0.0-real`: SisFall (4505 trials) + UCI HAR (2000 subsampled windows), subject-level split, `data_provenance_class=REAL`. Synthetic v1.2.0 archived under `ml/models/_archive/`. Demo/simulate paths (`football_fall.csv`) remain generated vectors — not clinical telemetry.

## Licenses

See `data/datasets/DATASET_LICENSES.md` for source licenses. Follow those terms; this repo is academic research use.
