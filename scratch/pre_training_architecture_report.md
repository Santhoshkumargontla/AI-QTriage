# AI-QTriage Pre-Training Architecture Report

Generated: 2026-08-30T06:16:35.115271+00:00

## Runtime stack

1. **Frontend** (Next.js): create-case wizard → `/api/cases/{id}/analyze` → case detail tabs.
2. **Backend** (FastAPI `backend/main.py`): lazy singleton YOLO → U-Net ROI/full → EfficientNet → 23-d fusion → XGBoost; VQC experimental.
3. **MongoDB** collection `cases`: stores `visible_injury`, fusion outputs, sensor summary.

## Canonical model paths

- **YOLO11 Detection**: `ml\models\vision\yolo11_injury_best.pt` — exists=True, sha=4d6e72f5f671fd60…
- **EfficientNetV2 Classification**: `ml\models\vision\efficientnetv2_injury_best.pt` — exists=True, sha=95cf385d85419a63…
- **ResNet34-UNet Segmentation**: `ml\models\vision\unet_injury_best.pt` — exists=True, sha=3c7f3f39196d71b9…
- **XGBoost Multimodal**: `ml\models\xgboost_best.json` — exists=True, sha=73bb5a5125c3e390…
- **Experimental 4-Qubit VQC**: `ml\models\vqc\vqc_weights.npz` — exists=True, sha=2db769bec3abd3c2…
- **Sensor Motion Event Classifier**: `ml\models\sensor_motion_best.json` — exists=True, sha=707ee9c25392d5e8…

## Data layout

- Raw Kaggle: `data/raw/kaggle/` (yasinpratomo, ibrahimfateen, shubhambaid burn YOLO, fracture X-ray — not skin canonical).
- External segmentation: `data/datasets/external/` (wseg, Medetec, AZH).
- Processed YOLO: `data/datasets/yolo_retrain_v2` (synthetic), `yolo_real_skin_v2` (mask-derived wound + negatives).
- Processed EffNet: `data/datasets/efficientnet_kaggle_v1`.
- Processed U-Net: `data/datasets/unet_deduped_subject`.

## Promotion policy

Candidates under `ml/models/*_candidate*` or training run folders are **not** loaded until gates pass.
YOLO real_skin_v2 candidate was **KEEP_BASELINE** (failed gate1 + gate4).

## Blockers for real-data YOLO retrain

- Requires Roboflow Universe injury-detection exports (CC BY 4.0) with honest bounding boxes.
- `ROBOFLOW_API_KEY` not set in this environment.
- Classification-only Kaggle folders must **not** be converted to YOLO boxes.