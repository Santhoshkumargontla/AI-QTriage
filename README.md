# AI-QTriage

Research prototype for academic evaluation of a multimodal injury-assessment pipeline (vision + questionnaire + optional smartphone sensors + experimental quantum classifier + local SOS simulation).

**Not a medical device. Not clinically validated. Not for diagnosis, triage, or real emergency dispatch.** An ordinary photograph cannot identify fractures or internal injuries. No 911 / 112 / ambulance contact is ever made.

Current honest status after the 2026-08-29 cleanup: **PARTIALLY_WORKING research demo**. See `ml/models/model_registry.json` for SHA-256, classes, provenance, and known limitations.

## Runtime models (canonical only)

| Model | Canonical path | Current Verified Status | Notes / Limitations |
| --- | --- | --- | --- |
| YOLO11 detect | `ml/models/vision/yolo11_injury_best.pt` | INFERENCE_EXECUTES (Research Baseline) | Real photographic wounds (`yolo_expanded_v1`, 2,813 images); 34.54% mAP@50, 16.74% mAP@50–95. |
| EfficientNetV2 | `ml/models/vision/efficientnetv2_injury_best.pt` | READY_FOR_RESEARCH_DEMO | Retrained on Kaggle multi-class photos (`kaggle-v1`, 473 test); 96.62% acc, 94.34% F1. Blank/OOD collapse gates pass. |
| ResNet34-UNet | `ml/models/vision/unet_injury_best.pt` | READY_FOR_RESEARCH_DEMO | Retrained on AZH+WSEG+Medetec (`deduped-subject-v1`, 69 test); 64.18% Dice, 51.78% IoU at $\tau=0.50$. Blank gates pass. |
| XGBoost | `ml/models/xgboost_best.json` | TRAINED (Research Feasibility) | Retrained on leak-free grouped split (200 vectors, seed 42); 83.33% acc (25/30), 78.50% macro-F1, MCC 0.7027. |
| VQC | `ml/models/vqc/vqc_weights.npz` | EXPERIMENTAL_ONLY | Retrained on same leak-free split; PennyLane `default.qubit` CPU simulation; 56.67% acc (17/30). No quantum advantage. |
| Sensor motion | `ml/models/sensor_motion_best.json` | TRAINED (Real Benchmark) | SisFall + UCI HAR (6,505 windows, 68 subjects); 96.34% acc (1,183/1,228), 89.85% macro-F1, MCC 0.9296. |

*Historical Note:* Earlier August 2026 evaluations marked EfficientNet/U-Net as `NOT_TRUSTWORTHY` due to blank-image collapse on naive drawing datasets. Subsequent retraining on photographic datasets with negative controls and OOD rejection resolved the collapse mode while preserving runtime confidence gates. All models remain academic research prototypes and are not clinically validated.

## Synthetic data, simulation, experimental pieces

- **Demo image** `data/sample/image/football_injury.jpg` is synthetic demonstration art.
- **Sensor simulate** (`football_fall`, `sudden_fall`, `sudden_impact`, `normal_movement`) generates kinematic vectors, then runs the same feature pipeline as an upload.
- **XGBoost / VQC** train on 200 synthetic multimodal fusion samples (0 genuinely paired clinical records).
- **VQC** is experimental only: excluded from SOS, first-aid, and the main case decision.
- **SOS** countdown is a local MongoDB simulation. Twilio, when enabled, is test/sandbox SMS to configured numbers — never emergency services.
- **First-aid LLM** (Gemini) falls back to rule-based research guidance on timeout or invalid key.

## Quick start

```bash
# Backend (venv already in backend/venv)
backend\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

MongoDB must be reachable (`MONGODB_URI`). Open `http://localhost:3000`. API defaults to `http://localhost:8000` (`NEXT_PUBLIC_API_URL`).

Copy `backend/.env.example` to `backend/.env`.

## Required / optional environment variables

Canonical names (see `backend/.env.example`):

| Variable | Required | Purpose |
| --- | --- | --- |
| `MONGODB_URI` | yes | Database |
| `MONGODB_DATABASE` | no | Default `ai_qtriage` |
| `GEMINI_API_KEY` | no | First-aid LLM; missing/invalid → rule-based fallback |
| `GEMINI_MODEL` | no | Default `gemini-flash-latest` |
| `TWILIO_ENABLED` | no | Default false. Must be true **and** credentials set to leave `TWILIO_NOT_CONFIGURED` |
| `TWILIO_ACCOUNT_SID` | if Twilio | Twilio account |
| `TWILIO_AUTH_TOKEN` | if Twilio | Secret; never returned by API |
| `TWILIO_FROM_NUMBER` | if Twilio | From |
| `TWILIO_TO_NUMBER` | if Twilio | Test destination only |
| `YOLO_CONF_THRESHOLD` | no | Box keep-threshold, default 0.25 (not 0.10) |
| `YOLO_LOW_CONFIDENCE_FLAG` | no | Display flag only, default 0.40 |
| `EFFNET_MIN_CONFIDENCE` | no | Classifier reject, default 0.80 |
| `EFFNET_TEMPERATURE` | no | Softmax temperature, default 1.5 |
| `SOS_DISPLAY_TIMEZONE` | no | Compact SOS timestamps, default `Asia/Kolkata` |
| `ALLOWED_ORIGINS` | no | CORS origins, default `http://localhost:3000,http://127.0.0.1:3000` |
| `QUESTIONNAIRE_ROUTING_THRESHOLD` | no | Template routing only, default 0.40. Not a medical threshold |
| `NEXT_PUBLIC_API_URL` | frontend | Default `http://localhost:8000` |

Twilio outcomes are only `LOCAL_SIMULATION`, `TWILIO_NOT_CONFIGURED`, `TWILIO_REQUEST_QUEUED`, `TWILIO_FAILED`. A SID means the API queued a request, not that a handset received SMS.

## Tests

```bash
backend\venv\Scripts\python.exe -m pytest backend/tests -q
cd frontend && npm run lint && npm run build
```

## Documentation

- `docs/limitations.md` — medical and technical boundaries
- `docs/datasets.md` — actual sample counts used in training (not marketing figures)
- `ml/models/_archive/README.md` — unused duplicate checkpoints
- `audit/` — historical forensic notes; some older SHA/path claims are superseded by `model_registry.json`
