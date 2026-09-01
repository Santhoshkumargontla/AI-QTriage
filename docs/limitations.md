# AI-QTriage System Limitations & Disclosures

Research prototype only. Not a medical diagnostic system. Not FDA-cleared. Not for real emergency dispatch.

## Simulation features

- Wizard **Simulate Sensor Data** and **Use Demo Log** are synthetic. They do not represent a real person's accident.
- **SOS countdown** is recorded in MongoDB. SMS is not sent unless Twilio is explicitly configured for a test destination. Real emergency services (911 / 112 / ambulance) are never contacted.
- **Twilio Test / Sandbox** still stays `TWILIO_NOT_CONFIGURED` when `TWILIO_ENABLED=false`.

## Synthetic datasets

- XGBoost and VQC: 200 synthetic 23-feature fusion samples, split 140 / 30 / 30. Labels are rule-derived research categories (LOW / MODERATE / HIGH), not clinical severity.
- Demo photograph: `data/sample/image/football_injury.jpg`.
- Sensor simulate scenarios: `football_fall`, `sudden_fall`, `sudden_impact`, `normal_movement`.

## Experimental models

- **VQC**: PennyLane `default.qubit` classical simulator. No quantum advantage. Status `EXPERIMENTAL_ONLY`. Isolated from SOS, first-aid, and the main case decision.
- **Grad-CAM**: model visualization only. UI label is **NOT CLINICAL EXPLANATION**.

## Model limitations (runtime)

| Model | Limitation |
| --- | --- |
| YOLO11 | Skin classes: cut, bruise, abrasion, burn, wound, laceration (`expanded-skin-v1`). Fracture is a separate X-ray model only — not used on phone photos. Normal/OOD_Reject are empty-label negatives, not box classes. Laceration has very few boxes. |
| EfficientNetV2 | Blank/uniform inputs collapse to swelling in raw softmax. Gates withhold; not a clinical filter. |
| U-Net | Black/white inputs produce large raw positive masks. Overlay withheld. ROI mask may not align to the full photo. |
| XGBoost | Synthetic fusion only. HIGH-class test support is 3. |
| Sensor | Simulated fall may classify as `normal_activity`. Missing columns are `FEATURE_MISSING`. |
| First-aid | Gemini timeout or invalid key → rule-based fallback. |

RGB photos cannot determine fractures or internal injuries.

## Environment variables

See the root `README.md` and `backend/.env.example`. Never put Twilio tokens or Gemini keys in the frontend or in exported PDF/JSON.
