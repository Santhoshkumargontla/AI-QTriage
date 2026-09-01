# AI-QTriage Kaggle Data Acquisition & Multi-Class Retrain Report

**Date:** 2026-08-29 / 2026-08-30  
**Workspace:** `C:\Users\santh\Capstone Project Code`  
**Auth:** Kaggle API token stored only under `%USERPROFILE%\.kaggle\access_token` (not in repo)  
**Device:** CPU (CUDA unavailable)

---

## Security notice

You pasted a live Kaggle token in chat. **Rotate/revoke that token in Kaggle account settings after this session.** It must never be committed to git.

Live API download verification hit SSL errors (`SSLEOFError` to `api.kaggle.com`). Training used **already-downloaded** local copies under `data/raw/kaggle/`.

---

## Executive verdict

| Item | Status |
| --- | --- |
| EfficientNet multi-class (abrasion/bruise/burn/cut/laceration/wound/normal/ood_reject) | **TRAINED + PROMOTED** |
| YOLO skin (cut/bruise/burn) | **TRAINED candidate; NOT promoted** (burn recall = 0) |
| YOLO fracture X-ray | Dataset ready; training started (separate artifact; **not** skin pipeline) |
| Swelling / edema | **NOT TRAINED** — no honest labeled skin-swelling set on disk |
| Clinical readiness | **Still NO** |

---

## Datasets used (local)

| Dataset | Path | Role | Notes |
| --- | --- | --- | --- |
| yasinpratomo/wound-dataset | `data/raw/kaggle/yasinpratomo_wound_dataset` | EffNet classes | Kaggle card: internet images; copyright caveats |
| ibrahimfateen/wound-classification | `data/raw/kaggle/ibrahimfateen_wound_classification` | EffNet classes | Large wound folders |
| shubhambaid/skin-burn | `data/raw/kaggle/shubhambaid_skin_burn` | YOLO burn boxes | CC0-1.0; degrees→single `burn` |
| pkdarabi bone fracture YOLO | `data/raw/kaggle/pkdarabi_bone_fracture_yolo` | Fracture X-ray | CC BY 4.0; **X-ray modality** |
| Existing `yolo_retrain_v2` | synthetic cut/bruise | YOLO base | Still mostly drawings |

### Prepared EffNet set `efficientnet_kaggle_v1`

- **n = 3039** (train 2104 / val 462 / test 473)
- Exact-hash cross-split leakage: **0**
- Class imbalance: **wound dominates** (train 1406)
- Swelling: `NOT_INCLUDED_NO_HONEST_LABELS`
- Fracture: excluded from this head (different modality)

### Prepared YOLO skin `yolo_skin_kaggle_v1`

- Boxes: cut 61, bruise 63, burn 330
- Images: train 205 / val 53 / test 44
- Wound/swelling boxes: **0**

---

## EfficientNet kaggle-v1 — PROMOTED

| Field | Value |
| --- | --- |
| Training | Real (`train_efficientnet_kaggle_v1.py`), 8 CPU epochs, warm-start from prior reject-v2 |
| Candidate/canonical SHA-256 | `95cf385d85419a63eeef8e46de9c9ef7e7487ef0f326fb60ca376b8bae0eec9f` |
| Classes | abrasion, bruise, burn, cut, laceration, wound, normal, ood_reject |
| Test n | **473** |
| Test accuracy | 0.964 |
| Test macro-F1 | **0.937** |
| OOD injury collapse | **0** |
| Blank/noise raw winners | **ood_reject** (verified post-promote) |
| Backup | `efficientnetv2_injury_best.pt.pre_kaggle_v1_backup` |
| Registry | version `kaggle-v1` |

### Honest caveats

- Wound class is majority → overall accuracy inflated vs rare classes.
- Cut support on test = **17** — not clinical proof.
- Kaggle licenses often “unknown” on cards — research use only.
- Metrics are **held-out research**, not clinical validation.

---

## YOLO skin cut/bruise/burn — NOT PROMOTED

| Field | Value |
| --- | --- |
| Candidate path | `ml/models/yolo_skin_kaggle_v1/run_v1/weights/best.pt` |
| Candidate SHA-256 | `d26b6b5626e02edb108d1544430141e77626a1e42d1325891d0bb73dc4a9c5da` |
| Test mAP50 | 0.623 (n=44 images) |
| Cut recall | 0.875 (8 boxes) |
| Bruise recall | 1.0 (10 boxes) |
| **Burn recall** | **0.0** (48 boxes) |
| Promote decision | **FALSE** — burn class dead; do not overwrite canonical |
| Canonical YOLO | Unchanged `4d6e72f5…` (cut/bruise/wound head) |

Burn failure mode: remapping all burn-degree labels to one class + severe class imbalance + short CPU schedule did not yield usable burn detections. Fix needs longer GPU train and/or class-balanced sampling — **not** silent promotion.

---

## Fracture X-ray

- Dataset prepared: 3631/348/169 images, multi-site fracture names.
- Artifact target: `ml/models/vision/yolo11_fracture_xray_best.pt`
- **Must never** be merged into skin-photo analyze path without modality routing.
- An ordinary phone photo **cannot** replace X-ray fracture diagnosis.

---

## What was not trained (and why)

| Requested class | Outcome | Reason |
| --- | --- | --- |
| Swelling / edema | **Skipped** | No honest labeled skin-swelling dataset available |
| Fracture (from skin photo) | **Skipped** | Only X-ray labels exist |
| Abrasion / laceration / burn / wound (classifier) | **Included in EffNet** | Folder labels from Kaggle wound sets |
| Abrasion / laceration / wound (YOLO boxes) | **Not added** | No honest YOLO boxes for those classes in this campaign |

---

## Runtime integration notes

1. Restart FastAPI so it loads the new 8-class EfficientNet + `*_classes.json`.
2. Questionnaire must not treat `normal` / `ood_reject` as injuries (already gated).
3. Keep YOLO canonical until a burn-capable (or cut-photo-improved) detector passes gates.
4. Do not claim SMS/clinical accuracy from this retrain.

---

## Evidence files

- `data/datasets/efficientnet_kaggle_v1/PREPARE_REPORT.json`
- `ml/models/efficientnet_kaggle_v1_training/TRAIN_REPORT.json`
- `ml/models/vision/efficientnetv2_metadata.json`
- `data/datasets/yolo_skin_kaggle_v1/PREPARE_REPORT.json`
- `ml/models/yolo_skin_kaggle_v1/TRAIN_REPORT.json`
- `data/datasets/yolo_fracture_xray_v1/PREPARE_REPORT.json`

---

## Next steps (recommended)

1. **Rotate Kaggle token.**
2. GPU train YOLO skin with balanced burn sampling; promote only if burn recall ≥ 0.15 and cut/bruise do not collapse.
3. Acquire a licensed **swelling/edema** set before reintroducing that class.
4. Wire fracture detector behind an explicit **X-ray modality** path only.
5. Prefer Roboflow cut/bruise **photo** detection sets if you obtain a Roboflow key (stronger for real-hand cut localization than synthetic YOLO alone).
