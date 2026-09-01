# AI-QTriage U-Net retrain report

**Date:** 2026-08-29  
**Promotion:** KEEP_BASELINE  
**Production SHA-256 (unchanged):** `2b4967aa04f6af309d3aa14fef2e098350bdd02fd0e4575e11baa033eb0424a2`  
**Candidate SHA-256 (not loaded):** `2d7c37d9298525d8925ca77730cbc7f9283670591ba976db6056077dc8562362`  
**Training status:** `MODEL_OUTPUT_NOT_TRUSTWORTHY`  
**FINAL STATUS:** `REQUIRES_MORE_DATA`

Not `VERIFIED_FIXED`. Not promoted. Overlay gates remain required.

Evidence: `ml/models/unet_processed_training/TRAINING_EVAL_REPORT.json`, `PROMOTION.json`.

---

## Verify-first inventory

Legitimate no-injury files on disk:

| File | Role |
|---|---|
| `data/datasets/yolo_injury/blank_skin.jpg` | eval-only empty GT |
| `data/datasets/yolo_injury/dummy_test.jpg` | eval-only empty GT |

n=2. Too few to train/validate/test an empty-mask class without leakage. They were **not** used as training labels.

YOLO empty-label images are drawings that still contain wounds with boxes dropped. Converting those to U-Net empty masks would be fake empty medical masks. They were not used.

Generated empty textures (`_make_empty_images`) were removed. No fake medical masks were created.

---

## Dataset correction (concrete, before retrain)

`ml/training/prepare_unet_processed_dataset.py` now:

- keeps unique pixel hashes from `public_wound_dataset` + `injury_dataset`
- hash-disjoint train/val/test (29 / 6 / 6)
- 189 duplicate/invalid exclusions
- 0 trained negatives
- writes `blank_skin` / `dummy_test` to `ood_eval.csv` with `used_as_training_label: false`
- provenance `SYNTHETIC`

Unique class counts after hash collapse: swelling 37, cut 2, bruise 2. Binary wound mask only.

---

## Loss / threshold / preprocess audit

| Item | Decision |
|---|---|
| Loss | Unweighted BCE + **per-image** soft Dice |
| BCE `pos_weight` | **1.0** (not inverse-frequency). Train FG area 0.0488; boosting FG would worsen blank painting with no empty targets |
| Dice | Empty-safe per image; no empty targets existed in train |
| Threshold | **0.5** (same as production). Val sweep 0.3–0.7 reported; not used to hide OOD painting |
| Resize | Image 256 `INTER_LINEAR`, mask `INTER_NEAREST` |
| Normalization | ImageNet z-score (0.485/0.456/0.406, 0.229/0.224/0.225) |
| Encoder | ResNet34 ImageNet frozen; decoder trained; 16 epochs; train loss 1.4627 → 0.0841 |

---

## Held-out injury (unique-hash test n=6, T=0.5)

| Metric | Baseline (production) | Candidate |
|---|---:|---:|
| Dice | 0.953060 | 0.970102 |
| IoU | 0.910365 | 0.942456 |
| Precision | 0.979929 | 0.967817 |
| Recall | 0.928548 | 0.973104 |
| False-positive area | 0.000646 | 0.003245 |
| Positive mask ratio | 0.058004 | 0.064774 |

In-domain Dice improved. That is **not** promotion.

Production leaked-split metadata Dice 0.982084 (n=30) is generator reconstruction, not this unique-hash test.

---

## Held-out no-injury (existing files, empty GT)

| Image | Baseline pos ratio / FP area | Candidate pos ratio / FP area |
|---|---:|---:|
| blank_skin.jpg | 0.429062 | 0.991501 |
| dummy_test.jpg | 0.996429 | 0.996841 |
| mean | 0.712746 | 0.994171 |

Candidate paints **more** of blank_skin, not less. Dice/IoU/Precision = 0 on both (empty GT, non-empty pred).

---

## Raw OOD probes (256×256, T=0.5, no gates)

Positive ratio = false-positive area when there is no wound.

| Probe | Baseline | Candidate |
|---|---:|---:|
| black | 0.9964 | 0.9968 |
| white | 0.9995 | 0.9999 |
| gray 180 | 0.1575 | 0.9727 |
| mid-gray 128 | 0.0000 | 0.0000 |
| blank_skin | 0.4291 | 0.9915 |
| dummy_test | 0.9964 | 0.9968 |
| blue (unrelated) | 0.9961 | 0.9959 |
| green (unrelated) | 0.0000 | 0.0000 |
| noise (unrelated) | 0.0015 | 0.2755 |
| football_injury.jpg | 0.4330 | 0.8210 |

Core collapse count (pos ratio > 0.05 on black/white/gray/mid_gray/blank_skin/dummy_test): **5 → 5**. Gray and blank_skin got worse.

---

## Promotion rule (failed)

Promote only if: hash-disjoint splits, train loss decreased, candidate SHA ≠ production, core OOD strictly better, **and remaining core collapse count = 0**.

Result: `KEEP_BASELINE`. Reasons: OOD not improved; still paints black/white/gray/blank_skin/dummy_test; empty-mask class not fabricated.

Production `unet_injury_best.pt` and `unet_metadata.json` were not overwritten.

---

## Frontend

Production U-Net still `MODEL_OUTPUT_NOT_TRUSTWORTHY`. Case page now shows this as **training** status (`segmentation_model_status`), separate from the prediction gate (`VALID` / `LOW_QUALITY`). Research `/api/models` includes U-Net `training_status`. Overlay gates unchanged.

---

## FINAL STATUS

**REQUIRES_MORE_DATA**

Also remains **NOT_TRUSTWORTHY** / `MODEL_OUTPUT_NOT_TRUSTWORTHY`.

Honest unique-hash retraining happened. Blank painting did not stop. Fixing it requires a real labeled no-injury / empty-mask set large enough for independent train/val/test — not two eval files and not generated textures.
