# EfficientNet retrain decision — 2026-08-29

Research prototype. Not clinical. Metrics below were computed from the live baseline checkpoint and a newly trained candidate. Negatives were not fabricated.

## Baseline

| Item | Value |
|---|---|
| Production path | `ml/models/vision/efficientnetv2_injury_best.pt` |
| SHA-256 | `6944605ae71685d909d505d12ce32d5bd9e953c10bbebb72a02635a729897e83` |
| Head | 3 classes `{cut, bruise, swelling}` |
| Status | **NOT_TRUSTWORTHY** |
| Training set claimed | `public_wound_dataset` (200 PIL drawings; abrasion/laceration/burn stored as swelling) |
| Unique pixel templates | cut 1, bruise 1, swelling 36 |
| Split | 140 / 30 / 30 with exact-pixel leakage across splits |
| Held-out accuracy | 1.0 on n=30 leaked copies (every max-softmax 1.0, ECE 0) |
| Raw OOD | uniforms and `blank_skin.jpg` / `dummy_test.jpg` collapse to swelling at 0.96–1.0 |
| Runtime gates | input-quality + T=1.5, min_conf 0.80, margin ≥ 0.20, entropy cap |

Forensic audit (`scratch/forensic_efficientnet_audit.json`) required retraining: do not train again on `public_wound_dataset` as-is.

## Dataset changes

Legitimate no-injury inventory on disk: **two files only**.

- `data/datasets/yolo_injury/blank_skin.jpg`
- `data/datasets/yolo_injury/dummy_test.jpg`

n=2 cannot support a `normal`/`reject` training class. Those files were recorded in `data/datasets/efficientnet_processed/ood_eval.csv` with `used_as_training_label: false`. No synthetic skin, noise, or background class was generated.

`efficientnet_processed` rebuild (`ml/training/prepare_efficientnet_processed_dataset.py`):

- Sources: unique-hash images from `yolo_processed` (cut/bruise), native public cut/bruise/swelling only (no abrasion/laceration/burn remap), `injury_dataset` unique cut/bruise/swelling.
- Exact pixel duplicates dropped. Splits are hash-disjoint (`leakage_free: true`).
- Class with fewer than 3 unique images dropped: **swelling unique count = 2**.
- Taxonomy written to `data/datasets/efficientnet_processed/taxonomy.json`.

| Class | Unique kept | Train | Val | Test |
|---|---|---|---|---|
| cut | 62 | 44 | 9 | 9 |
| bruise | 64 | 44 | 10 | 10 |
| swelling | dropped | — | — | — |

Total n=126 synthetic drawings. Provenance: SYNTHETIC. 99 public images dropped as not-remapped abrasion/laceration/burn.

Rejection strategy: **no trained reject class**. Runtime keeps input-quality gates plus closed-set confidence/margin/entropy rejection. Remaining limitation: photographs that pass the gates can still receive a confident injury label.

## Training changes

- Head-only EfficientNetV2-S (`tf_efficientnetv2_s.in21k_ft_in1k`), Adam 1e-3, class-weighted CE, batch 8, seed 42, patience 5, max 16 epochs.
- Early stop epoch 7; best val_loss at epoch 2.
- Candidate path: `ml/models/efficientnet_processed_training/efficientnetv2_candidate.pt`
- Candidate SHA-256: `0080565d653cbe4a9ff8f1e83f47a95fdc08c8e606673ecec64e4167b453a8a4`
- Candidate taxonomy: `{cut, bruise}` only.

Promotion rule used for the final decision: hash-disjoint splits, train loss decreased, held-out support for every trained class, candidate SHA ≠ production, **OOD injury-collapse count strictly lower than baseline, and remaining collapse count must be 0**. Accuracy alone does not promote. Status stays NOT_TRUSTWORTHY even if promoted.

## OOD results

Watch list collapse = injury argmax and max-softmax ≥ 0.95. Raw softmax T=1.0, no gates.

| Probe | Baseline (3-class) | Candidate (2-class) |
|---|---|---|
| gray | swelling 0.9624 collapse | cut 1.0 collapse |
| black | swelling 1.0 collapse | cut 1.0 collapse |
| white | swelling 1.0 collapse | cut 1.0 collapse |
| noisy_gray | bruise 0.9989 collapse | cut 1.0 collapse |
| uniform_skin | swelling 0.9998 collapse | cut 1.0 collapse |
| blue | swelling 1.0 collapse | cut 1.0 collapse |
| noise | bruise 1.0 collapse | cut 0.9076 (below 0.95) |
| blank_skin.jpg | swelling 0.9891 collapse | cut 1.0 collapse |
| dummy_test.jpg | swelling 1.0 collapse | cut 1.0 collapse |
| football_injury.jpg | swelling 0.9987 collapse | cut 1.0 collapse |
| qa_swelling_offcenter.jpg | swelling 0.8651 | cut 1.0 collapse |

Collapse count on the watch list: **baseline 9, candidate 8**. The single “improvement” is unstructured noise falling from 1.0 to 0.9076, still an injury class. Core blanks are **worse** (max-softmax 1.0 onto cut). Gates still withhold uniforms; they do not certify photos.

## Held-out results (candidate, unique-hash 2-class)

Val n=19 accuracy 1.0, loss 0.0. Test n=19 accuracy 1.0, loss 0.000713, ECE (10-bin) 0.000709, mean max-softmax 0.999291.

Confusion matrix labels `[cut, bruise]`:

```
[[9, 0],
 [0, 10]]
```

Per-class test: cut P/R/F1 = 1.0 support 9; bruise P/R/F1 = 1.0 support 10.

This is in-domain drawing discrimination after unique-hash splitting. It is not evidence of real-injury generalization. Calibration looks perfect because softmax is saturated.

Baseline held-out 1.0 on leaked 3-class copies is also not generalization.

## Candidate comparison

| | Baseline (production) | Candidate |
|---|---|---|
| SHA | `6944605a…` | `0080565d…` |
| Classes | cut, bruise, swelling | cut, bruise |
| Train n | 140 leaked drawings | 88 unique-hash drawings |
| Test acc | 1.0 (leaked) | 1.0 (unique drawings) |
| OOD collapse count | 9 | 8 |
| Core blanks | swelling ~1.0 | cut 1.0 |
| Reject class | none | none |
| Status | NOT_TRUSTWORTHY | NOT_TRUSTWORTHY |

Accuracy improved nothing that matters. OOD safety did not improve.

## Promotion decision

**KEEP_BASELINE.** Production restored to SHA `6944605ae71685d909d505d12ce32d5bd9e953c10bbebb72a02635a729897e83`.

An initial script pass treated 9→8 as promotion. That was reversed: remaining blanks still collapse; the count drop is a 0.95-threshold artifact on noise; a 2-class head would drop swelling from the runtime API without fixing OOD.

Candidate weights and `TRAINING_EVAL_REPORT.json` remain under `ml/models/efficientnet_processed_training/` for audit. They are not loaded at runtime.

## Final readiness

| Item | Result |
|---|---|
| Runtime EfficientNet | baseline 3-class SHA `6944605a…` |
| Status | **NOT_TRUSTWORTHY** |
| Frontend | Research registry Status column; `/api/models` `training_status`; case page `classifier_model_status` |
| Gates | kept (uniform/OOD/confidence). Not a clinical filter |
| What would change this | Legitimate labeled no-injury images sufficient to train a reject class, plus unique clinical photos; OOD collapse count 0 on blanks/unrelated inputs |

Do not treat test accuracy 1.0 as readiness.
