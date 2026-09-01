# U-Net forensic audit — 2026-08-29

Verify-first. Production weights were not modified. Metrics below were computed live from SHA `2b4967aa04f6af309d3aa14fef2e098350bdd02fd0e4575e11baa033eb0424a2`. Source: `scratch/forensic_unet_audit.json`.

Research prototype. Not clinical. Correct mask geometry on leaked drawings is not reliability.

## Raw OOD behavior

Raw path: resize to 256×256 (`cv2` default INTER_LINEAR), ImageNet z-score, sigmoid, threshold 0.5. **Gates were not used to compute these numbers.** Overlay status is recorded separately. False-positive area on probes without a ground-truth mask equals the positive-mask ratio.

Production input gate is `assess_input_quality` (uniform if **std &lt; 12 and ptp &lt; 40**, plus other checks). It is **not** std &lt; 3. Several uniforms fire even when std is 0.

| Probe | Pos. ratio | Max p | Mean p | FP area | Mask hw | After gate |
|---|---|---|---|---|---|---|
| gray 180 | 0.1575 | 0.9210 | 0.1398 | 0.1575 | 256×256 | LOW_QUALITY withheld |
| black | 0.9964 | 1.0000 | 0.9972 | 0.9964 | 256×256 | LOW_QUALITY withheld |
| white | 0.9995 | 1.0000 | 0.9970 | 0.9995 | 256×256 | LOW_QUALITY withheld |
| blank mid-gray 128 | 0.0000 | 0.1094 | 0.0021 | 0.0000 | 256×256 | LOW_QUALITY withheld |
| blank_skin.jpg | 0.4291 | 0.9978 | 0.4774 | 0.4291 | 256×256 | LOW_QUALITY withheld |
| dummy_test.jpg | 0.9964 | 1.0000 | 0.9972 | 0.9964 | 256×256 | LOW_QUALITY withheld |
| uniform skin (185,145,125) | 0.0000 | 0.1066 | 0.0021 | 0.0000 | 256×256 | LOW_QUALITY withheld |
| blue (unrelated) | 0.9961 | 1.0000 | 0.9963 | 0.9961 | 256×256 | LOW_QUALITY withheld |
| green (unrelated) | 0.0000 | 0.1399 | 0.0009 | 0.0000 | 256×256 | LOW_QUALITY withheld |
| high-frequency noise | 0.0344 | 0.9915 | 0.0749 | 0.0344 | 256×256 | UNTRUSTWORTHY withheld |
| football_injury.jpg (photo, no GT) | 0.4330 | 0.9969 | 0.4193 | 0.4330 | 256×256 | **VALID displayed** |
| held-out cut drawing (with GT) | 0.0368 | 1.0000 | 0.0393 | 0.0017 vs GT | 256×256 | VALID displayed |

Black/white/blue saturate. The training-background skin fill does not. The demo photo receives a **displayed** mask covering 43% of the frame. That is not withheld by std&lt;12 because the photo has structure.

## Dataset composition

Training set of the live checkpoint (metadata `dataset_name`: `public_wound_dataset`):

- 200 PIL drawings from `generate_expanded_wound_dataset`, not downloaded Kaggle/Roboflow/WOUNDSEG images. Manifest `source` field is false.
- Visual classes include abrasion/laceration/burn; those files are stored as class `swelling`.
- Counts: cut 34, bruise 34, swelling 132. Split 140 / 30 / 30.
- Unique pixel templates: **cut 1, bruise 1, swelling 36** (38 unique images total).
- Mask area: min 0.025191, mean 0.139541, max 0.268435. Every mask is non-empty.
- Image and mask dimensions match (224×224).

`data/datasets/unet_processed` exists (126 pairs, 85 generated empty masks). That set was **not** used to train this SHA. Those empties are synthetic textures, not a legitimate clinical no-injury corpus.

## Empty-mask availability

**Zero** empty masks in the production training set. **Zero** labeled no-wound pairs.

On disk, only two existing no-injury files (`blank_skin.jpg`, `dummy_test.jpg`) were used here as **eval-only** probes. They were not training labels. n=2 cannot train a background class.

## Preprocessing audit

| Stage | What happens |
|---|---|
| Train image | `cv2.resize` 256×256 INTER_LINEAR, RGB, `/255`, ImageNet mean/std. No augmentation. |
| Train mask | `cv2.resize` 256×256 **INTER_NEAREST**, binary `>127`. |
| Inference | Same image resize and ImageNet z-score. Sigmoid, fixed threshold **0.5**. |
| Gated overlay | 3×3 morphological opening; keep largest CC if ≥80% of positives; withhold if area &gt; 0.70 or input quality fails. Back-project with INTER_NEAREST. |
| Encoder | ResNet34 ImageNet init, frozen in the production train loop. Decoder + segmentation head trained with BCE+Dice. |

ImageNet z-score of uniforms: black ≈ (−2.12, −2.04, −1.80); white ≈ (2.25, 2.43, 2.64); training skin (185,145,125) ≈ (1.05, 0.50, 0.37).

Last-layer bias is **−0.075** (not a large positive bias). Collapse is feature-space OOD, not a stuck-on bias.

## Leakage audit

- Exact duplicate **image** groups: 5 (167 duplicate files).
- Exact duplicate **mask** groups spanning splits: 5.
- The single cut drawing appears in train, val, and test (34 copies). Same for bruise.
- Subject IDs are disjoint across splits. That does **not** prevent pixel leakage; subject IDs are generator labels.
- `leakage_free`: **false**.

## Held-out metrics (live recompute)

Same 256 / ImageNet / 0.5 pipeline as training.

| Set | n | Dice | IoU | Precision | Recall | Mean FP area |
|---|---|---|---|---|---|---|
| Advertised test (leaked) | 30 | 0.982084 | 0.965434 | 0.986665 | 0.977954 | 0.000901 |
| Unique pixel templates | 10 | 0.968878 | 0.940490 | 0.985750 | 0.953265 | 0.000700 |

Leaked n=30 matches `unet_metadata.json` test metrics exactly. Unique n=10 is still only generator templates. High Dice is reconstruction of the drawings, not generalization to photographs.

## Root cause

The model predicts near-full positive masks on black and white because:

1. Every training target is a non-empty blob on a skin-tone canvas. Dice+BCE never saw an all-zero mask.
2. Black and white (and solid blue) are far from that skin-tone mode after ImageNet normalization, so encoder activations are OOD.
3. The decoder’s only trained behavior on “not the skin canvas” is “there is foreground.” Logits go large and positive almost everywhere; sigmoid ≈ 1.
4. Uniform skin at the exact training fill stays empty, which shows the failure is mode-specific, not “all constant images.”
5. Overlay gates hide black/white from the UI; they do not change the raw map. `football_injury.jpg` has enough texture to pass the gate and still gets a 43% VALID mask.

Mask geometry on the leaked cut stroke can look correct (held-out cut positive ratio 0.0368, FP area vs GT 0.0017) while OOD and photos remain untrustworthy. **Geometry and reliability are separate.**

## Retraining recommendation

Do **not** retrain on `public_wound_dataset` as-is. Do **not** promote because leaked Dice is high.

Do **not** invent a large empty class from random fills and call it clinical negatives. The only legitimate no-injury files found are two eval-only images.

A candidate should stay off production until **raw** positive ratio on black, white, gray, and `blank_skin.jpg` is &lt; 0.05 **without** gates, splits are hash-disjoint, and unique-template Dice is reported separately from leaked Dice. Keep overlay gates until then. Keep status `MODEL_OUTPUT_NOT_TRUSTWORTHY`.

`unet_processed` (85 generated empties) is a later synthetic experiment, not evidence that the live SHA was trained on negatives.

## Final status

**MODEL_OUTPUT_NOT_TRUSTWORTHY**

| Item | Value |
|---|---|
| Production SHA | `2b4967aa04f6af309d3aa14fef2e098350bdd02fd0e4575e11baa033eb0424a2` |
| Matches metadata / registry | yes |
| Production weights changed this audit | no |
| Gates | keep (withhold uniforms; they do not certify photos) |
| Readiness | not ready; not clinical |
