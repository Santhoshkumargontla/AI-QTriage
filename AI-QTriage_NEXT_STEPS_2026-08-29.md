# AI-QTriage next-steps pass — 2026-08-29

Research prototype only. No clinical claims.

## 1. Case-page section tabs
Exclusive panels: each of overview / image / questionnaire / sensor / ai / explainability / sos / report mounts only when selected (`isTab(...)`). `selectTab` scrolls to top. Frontend `tsc --noEmit` passed.

## 2. EfficientNet — real normal + subject-aware split
- Dataset: `data/datasets/efficientnet_subject_normal` (cut/bruise SYNTHETIC drawings + REAL AZH/Medetec empty / empty-region normals; subject-aware for normals).
- Best observed OOD injury-collapse: **9 → 2** (still NOT 0 → **KEEP_BASELINE**, production unchanged).
- Later flat-guard attempt worsened collapse (→3/4); reverted. Candidate left under `ml/models/efficientnet_subject_training/`.
- Swelling still omitted (only 2 unique drawings). Cut/bruise photos still blocked without Roboflow key.

## 3. U-Net — deduped subject-aware + blank-collapse gate
- Dataset: `data/datasets/unet_deduped_subject` (n=464, leakage-free).
- Blank/OOD collapse: **0** (gate passed). Test Dice ≈ **0.64**.
- **PROMOTED** to `ml/models/vision/unet_injury_best.pt`  
  SHA `82419176…` → `3c7f3f39…`  
  Backup: `.pre_deduped_subject_backup`

## 4. YOLO — wound boxes collected + retrain
- Dataset: `data/datasets/yolo_wound_boxes_v1` — **wound=1262** public-mask boxes + cut/bruise from retrain_v2.
- Test label support: wound **195**, cut 6, bruise 13 (wound was previously 0).
- Partial-epoch CPU checkpoint: blank_skin/dummy_test TN clear @ 0.25; aggregated test mAP50 ≈ 0.47.
- Temporary promote **reverted**: forensic OOD upload kept a `wound` box at conf≈0.251 under keep-threshold 0.25.
- Production SHA restored to `4d6e72f5…`. Candidate kept at `ml/models/yolo_wound_boxes_v1/run_wound_v1/weights/best.pt`.
- Honest limit: wound boxes are ulcer-domain envelopes, not sports-injury photos; cut/bruise still mostly drawings. Need more epochs + broader TN set before promote.


## 5. Fusion / clinical claim
- Cannot replace synthetic fusion labels (0 paired clinical multimodal records).
- Analyze + case GET now expose `clinical_claim_blocked`, `fusion_label_source=SYNTHETIC_RULE_LABELS`, `paired_clinical_samples=0`.
- Test: `backend/tests/test_clinical_claim_blocked.py`.

## Still blocked / next
- EfficientNet production remains Swelling-collapse until OOD collapse hits 0 with a trustworthy normal class.
- Real cut/bruise/wound **photographs** for YOLO need a valid Roboflow (or equivalent) key.
- Real paired clinician fusion labels required before any clinical accuracy claim.
