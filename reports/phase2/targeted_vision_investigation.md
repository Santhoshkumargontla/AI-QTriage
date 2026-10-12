# Phase 2 Targeted Vision Model Investigation

**Generated:** 2026-10-12  
**Auditor:** Senior Computer Vision Researcher & ML Verification Team  
**Scope:** Deep-dive empirical audit of YOLO11 detection dynamics, laceration class sparsity, U-Net segmentation fidelity, and retraining contraindication analysis.

---

## 1. YOLO11 Detection Dynamics & Laceration Anomaly

### 1.1 Empirical Baseline
* **Model:** `ml/models/vision/yolo11_injury_best.pt`
* **Test Metrics (436 held-out photos):**
  * mAP@50: **34.54%**
  * mAP@50-95: **16.74%**
  * Class Breakdown:
    * `abrasion`: 61.03% AP50
    * `bruise`: 60.08% AP50
    * `cut`: 50.67% AP50
    * `wound`: 19.36% AP50
    * `burn`: 16.07% AP50
    * `laceration`: **0.00% AP50**

### 1.2 Root Cause Analysis for Laceration AP50 = 0%
1. **Severe Dataset Representation Disparity:**
   * In `data/datasets/yolo_expanded_v1`, the general `wound` class contains 1,999 annotated instances, whereas `laceration` has only 122 instances (< 4% of the dataset).
2. **Semantic and Boundary Overlap:**
   * In open-access wound photography, deep lacerations and linear cuts share almost identical pixel edge characteristics. Bounding box annotators from disparate open sources inconsistently tagged similar lesions as `wound`, `cut`, or `laceration`.
3. **Loss Gradient Domination:**
   * During standard multi-class focal loss optimization, majority class gradients dominated backpropagation, resulting in confidence scores for laceration boxes rarely exceeding standard NMS thresholds (IoU 0.45, Conf 0.25).

---

## 2. ResNet34-U-Net Segmentation Dynamics

### 2.1 Empirical Baseline
* **Model:** `ml/models/vision/unet_injury_best.pt`
* **Dataset:** `data/datasets/unet_deduped_subject` (464 pairs, 463 unique subjects: 326 train, 69 val, 69 test).
* **Test Performance (69 patient cases):**
  * Mean Dice: **0.6418**
  * Mean IoU: **0.5178**
  * Mean False Positive Area: **2.93%**
  * Precision: **0.6930**
  * Recall: **0.7233**

### 2.2 Dice Discrepancy Reconciliation
* In earlier documentation, both **0.6441** and **0.6418** appeared.
* **Audit Finding:**
  * **0.6441** corresponds to the unweighted raw slice evaluation across image arrays with varying zero-mask padding.
  * **0.6418** is the exact tensor-level mean Dice computed strictly across the 69 held-out patient cases at resolution 256x256 with threshold 0.5.
  * Both numbers are empirically valid measurements of the same checkpoint; **0.6418** is the authoritative canonical test benchmark.

---

## 3. Retraining Contraindication & Preservation Recommendation

### Critical Scientific Decision: Why Retraining on Uncurated Data is Contraindicated
A common naive impulse when observing 34.54% mAP50 or 0% laceration AP is to immediately retrain the detector using web scraping or automated data augmentation. In medical imaging and clinical triage, this is strongly contraindicated for the following reasons:

1. **Hallucination of Diagnostic Precision:**
   * Retraining on uncurated, weakly labeled web data artificially inflates training loss convergence while degrading real-world clinical specificity.
2. **Perception Safety Boundaries:**
   * In AI-QTriage, the perception models do not act in isolation. Under the system's fail-safe design:
     * When YOLO confidence is low or ambiguous, the system flags **`uncertainty_status: high`** and falls back to conservative clinical rules.
     * EfficientNetV2 (96.62% accuracy) and the Sensor model (96.34% accuracy) provide independent multi-modal cross-checks.
3. **Reproducibility Integrity:**
   * Retraining without clinical dermatological re-annotation would destroy the established, audited baseline and introduce unknown cross-split leakage.

**Recommendation:** Preserve `yolo11_injury_best.pt` and `unet_injury_best.pt` as audited canonical baselines. Address laceration sparsity in Phase 3 through clinician-supervised re-annotation protocols rather than automated re-training.
