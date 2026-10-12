# Phase 3 Handoff & Manuscript Revision Guidance

**Target:** AI-QTriage Research Manuscript Preparation & Academic Defense  
**Auditor:** Senior Research Reproducibility Auditor & Scientific Writer  

---

## 1. Key Narrative Corrections Required in the Manuscript

| Manuscript Section | Previous Inaccurate / Stale Claim | Required Phase 3 Empirical Statement |
| :--- | :--- | :--- |
| **Abstract / Intro** | "Multimodal triage validated on paired real emergency patients." | State clearly that vision and sensor perception models are trained on real human datasets, while multimodal fusion triage is validated on an **experimental synthetic benchmark** due to the absence of open-access synchronized trauma telemetry. |
| **Methods: Perception** | "YOLO11 achieves balanced multi-class injury detection across all trauma types." | Accurately report that YOLO11 achieves 34.54% mAP@50 overall, performing strongly on cuts (50.67%), bruises (60.08%), and abrasions (61.03%), while lacerations (0.00% AP50) are limited by dataset representation sparsity. |
| **Methods: Sensor** | "Sensor fall detection powered by an end-to-end 1D-CNN-LSTM deep neural network." | Correct architecture to an **XGBoost gradient-boosted tree classifier** trained on 7 engineered kinematic summary features from 6,505 real SisFall and UCI HAR windows, achieving 96.34% accuracy. |
| **Results: Quantum ML** | "4-Qubit VQC achieves clinical triage parity or quantum advantage." | State transparently that the 4-qubit VQC achieves **56.67% accuracy** (17 / 30 correct) on held-out test data, demonstrating current expressivity limitations of NISQ-era quantum circuits relative to classical XGBoost (83.33%). Emphasize its role as an **exploratory proof-of-concept**. |
| **Results: Ablation** | "Full 23-feature model strictly outperforms all sub-modalities." | Present the empirical ablation finding: **Questionnaire + Sensor achieves the highest accuracy (90.00%)**, whereas adding vision features (83.33%) introduces mild score variance while enhancing localized wound localization. |

---

## 2. Research Assets Available for Publication
1. `reports/phase2/dataset_inventory.csv`: Complete metadata and open-access licensing for all 5 datasets.
2. `reports/phase2/leakage_audit.json`: Mathematical proof of 0 cross-split leakage.
3. `data/results/phase2_baseline_evaluation.json`: Exact per-class precision, recall, and confusion matrices across all 6 models.
4. `data/results/phase2_modality_ablation.json`: Complete 7-configuration ablation benchmark.
