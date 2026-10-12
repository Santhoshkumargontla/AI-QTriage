# AI-QTriage Phase 2 — Initial Issue Register

**Document:** `reports/phase2/initial_issue_register.md`  
**Generated:** 2026-10-12T17:55:00Z  
**Branch:** `main` | **HEAD:** `e8ecd2d59a6a1876fc5f1f241495034b1a29ba38`  

---

| Issue ID | Category | Severity | Description & Evidence | Impact / Risk | Corrective Action Planned | Status |
| :---: | :--- | :---: | :--- | :--- | :--- | :---: |
| **P2-01** | Evaluation Pipeline | **HIGH** | `ml/models/canonical_paths.py` defines `EVAL_HELD_OUT` pointing to legacy `data/results/canonical_held_out_evaluation.json` (Sept 2026, 16/30 VQC) rather than `canonical_held_out_evaluation_corrected.json` (Oct 2026, 17/30 VQC, leak-free). | Stale pre-remediation metrics may be served by API endpoints if fallback triggers. | Update `EVAL_HELD_OUT` in `canonical_paths.py` to point to the corrected canonical evaluation artifact. | **OPEN** |
| **P2-02** | Frontend & API Integration | **HIGH** | `/api/evaluation/ablation` reads from `data/results/evaluation_results.json` where `ablation_study` is an empty array `[]`, causing the Research page to miss the 7 independent ablation models. | Frontend Research Results page cannot render modality ablation curves from canonical source. | Update endpoint in `backend/main.py` to ingest `data/results/independent_ablation_results.json`. | **OPEN** |
| **P2-03** | Architectural Documentation | **MEDIUM** | Historical manuscript drafts labeled the sensor classifier as a "1D-CNN-LSTM", but the canonical production checkpoint on disk is `ml/models/sensor_motion_best.json` executed via `xgboost.XGBClassifier` on 7 kinematic features. | Reviewers may note contradiction between manuscript text and repository code. | Explicitly document XGBoost booster architecture on SisFall+UCI HAR summary features in all Phase 2 reports. | **OPEN** |
| **P2-04** | Vision Model Performance | **MEDIUM** | YOLO11 achieved 34.54% mAP@50 on held-out test photos, but Laceration class achieves 0.0% AP50 due to severe annotation sparsity in public sets. | Lacerations cannot be localized with high confidence by the current bounding box model. | Document honest research bounds; verify OOD/abstention behavior; do not claim clinical diagnostic readiness. | **OPEN** |
| **P2-05** | Sensor Benchmark Performance | **MEDIUM** | Sensor classifier achieves 96.34% accuracy on 1,228 test windows, but minority Impact class recall is 75.00% (15/60 impacts classified as fall). | Rapid stumble/fall transitions exhibit overlapping kinematic peak jerk and acceleration signatures. | Fully report class-wise confusion matrix; maintain explicit research prototype boundary. | **OPEN** |
| **P2-06** | Data Provenance & Realism | **CRITICAL** | Multimodal tabular triage vectors ($N=200$) are synthetically generated; 0 genuinely paired real human clinical cohorts exist. Labels derive from heuristic formula ($0.4\times\text{pain} + 0.3\times\text{peak\_g} + 0.3\times\text{area}$). | Model cannot be claimed to predict human clinical trauma outcomes. | Enforce `SYNTHETIC` provenance tags across all reports and UI headers; forbid claims of medical device clearance. | **OPEN** |
| **P2-07** | Quantum Machine Learning | **MEDIUM** | PennyLane 4-qubit VQC achieves 56.67% accuracy on classical `default.qubit` CPU simulator; does not outperform classical XGBoost (83.33%). | Misleading claims of quantum supremacy could be flagged by peer reviewers. | Keep VQC strictly designated as `EXPERIMENTAL_ONLY` and gated out of primary decision pipelines. | **OPEN** |

---

*This running register will be maintained across Phase 2 execution and finalized in `reports/phase2/phase2_issue_register.csv`.*
