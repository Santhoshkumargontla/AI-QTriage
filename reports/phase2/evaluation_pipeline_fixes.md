# Phase 2 Evaluation Pipeline Fixes Report

**Generated:** 2026-10-12  
**Auditor:** Senior Research Reproducibility Auditor & Full-Stack Verification Engineer  
**Scope:** Single source of truth reconciliation for evaluation artifacts and REST API endpoints.

---

## 1. Summary of Systemic Issues Identified & Resolved

### Issue A: Outdated Canonical Path Pointer (`EVAL_HELD_OUT`)
* **Problem:** In `ml/models/canonical_paths.py`, `EVAL_HELD_OUT` pointed to `data/results/canonical_held_out_evaluation.json`, an obsolete artifact from September 2026 that recorded outdated VQC performance (16/30 correct) before leak-free grouping.
* **Resolution:** 
  * Updated `EVAL_HELD_OUT` to `data/results/canonical_held_out_evaluation_corrected.json`.
  * Added `EVAL_INDEPENDENT_ABLATION` constant pointing to `data/results/independent_ablation_results.json`.
* **Verification:** Confirmed both artifacts exist on disk and contain zero cross-split leakage.

### Issue B: Truncated Ablation Study in REST API (`/api/evaluation/ablation`)
* **Problem:** The endpoint loaded exclusively from `evaluation_results.json`, which had `ablation_study: []`. As a consequence, frontend clients saw empty ablation tables.
* **Resolution:** 
  * Enhanced `get_ablation_benchmarks()` in `backend/main.py` to check `resolve_existing(EVAL_INDEPENDENT_ABLATION)` first.
  * Corrected JSON structure parsing to return the complete 7-configuration ablation array and exact 30 held-out test sample count.
* **Verification:** Tested endpoint directly via `TestClient`. Confirmed HTTP 200 response returning all 7 independent modality ablation configurations.

### Issue C: Schema Discrepancy in Model Comparison Endpoint (`/api/evaluation/comparison`)
* **Problem:** Helper function `_comparison_api_payload()` expected nested dictionary key `xgboost`, whereas the canonical corrected artifact utilizes `xgboost_evaluation` and `vqc_evaluation`. This resulted in `accuracy: None` and `sample_count: None` responses.
* **Resolution:** Updated `_comparison_api_payload()` in `backend/main.py` to support `xgboost_evaluation`, `vqc_evaluation`, and test split lengths dynamically.
* **Verification:** Tested endpoint directly. Verified live returned values:
  * Sample count: **30**
  * Classical XGBoost Accuracy: **83.33% (25 / 30)**
  * Quantum VQC Accuracy: **56.67% (17 / 30)**
  * Status: **evaluated**

---

## 2. Regression Test Coverage

Added integration tests ensuring:
1. `GET /api/evaluation/comparison` never returns `null` or `None` for accuracy, precision, recall, or sample count when canonical artifacts are present.
2. `GET /api/evaluation/ablation` returns exactly 7 experimental modality configurations with active feature counts from 6 to 23.
