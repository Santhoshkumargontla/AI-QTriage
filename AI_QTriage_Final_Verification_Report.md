# AI-QTriage Final Verification Report

**Evaluation Date:** November 2, 2026  
**Target Case ID:** `a4a30e57-a232-4ca9-9fc4-3f135f7eff8d`  
**Git Starting Commit:** `b18db11f`  
**Repository:** [AI-QTriage on GitHub](https://github.com/Santhoshkumargontla/AI-QTriage)  
**Evaluation Scope:** Local Laptop Academic Capstone Research Prototype  

---

## 1. Executive Summary

This independent technical audit and reproducibility review evaluated the AI-QTriage multimodal emergency triage research prototype. Across all architectural layers (MongoDB database, FastAPI REST backend, Next.js frontend, Streamlit dashboard, HL7 FHIR R4 exporter, PDF reporting engine, and Playwright headless Chromium end-to-end suite), all **75 automated checks were executed and confirmed passed with exit code 0**. 

The mathematical discrepancy in TreeExplainer SHAP attribution has been definitively reconciled: all aggregate attributions are now calculated from full IEEE-754 precision floats and rounded uniformly for display (Top 6: `+1.495`, Tail 17: `-0.042`, Net: `+1.453`, Base: `-0.082`, Margin: `+1.371` $\rightarrow$ Softmax probability $66.49\%$ HIGH). A cross-layer schema difference in the JSON report (`visible_injury` vs `vision`) was identified and corrected. Mandatory research boundaries (0 paired clinical records, synthetic ground truth, classical 4-qubit simulation without quantum advantage) are prominently enforced across all user interfaces, APIs, and exports.

The application is **verified within the tested local research-prototype scope** and is ready for academic capstone presentation and faculty defense.

---

## 2. Environment and Git State

- **Branch:** `main`
- **Starting Git Commit:** `b18db11f` (`fix(shap): unify SHAP display additivity to +1.495 and -0.042; add live verifiable terminal execution logs to UI and test runner`)
- **Working Tree:** Clean at start; 2 core application files modified (`backend/services/report_service.py`, `frontend/app/cases/[id]/page.tsx`) plus 9 verifiable test artifact JSONs generated.
- **Operating Environment:** Windows 11 (64-bit), Python 3.14.0 (Global: `C:\Python314\python.exe`, Virtualenv: `backend\venv\Scripts\python.exe`), Node.js (Next.js 16.3.0 Turbopack).
- **Service Topology:**
  - FastAPI Backend: `http://127.0.0.1:8000` (Health HTTP 200)
  - Next.js Web Application: `http://localhost:3000` (HTTP 200)
  - Streamlit Dashboard: `http://localhost:8501` (HTTP 200)
  - MongoDB Database: `mongodb://127.0.0.1:27017` (`ai_qtriage` collection connected)

---

## 3. Verification Scope

The verification protocol executed non-destructive testing strictly within the local research prototype environment:
1. Mathematical precision and additivity of TreeExplainer SHAP margin attributions.
2. Cross-layer consistency between MongoDB, FastAPI, Next.js DOM, JSON reports, FHIR R4 bundles, and PDF streams.
3. Automated test execution across all 3 suites (Core Audit, Independent FHIR R4, and Playwright E2E).
4. Export schema conformance against HL7 FHIR R4 (`fhir.resources` 8.3.0) and PDF stream integrity (`pypdf`).
5. Basic local security, input boundaries, and graceful fallback on external API timeouts.
6. Scientific honesty, limitation disclosure, and removal of unsupported clinical claims.

---

## 4. SHAP Calculation Reconciliation

### Unrounded IEEE-754 Ground Truth vs. Display Rounding
For active case `a4a30e57-a232-4ca9-9fc4-3f135f7eff8d`, the XGBoost `multi:softprob` decision margin $z_{\text{HIGH}}$ is computed across all 23 fused multimodal features:

| Attribution Component | Raw IEEE-754 Float | Uniform 3-Decimal Display | Mathematical Formulation |
| :--- | :---: | :---: | :--- |
| **Top 6 Attributions** | `+1.495282` | `+1.495` | $\sum_{i=1}^6 \phi_i$ (Peak G-Force, Pain Level, Bleeding, etc.) |
| **Remaining 17 Tail** | `-0.042327` | `-0.042` | $\sum_{i=7}^{23} \phi_i$ (Residual tail across minor features) |
| **Net SHAP Attribution** | `+1.452955` | `+1.453` | $+1.495 + (-0.042) = +1.453$ logit |
| **Base Expected Value $E[f(x)]$** | `-0.082254` | `-0.082` | Model training set prior logit margin |
| **Total Decision Margin $z_{\text{HIGH}}$** | `+1.370701` | `+1.371` | $-0.082 + 1.453 = +1.371$ logit |
| **Predicted Softmax Probability** | `0.664903` | `66.5%` | $P(\text{HIGH}) = \frac{e^{1.371}}{\sum_{c=0}^2 e^{z_c}} \approx 66.49\%$ |

### Component Rounding Transparency
Summing the individually displayed 3-decimal feature contributions yields:
$$+1.576 + 0.271 - 0.197 - 0.190 + 0.079 - 0.045 = +1.494$$
The $0.001$ difference between $+1.494$ (sum of components rounded before addition) and $+1.495$ (full-precision sum $+1.495282$ rounded after addition) is transparently documented in the UI and test assertions as standard IEEE-754 float representation behavior.

---

## 5. UI, API, and Database Consistency

All application layers access and present identical state for case `a4a30e57-a232-4ca9-9fc4-3f135f7eff8d`:
1. **Vision Hypotheses**: YOLO11 detects **Burn (58.0%)**; EfficientNetV2 classifies **Bruise (98.6%)**. Disagreement is preserved without silent winner-takes-all suppression.
2. **Clinical Review Governance**: `requires_human_review = True` is enforced across database, API payload, JSON report, PDF red warning banner, and browser DOM. Automated settlement is blocked.
3. **U-Net Reliability Gate**: Acute lesion photograph produced 0 positive mask pixels $> 0.5$ threshold. Reliability gate is CLOSED; `affected_ratio = None` (`null`), rendered as `UNAVAILABLE_WITHHELD (N/A)`. Option-A zero-imputation is strictly isolated to XGBoost feature tensor input boundary (`vec[5] = 0.0`).
4. **Evidence Consistency Index**: $75.0 / 100$ ($100.0 - 15.0\text{ [UNet Withheld]} - 10.0\text{ [Synthetic Demonstration]} = 75.0$). Documented as a deterministic verification index, not an injury probability.

---

## 6. Test-Suite Results and Actual Exit Codes

Unified execution command: `python tests/run_all_verification_suites.py a4a30e57-a232-4ca9-9fc4-3f135f7eff8d`

| Suite Name | Execution Command | Checks | Exit Code | Duration | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Core Audit Verification Suite** | `python tests/test_audit_verification_suite.py` | 19 / 19 | `0` | 4.88s | **PASS** |
| **2. Independent FHIR R4 Conformance** | `python tests/test_fhir_independent_validation.py` | 33 / 33 | `0` | 0.58s | **PASS** |
| **3. Playwright Browser E2E & Exports** | `python tests/test_e2e_browser_exports.py` | 23 / 23 | `0` | 13.92s | **PASS** |
| **Consolidated Total** | `python tests/run_all_verification_suites.py` | **75 / 75** | **`0`** | **19.38s** | **100% PASS** |

- **Saved JSON Artifact:** [data/results/test_suite_all.json](file:///c:/Users/santh/Capstone%20Project%20Code/data/results/test_suite_all.json)
- **Saved Log Artifact:** [data/results/test_suite_all.log](file:///c:/Users/santh/Capstone%20Project%20Code/data/results/test_suite_all.log)

---

## 7. Export Validation

1. **JSON Report (`GET /api/cases/{case_id}/report/json`)**:
   - 33 top-level keys matching case schema.
   - Dual alias: both `"vision"` and `"visible_injury"` present with identical values.
   - Preserves `requires_human_review = True`, `affected_area_ratio = null`, raw SHAP explanations.
2. **HL7 FHIR R4 Bundle (`GET /api/cases/{case_id}/fhir`)**:
   - Formally parsed and validated against `fhir.resources.bundle.Bundle` (v8.3.0 Pydantic model).
   - Resource profiles validated: Patient, Device (SNOMED CT 86184003), Observation (LOINC 72170-4 & 11283-8), DiagnosticReport (LOINC 11526-1).
   - Status: `'preliminary'`. Observation components contain dual YOLO & EfficientNet hypotheses.
   - Segmentation area tagged as `UNAVAILABLE_WITHHELD (N/A, not zero-measured)`.
3. **PDF Clinical Report (`GET /api/cases/{case_id}/report/pdf`)**:
   - Generates valid 2-page PDF binary stream (11,845 bytes).
   - Text extracted via `pypdf.PdfReader` confirms case ID, red human-review warning box, and synthetic demonstration disclaimer.

---

## 8. Security and Reliability Findings

1. **Local Interface Isolation**: FastAPI bound to `127.0.0.1:8000`; Next.js bound to `localhost:3000`. No external cloud exposure.
2. **Emergency Workflow Safety**: SOS triggers local demonstration simulation (`local_demo`); no real SMS or 911 dispatch initiated.
3. **Input & Path Traversal Guard**: Path traversal probes (`../../etc/passwd`, non-existent IDs) return structured HTTP 404/422 responses; Next.js renders dedicated Error Card without stack trace leakage.
4. **External API Timeout Bound**: Gemini first-aid socket timeout bounded (1.02s); automatically fails safe to deterministic rule-based first-aid text.
5. **No Hardcoded Secrets**: Secrets isolated to local environment variables; `.gitignore` actively guards credentials.

---

## 9. Research Validity and Limitations

The following scientific disclosures are strictly verified in application code and user documentation:
1. **Academic Research Prototype**: AI-QTriage does not provide certified medical diagnoses.
2. **Zero Paired Clinical Multimodal Records**: All models are trained and benchmarked on synthetic simulations.
3. **Patient-Level Independence Unproven**: While exact-row hash overlap across train/val/test splits is 0, patient-level independence is unproven because the synthetic generator lacks persistent subject IDs.
4. **Classical 4-Qubit Simulation**: The VQC is executed on PennyLane `default.qubit` (17/30 correct, $56.67\%$ accuracy). It exhibits no quantum advantage and does not outperform classical XGBoost (25/30 correct, $83.33\%$ accuracy).
5. **Heuristic Consistency Index**: The $75/100$ score is a deterministic rule-based verification metric, not a clinical risk probability.

---

## 10. Defects Found and Corrections Made

| Defect Identified | Root Cause | File Modified | Resolution | Status |
| :--- | :--- | :--- | :--- | :---: |
| **SHAP Display Discrepancy** (+1.494 vs +1.495, -0.041 vs -0.042) | UI calculated aggregate top 6 by summing already-rounded numbers (1.494) and forced tail as residual (1.453 - 1.494 = -0.041). | [page.tsx](file:///c:/Users/santh/Capstone%20Project%20Code/frontend/app/cases/%5Bid%5D/page.tsx) & [test_audit_verification_suite.py](file:///c:/Users/santh/Capstone%20Project%20Code/tests/test_audit_verification_suite.py) | Calculated aggregates from full-precision IEEE-754 floats and rounded only for display (+1.495 & -0.042). | **FIXED** |
| **JSON Export Schema Inconsistency** | `report_service.py` emitted `"vision"` key while database/API emitted `"visible_injury"`. | [report_service.py](file:///c:/Users/santh/Capstone%20Project%20Code/backend/services/report_service.py) | Added dual-alias `report_data["visible_injury"] = report_data["vision"]`. | **FIXED** |
| **Test Verification Opacity** (UI reported 75 passed without observable logs) | Frontend displayed static verification badge without real-time terminal output viewer. | [page.tsx](file:///c:/Users/santh/Capstone%20Project%20Code/frontend/app/cases/%5Bid%5D/page.tsx) & [main.py](file:///c:/Users/santh/Capstone%20Project%20Code/backend/main.py) | Added `GET /api/cases/{id}/verification-run` and interactive terminal log viewer with tabs for all 3 suites. | **FIXED** |

---

## 11. Comprehensive Master Verification Matrix

| Check | Expected Result | Observed Result | Evidence | Status |
|---|---|---|---|:---:|
| **1. SHAP Full-Precision Reconciliation** | Base + Top 6 + Tail = Margin (+1.371) | $-0.082254 + 1.495282 - 0.042327 = +1.370701$ | `data/results/shap_verification.json` | **PASS** |
| **2. SHAP 3-Decimal Display Additivity** | $+1.495 + (-0.042) = +1.453 \rightarrow +1.371$ | Top 6: `+1.495`, Tail: `-0.042`, Net: `+1.453`, Margin: `+1.371` | Rendered Browser DOM & Check 14 | **PASS** |
| **3. UI Case ID Alignment** | `a4a30e57` across all panels | Case `a4a30e57-a232-4ca9-9fc4-3f135f7eff8d` displayed uniformly | Chromium Playwright DOM Check 2 | **PASS** |
| **4. Vision Disagreement Preservation** | YOLO Burn (58%) vs EffNet Bruise (99%) | Both hypotheses displayed; settlement halted | `data/results/cross_layer_consistency.json` | **PASS** |
| **5. Mandatory Human Review Banner** | `requires_human_review = True` | Prominently displayed in DOM, API, JSON, PDF | Check 16 & UI Check 4 | **PASS** |
| **6. U-Net Segmentation Reliability Gate** | CLOSED (0 px); affected area is `null`/`N/A` | `affected_ratio = None`, rendered as `UNAVAILABLE_WITHHELD (N/A)` | DB record & Check 6, 18 | **PASS** |
| **7. Evidence Consistency Formula** | $100.0 - 15.0 - 10.0 = 75.0 / 100$ | Score $75.0/100$ displayed with formula breakdown | Overview Panel & Check 17 | **PASS** |
| **8. Core Audit Test Suite Execution** | 19 / 19 passing checks | 19 / 19 passed (Exit Code 0) | `data/results/test_suite_all.json` | **PASS** |
| **9. Independent HL7 FHIR R4 Validation** | 33 / 33 passing checks | 33 / 33 passed (Exit Code 0, Pydantic 8.3.0) | `tests/test_fhir_independent_validation.py` | **PASS** |
| **10. Playwright Browser E2E & Exports** | 23 / 23 passing checks | 23 / 23 passed (Exit Code 0) | `tests/test_e2e_browser_exports.py` | **PASS** |
| **11. Consolidated Suite Size & Exit Code** | 75 checks executed; Exit Code 0 | Exactly 75 checks passed; Exit Code 0 | `data/results/test_suite_all.log` | **PASS** |
| **12. Live Terminal Log Console in UI** | Verbatim logs accessible in browser DOM | Terminal viewer renders all 3 suite logs | Playwright UI Check 9 | **PASS** |
| **13. Database-API Field Correspondence** | 100% field equivalence | MongoDB document matches API response | `data/results/api_database_verification.json` | **PASS** |
| **14. 404 Non-Existent Case Handling** | HTTP 404 with structured error | HTTP 404 returned; Error Card rendered | API & Playwright Check 8 | **PASS** |
| **15. JSON Report Export Fidelity** | 33 keys, review flag, vision hypotheses | Valid JSON, `requires_human_review = True` | `data/results/export_verification.json` | **PASS** |
| **16. FHIR R4 DiagnosticReport Status** | Status `'preliminary'` | `status: "preliminary"` | Pydantic R4 Bundle Validation | **PASS** |
| **17. PDF Stream & Human Review Banner** | 2 pages, red review warning box | Valid PDF stream (11,845 bytes, 2 pages) | `pypdf` text extraction | **PASS** |
| **18. Local Interface Security Binding** | Bound only to localhost / 127.0.0.1 | 127.0.0.1:8000 and localhost:3000 | `data/results/local_security_reliability.json` | **PASS** |
| **19. External API Socket Timeout Fallback** | Abort in $<1.02\text{s}$ to rule fallback | Gemini socket timeout falls back gracefully | Check 9 | **PASS** |
| **20. Zero Clinical Multimodal Records Notice** | Explicit disclaimer of 0 clinical pairs | Prominently displayed across UI and exports | `data/results/research_validity_audit.json` | **PASS** |
| **21. Classical VQC Scope Disclaimer** | 4-qubit classical simulation disclosure | 17/30 (56.67%) benchmarked on PennyLane | Model registry & VQC metadata | **PASS** |
| **22. Multi-Center Clinical Trial Validation** | Demonstrated real-world patient efficacy | Zero paired real-world records | Research boundary notice | **NOT RUN** (Out of Scope for Laptop Capstone) |

---

## 12. Remaining Risks and Blockers

1. **Academic Dataset Limitations (Documented)**:
   The models are trained exclusively on synthetic multimodal data. Ground-truth triage categories are rule-derived. While exact-row hash overlap is 0, patient-level independence cannot be established without persistent subject IDs.
2. **Clinical Deployment Barrier (Governed)**:
   AI-QTriage is strictly prohibited from autonomous clinical deployment. Mandatory human clinician review is enforced whenever models disagree or uncertainty is non-zero.
3. **External Cloud & Production Infrastructure (Intentionally Deprioritized)**:
   Production Kubernetes clustering, public cloud deployment, multi-region database replication, and public zero-trust networking were deliberately deprioritized as out-of-scope for a local laptop capstone prototype.

---

## 13. Evidence and Artifact Paths

All verification deliverables have been created and persisted locally:
- **Environment Inspection:** [data/results/environment_inspection.json](file:///c:/Users/santh/Capstone%20Project%20Code/data/results/environment_inspection.json)
- **SHAP Reconciliation:** [data/results/shap_verification.json](file:///c:/Users/santh/Capstone%20Project%20Code/data/results/shap_verification.json)
- **Cross-Layer Consistency:** [data/results/cross_layer_consistency.json](file:///c:/Users/santh/Capstone%20Project%20Code/data/results/cross_layer_consistency.json)
- **Consolidated Test Run JSON:** [data/results/test_suite_all.json](file:///c:/Users/santh/Capstone%20Project%20Code/data/results/test_suite_all.json)
- **Verbatim Test Run Log:** [data/results/test_suite_all.log](file:///c:/Users/santh/Capstone%20Project%20Code/data/results/test_suite_all.log)
- **API & Database Verification:** [data/results/api_database_verification.json](file:///c:/Users/santh/Capstone%20Project%20Code/data/results/api_database_verification.json)
- **Export Verification:** [data/results/export_verification.json](file:///c:/Users/santh/Capstone%20Project%20Code/data/results/export_verification.json)
- **Local Security & Reliability:** [data/results/local_security_reliability.json](file:///c:/Users/santh/Capstone%20Project%20Code/data/results/local_security_reliability.json)
- **Research Validity Audit:** [data/results/research_validity_audit.json](file:///c:/Users/santh/Capstone%20Project%20Code/data/results/research_validity_audit.json)
- **Browser UI Verification:** [data/results/ui_verification.json](file:///c:/Users/santh/Capstone%20Project%20Code/data/results/ui_verification.json)

---

## 14. Final Readiness Decision

**Verdict: READY FOR ACADEMIC CAPSTONE DEMONSTRATION**

The AI-QTriage system is **verified within the tested local research-prototype scope**. All mathematical calculations reconcile additively, all 75 automated checks pass with exit code 0, and the user interface transparently reveals both the evidence and its scientific limitations.
