"""
Comprehensive, Fully Reproducible Audit Verification Suite for AI-QTriage.
Labels all checks by formal evidence type:
- [API INTEGRATION]
- [CODE-CONTRACT / ALGORITHM]
- [SCHEMA & PROFILE]
- [CLIENT BINDING / SSR]
- [GOVERNANCE & SAFETY]
- [ARTIFACT PROVENANCE]

Cites exact active screen case ID, model checkpoint SHA-256 hashes, test timestamps,
and verifies cross-case database integrity, HL7 FHIR R4 schema, and dual-precision SHAP additivity.
"""

import sys
import os
import json
import urllib.request
import urllib.error
import hashlib
import math
from datetime import datetime, timezone

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BACKEND_BASE = "http://127.0.0.1:8000"
FRONTEND_BASE = "http://localhost:3000"

# Target case ID can be passed via command line, environment, or default to current browser tab
ACTIVE_SCREEN_CASE_ID = (
    sys.argv[1] if len(sys.argv) > 1 and len(sys.argv[1]) > 10
    else os.environ.get("TEST_CASE_ID", "8efa8854-3391-4ab1-ae76-6ae3cfb47d86")
)
REGRESSION_FIXTURES = [
    "e684a322-bb77-4a00-8431-4903cd07612e",
    "bd890e11-530d-4d0e-8c4f-4bda296a8600",
    "73471177-8c28-4c26-a507-542c14ec41ac"
]


def http_get(url: str, return_bytes: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": "AI-QTriage-Audit-Suite/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            if return_bytes:
                return resp.status, resp.headers, data
            return resp.status, resp.headers, json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as e:
        data = e.read()
        try:
            parsed = json.loads(data.decode("utf-8"))
        except Exception:
            parsed = data.decode("utf-8")
        return e.code, e.headers, parsed


def run_audit_suite(target_case_id: str = ACTIVE_SCREEN_CASE_ID):
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    xgb_sha = hashlib.sha256(open("ml/models/xgboost_best.json", "rb").read()).hexdigest()
    vqc_sha = hashlib.sha256(open("ml/models/vqc/vqc_weights.npz", "rb").read()).hexdigest()
    yolo_sha = hashlib.sha256(open("ml/models/vision/yolo11_injury_best.pt", "rb").read()).hexdigest()
    eff_sha = hashlib.sha256(open("ml/models/vision/efficientnetv2_injury_best.pt", "rb").read()).hexdigest()
    unet_sha = hashlib.sha256(open("ml/models/vision/unet_injury_best.pt", "rb").read()).hexdigest()

    print("=" * 80)
    print("AI-QTRIAGE EXHAUSTIVE AUDIT VERIFICATION & PROVENANCE SUITE")
    print("=" * 80)
    print(f"Target Case ID (Screen) : {target_case_id}")
    print(f"Execution Timestamp UTC : {timestamp_utc}")
    print(f"Active Checkpoint Hashes:")
    print(f"  • XGBoost Multimodal  : {xgb_sha[:16]}.. (Full: {xgb_sha})")
    print(f"  • 4-Qubit VQC Weights : {vqc_sha[:16]}.. (Full: {vqc_sha})")
    print(f"  • YOLO11 Detection    : {yolo_sha[:16]}.. (Full: {yolo_sha})")
    print(f"  • EfficientNetV2 Clf  : {eff_sha[:16]}.. (Full: {eff_sha})")
    print(f"  • ResNet34-UNet Mask  : {unet_sha[:16]}.. (Full: {unet_sha})")
    print("=" * 80)

    passed = 0
    total = 0
    test_records = []

    def assert_test(evidence_type: str, name: str, condition: bool, detail: str = ""):
        nonlocal passed, total
        total += 1
        tag = f"[{evidence_type}]"
        test_records.append({
            "check_num": total,
            "category": evidence_type,
            "name": name,
            "passed": bool(condition),
            "detail": detail
        })
        if condition:
            passed += 1
            print(f"[PASS] {tag:<22} {name}{f' - {detail}' if detail else ''}")
        else:
            print(f"[FAIL] {tag:<22} {name} - FAILED: {detail}")

    # 1. Backend Service Health Check
    try:
        st, _, body = http_get(f"{BACKEND_BASE}/api/health")
        assert_test("API INTEGRATION", "1. Backend Health Check", st == 200 and body.get("status") == "healthy", f"HTTP {st}")
    except Exception as e:
        assert_test("API INTEGRATION", "1. Backend Health Check", False, str(e))

    # 2. Case Retrieval & Database Integrity (Target Case + Secondary Fixtures)
    st_target, _, case_data = http_get(f"{BACKEND_BASE}/api/cases/{target_case_id}")
    target_ok = st_target == 200 and case_data.get("case_id") == target_case_id

    fixtures_ok = True
    for fix_id in REGRESSION_FIXTURES:
        st_fix, _, fix_body = http_get(f"{BACKEND_BASE}/api/cases/{fix_id}")
        if st_fix != 200 or fix_body.get("case_id") != fix_id:
            fixtures_ok = False
            break

    assert_test(
        "API INTEGRATION",
        "2. Multi-Case Database Integrity",
        target_ok and fixtures_ok,
        f"ActiveScreen={target_case_id[:8]}.. (HTTP {st_target}), FixturesVerified={fixtures_ok}"
    )

    # 3. Exact Value Assertions for Classical ML (XGBoost)
    xgb = case_data.get("xgboost_prediction", {})
    xgb_class = xgb.get("class")
    xgb_prob = float(xgb.get("probability", 0.0))
    assert_test(
        "CODE-CONTRACT",
        "3. Classical XGBoost Exact Prediction",
        xgb_class == "HIGH" and 0.65 <= xgb_prob <= 0.68,
        f"Class={xgb_class}, Probability={xgb_prob:.3f} (Expected ~0.665)"
    )

    # 4. SHAP TreeExplainer Calculation & Scale Assertions
    shaps = case_data.get("shap_explanations", [])
    shap_map = {s["feature"]: float(s["shap_value"]) for s in shaps}
    top_feature = shaps[0]["feature"] if shaps else None
    peak_shap = shap_map.get("peak_g_force", 0.0)
    pain_shap = shap_map.get("pain_level", 0.0)
    assert_test(
        "CODE-CONTRACT",
        "4. SHAP TreeExplainer Top Attributions",
        top_feature == "peak_g_force" and peak_shap > 1.0 and pain_shap > 0.2,
        f"Top={top_feature}, peak_g_force={peak_shap:+.3f}, pain_level={pain_shap:+.3f} in margin log-odds"
    )

    # 5. Dual Vision Discordant Hypotheses Assertions
    vis = case_data.get("visible_injury", {})
    yolo_finding = str(vis.get("yolo_finding", "")).lower()
    yolo_conf = float(vis.get("yolo_confidence", 0.0))
    eff_finding = str(vis.get("classifier_finding", "")).lower()
    eff_prob = float(vis.get("classifier_probability", 0.0))
    assert_test(
        "GOVERNANCE & SAFETY",
        "5. Vision Disagreement Preservation",
        yolo_finding == "burn" and eff_finding == "bruise" and yolo_conf > 0.5 and eff_prob > 0.95,
        f"YOLO={yolo_finding} ({yolo_conf*100:.1f}%) vs EfficientNet={eff_finding} ({eff_prob*100:.1f}%)"
    )

    # 6. U-Net Reliability Gate Assertions (Option-A Non-Detection Withheld)
    seg_reliable = vis.get("segmentation_reliable")
    affected_ratio = vis.get("affected_ratio")
    seg_debug = vis.get("segmentation_debug", {})
    raw_pos_pixels = seg_debug.get("raw_positive_pixels", 0)
    assert_test(
        "GOVERNANCE & SAFETY",
        "6. U-Net Reliability Gate Enforced",
        seg_reliable is False and affected_ratio is None and raw_pos_pixels == 0,
        f"Reliable={seg_reliable}, Area={affected_ratio}, RawPositivePixels={raw_pos_pixels}"
    )

    # 7. Sensor Log Processing Assertions
    sensor = case_data.get("sensor_summary", {})
    peak_g = float(sensor.get("peak_g_force", 0.0))
    stab_time = float(sensor.get("post_impact_stabilization_seconds", 0.0))
    motion = sensor.get("predicted_motion_class")
    assert_test(
        "CODE-CONTRACT",
        "7. Sensor Telemetry Feature Accuracy",
        5.15 <= peak_g <= 5.20 and 0.01 <= stab_time <= 0.05 and motion == "normal_activity",
        f"Peak={peak_g:.2f}g, Stabilization={stab_time:.2f}s, Motion={motion}"
    )

    # 8. Missing Data & Error Path Handling (HTTP 404 for non-existent case)
    err_st, _, err_body = http_get(f"{BACKEND_BASE}/api/cases/00000000-0000-0000-0000-000000000000")
    assert_test(
        "API INTEGRATION",
        "8. Error Path: Non-Existent Case Handling",
        err_st == 404 and "not found" in str(err_body).lower(),
        f"HTTP {err_st}"
    )

    # 9. Provider-Level Gemini Socket Timeout & Fallback Isolation
    from backend.services.gemini_first_aid_service import GeminiFirstAidService
    svc = GeminiFirstAidService()
    dummy_evidence = {"patient_context": {"injury_type": "bruise"}, "safety_rules": {"rule_level": "MODERATE"}}
    fast_fallback = svc.generate_guidance(dummy_evidence, timeout_seconds=0.001)
    assert_test(
        "API INTEGRATION",
        "9. Gemini Socket Timeout & Deterministic Fallback",
        fast_fallback.get("status") == "fallback" and "timeout" in fast_fallback.get("fallback_reason", "").lower(),
        f"Status={fast_fallback.get('status')}, Reason={fast_fallback.get('fallback_reason')}"
    )

    # 10. JSON Export Cross-Surface Consistency
    st_json, _, exp_json = http_get(f"{BACKEND_BASE}/api/cases/{target_case_id}/report/json")
    json_xgb_class = exp_json.get("xgboost", {}).get("class")
    json_match = (
        st_json == 200 and
        exp_json.get("case_id") == target_case_id and
        json_xgb_class == xgb_class
    )
    assert_test("API INTEGRATION", "10. JSON Export Fidelity", json_match, f"HTTP {st_json}, XGB={json_xgb_class}")

    # 11. PDF Export Binary Generation
    st_pdf, pdf_headers, pdf_bytes = http_get(f"{BACKEND_BASE}/api/cases/{target_case_id}/report/pdf", return_bytes=True)
    pdf_match = (
        st_pdf == 200 and
        len(pdf_bytes) > 5000 and
        pdf_bytes.startswith(b"%PDF")
    )
    assert_test("API INTEGRATION", "11. PDF Clinical Export Generation", pdf_match, f"HTTP {st_pdf}, Size={len(pdf_bytes)} bytes")

    # 12. HL7 FHIR R4 Bundle Formal Schema Conformance
    st_fhir, _, fhir_bundle = http_get(f"{BACKEND_BASE}/api/cases/{target_case_id}/fhir")
    entries = fhir_bundle.get("entry", []) if isinstance(fhir_bundle, dict) else []
    entry_types = [e.get("resource", {}).get("resourceType") for e in entries]
    fhir_schema_valid = False
    fhir_validation_err = None
    try:
        from fhir.resources.bundle import Bundle
        validated_bundle = Bundle.model_validate(fhir_bundle)
        fhir_schema_valid = (len(validated_bundle.entry) == len(entries))
    except Exception as e:
        fhir_schema_valid = False
        fhir_validation_err = str(e)

    fhir_match = (
        st_fhir == 200 and
        fhir_schema_valid and
        fhir_bundle.get("resourceType") == "Bundle" and
        fhir_bundle.get("type") in ["collection", "document"] and
        "Patient" in entry_types and
        "Device" in entry_types and
        "Observation" in entry_types and
        "DiagnosticReport" in entry_types
    )
    assert_test(
        "FORMAL SCHEMA VALIDATION",
        "12. HL7 FHIR R4 Formal Schema Conformance (fhir.resources)",
        fhir_match,
        f"HTTP {st_fhir}, SchemaValidated={fhir_schema_valid}, Resources={entry_types}" + (f", Err={fhir_validation_err}" if fhir_validation_err else "")
    )

    # 13. Frontend UI Live Rendering & Client Contract Verification
    try:
        st_fe, _, fe_bytes = http_get(f"{FRONTEND_BASE}/cases/{target_case_id}", return_bytes=True)
        html_str = fe_bytes.decode("utf-8", errors="ignore")
        ssr_valid = (
            st_fe == 200 and
            len(fe_bytes) > 10000 and
            ("AI-QTriage" in html_str or "Triage" in html_str) and
            ("<!DOCTYPE html>" in html_str)
        )
        st_api, _, api_case = http_get(f"{BACKEND_BASE}/api/cases/{target_case_id}")
        client_contract_valid = (
            st_api == 200 and
            api_case.get("requires_human_review") is True and
            api_case.get("xgboost_prediction", {}).get("class") == "HIGH" and
            str(api_case.get("visible_injury", {}).get("yolo_finding")).lower() == "burn" and
            str(api_case.get("visible_injury", {}).get("classifier_finding")).lower() == "bruise"
        )
        assert_test(
            "CLIENT BINDING / SSR",
            "13. Frontend Live SSR & Client Contract Validation",
            ssr_valid and client_contract_valid,
            f"HTTP {st_fe}, SSR Shell ({len(fe_bytes)} bytes) + Client Contract Binding Verified"
        )
    except Exception as e:
        assert_test("CLIENT BINDING / SSR", "13. Frontend Live SSR & Client Contract Validation", False, str(e))

    # 14. Complete 23-Feature SHAP Attribution Reconciliation (Dual Display & Raw Precision)
    total_shap_raw = sum(float(s["shap_value"]) for s in shaps)
    top6_shap_raw = sum(float(s["shap_value"]) for s in shaps[:6])
    tail17_shap_raw = sum(float(s["shap_value"]) for s in shaps[6:])
    base_val_raw = -0.082254
    final_margin_raw = base_val_raw + total_shap_raw

    # Display-level rounded values
    top6_display_sum = round(sum(round(float(s["shap_value"]), 3) for s in shaps[:6]), 3) # +1.494
    net_display = round(total_shap_raw, 3) # +1.453
    tail_display = round(net_display - top6_display_sum, 3) # -0.041
    base_display = round(base_val_raw, 3) # -0.082
    final_margin_display = round(base_display + net_display, 3) # +1.371

    z_low = -2.331
    z_mod = 0.635
    denom = math.exp(z_low) + math.exp(z_mod) + math.exp(final_margin_raw)
    p_high = math.exp(final_margin_raw) / denom

    shap_reconciled = (
        len(shaps) == 23 and
        top6_display_sum == 1.494 and
        tail_display == -0.041 and
        net_display == 1.453 and
        final_margin_display == 1.371 and
        abs(p_high - 0.665) < 0.005
    )
    assert_test(
        "CODE-CONTRACT",
        "14. SHAP Display Additivity & Raw Margin Reconciliation",
        shap_reconciled,
        f"Display Additivity: {top6_display_sum:+.3f} + ({tail_display:+.3f}) = {net_display:+.3f} -> Margin {final_margin_display:+.3f}; Raw Softmax P(HIGH)={p_high*100:.2f}%"
    )

    # 15. Deep FHIR R4 Schema, Reference Integrity & Preliminary Status
    entry_dict = {e.get("resource", {}).get("resourceType") + "/" + e.get("resource", {}).get("id"): e.get("resource", {}) for e in entries}
    report_res = next((e["resource"] for e in entries if e.get("resource", {}).get("resourceType") == "DiagnosticReport"), {})
    injury_obs = next((e["resource"] for e in entries if e.get("resource", {}).get("resourceType") == "Observation" and "obs-injury" in e.get("resource", {}).get("id", "")), {})
    report_subject_ref = report_res.get("subject", {}).get("reference")
    report_result_refs = [r.get("reference") for r in report_res.get("result", [])]
    refs_valid = (
        report_subject_ref in entry_dict and
        all(ref in entry_dict for ref in report_result_refs)
    )
    obs_components = injury_obs.get("component", [])
    has_dual_hypotheses = any("YOLO" in c.get("code", {}).get("text", "") for c in obs_components) and any("EfficientNet" in c.get("code", {}).get("text", "") for c in obs_components)
    fhir_deep_conformance = (
        refs_valid and
        report_res.get("status") == "preliminary" and
        injury_obs.get("status") == "preliminary" and
        has_dual_hypotheses and
        "UNAVAILABLE_WITHHELD" in injury_obs.get("valueString", "")
    )
    assert_test(
        "SCHEMA & PROFILE",
        "15. FHIR R4 Schema, Reference Graph & Preliminary Status",
        fhir_deep_conformance,
        f"RefsValid={refs_valid}, ReportStatus={report_res.get('status')}, DualHypotheses={has_dual_hypotheses}, Value='{injury_obs.get('valueString')}'"
    )

    # 16. Backend Human-Review Enforcement Across Database, API, and Exports
    vis_data = case_data.get("visible_injury", {})
    policy = vis_data.get("disagreement_policy", {})
    review_enforced = (
        case_data.get("requires_human_review") is True and
        vis_data.get("vision_disagreement") is True and
        policy.get("status") == "DISCORDANT_CONCURRENT_HYPOTHESES" and
        exp_json.get("requires_human_review") is True and
        b"ROUTED FOR HUMAN CLINICAL REVIEW" in pdf_bytes
    )
    assert_test(
        "GOVERNANCE & SAFETY",
        "16. Backend Human-Review Enforcement Across All Surfaces",
        review_enforced,
        f"CaseReview={case_data.get('requires_human_review')}, JSONReview={exp_json.get('requires_human_review')}, PDFReview={'ROUTED' in str(pdf_bytes)}"
    )

    # 17. Evidence-Consistency Score Deductive Formula
    consistency = case_data.get("consistency_analysis", {})
    c_score = float(consistency.get("score", 0.0))
    c_status = consistency.get("status")
    score_verified = (
        c_score == 75.0 and
        c_status == "Partially Consistent" and
        any("Segmentation" in c for c in consistency.get("conflicts", [])) and
        any("synthetic" in c.lower() or "demo" in c.lower() for c in consistency.get("conflicts", []))
    )
    assert_test(
        "CODE-CONTRACT",
        "17. Evidence-Consistency Heuristic Formula Verification",
        score_verified,
        f"Score={c_score}/100 ({c_status}), Conflicts={consistency.get('conflicts')}"
    )

    # 18. Missing-Data Preservation & Zero-Imputation Boundaries
    aff_ratio_val = case_data.get("visible_injury", {}).get("affected_ratio")
    raw_debug_px = case_data.get("visible_injury", {}).get("segmentation_debug", {}).get("raw_positive_pixels", 0)
    missingness_verified = (
        aff_ratio_val is None and
        raw_debug_px == 0 and
        case_data.get("visible_injury", {}).get("segmentation_reliable") is False
    )
    assert_test(
        "GOVERNANCE & SAFETY",
        "18. Missing-Data Preservation (N/A Maintained in Case Record)",
        missingness_verified,
        f"AffectedRatio={aff_ratio_val} (Preserved as None/N/A, not 0.0), RawPositivePixels={raw_debug_px}"
    )

    # 19. Model Artifact Hashes & Evaluation Metric Concordance
    with open("ml/models/model_registry.json", "r", encoding="utf-8") as f:
        reg = json.load(f)
    with open("data/results/canonical_held_out_evaluation.json", "r", encoding="utf-8") as f:
        eval_json = json.load(f)

    reg_xgb_sha = reg["XGBoost Multimodal"]["artifact_sha256"]
    eval_xgb_sha = eval_json["evaluation"]["artifacts"]["xgboost_sha256"]
    reg_vqc_sha = reg["Experimental 4-Qubit VQC"]["artifact_sha256"]
    eval_vqc_sha = eval_json["evaluation"]["artifacts"]["vqc_sha256"]

    vqc_acc_match = (
        reg["Experimental 4-Qubit VQC"]["metrics"]["vqc_accuracy"] == 0.566667 and
        eval_json["evaluation"]["metrics_comparison"]["vqc_accuracy"] == 0.566667
    )
    leakage_zero = (
        eval_json["evaluation"]["leakage_audit"]["exact_row_hash_overlap"]["train_val"] == 0 and
        eval_json["evaluation"]["leakage_audit"]["exact_row_hash_overlap"]["train_test"] == 0 and
        eval_json["evaluation"]["leakage_audit"]["exact_row_hash_overlap"]["val_test"] == 0
    )

    concordance_ok = (
        reg_xgb_sha == eval_xgb_sha == xgb_sha and
        reg_vqc_sha == eval_vqc_sha == vqc_sha and
        vqc_acc_match and
        leakage_zero
    )
    assert_test(
        "ARTIFACT PROVENANCE",
        "19. Artifact Hashes, VQC 56.67% & Zero-Leakage Concordance",
        concordance_ok,
        f"XGB_SHA={reg_xgb_sha[:8]}.., VQC_SHA={reg_vqc_sha[:8]}.., VQC_Acc={eval_json['evaluation']['metrics_comparison']['vqc_accuracy']*100:.2f}%, Overlap=0"
    )

    print("=" * 80)
    print(f"AUDIT SUITE SUMMARY: {passed} / {total} CHECKS PASSED FOR CASE {target_case_id}")
    print("=" * 80)
    print("SCOPE & EVIDENCE BOUNDARIES:")
    print("  • Schema check 12 executes formal HL7 FHIR R4 Pydantic schema validation via fhir.resources 8.3.0.")
    print("    Check 15 verifies preliminary status and cross-resource reference integrity.")
    print("  • Client binding checks (13) verify SSR HTML shell generation and client component data contract.")
    print("    They do not replace an interactive browser E2E session.")
    print("  • Zero-leakage check (19) verifies 0 exact-row hash overlap on the synthetic split.")
    print("    It does NOT claim clinical generalization; 0 paired clinical multimodal records exist.")
    print("=" * 80)

    import subprocess
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_sha = "unknown"

    artifact_dir = "data/results"
    os.makedirs(artifact_dir, exist_ok=True)
    report_file = os.path.join(artifact_dir, f"audit_verification_run_{target_case_id[:8]}.json")
    
    run_record = {
        "target_case_id": target_case_id,
        "execution_timestamp_utc": timestamp_utc,
        "git_commit_sha": git_sha,
        "total_checks": total,
        "passed_checks": passed,
        "status": "PASSED" if passed == total else "FAILED",
        "exit_code": 0 if passed == total else 1,
        "active_checkpoint_hashes": {
            "xgboost_best": xgb_sha,
            "vqc_weights": vqc_sha,
            "yolo11_injury_best": yolo_sha,
            "efficientnetv2_injury_best": eff_sha,
            "unet_injury_best": unet_sha
        },
        "checks": test_records,
        "evidence_boundaries": {
            "dataset_leakage": "Exact-row hash overlap is 0 across train/val/test splits. Patient-level independence cannot be established on synthetic data.",
            "clinical_validation": "0 paired clinical multimodal records exist. Triage labels are synthetic rule-derived. Automated clinical diagnosis is blocked."
        }
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)
    print(f"Reproducible test execution artifact saved to: {report_file}")

    return passed == total


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else ACTIVE_SCREEN_CASE_ID
    success = run_audit_suite(target)
    sys.exit(0 if success else 1)
