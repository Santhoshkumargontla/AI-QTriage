"""
End-to-End Browser Automation & Multimodal Export Inspection Suite
==================================================================
Uses Playwright (Chromium Headless) to verify:
  1. Live Next.js Browser UI:
     - Case loading and metadata binding
     - Critical human review warning banner
     - Conflicting vision predictions presentation (Burn 58.0% vs Bruise 98.6%)
     - Segmentation withheld notice (Option-A zero-imputation safety)
     - Interactive tab switching (Overview, SHAP, Audit, Counterfactuals)
     - Error state handling (non-existent case page)
  2. Multimodal Export Inspection:
     - JSON Export: parsed and verified for review flag, predictions, and SHAP
     - PDF Export: binary download, pypdf text extraction, warning and case ID assertion
     - FHIR R4 Bundle: full Pydantic schema validation and preliminary status check
"""

import sys
import io
import json
import urllib.request
from typing import Dict, Any

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("[ERROR] Playwright is not installed. Install with 'pip install playwright' and 'python -m playwright install chromium'.")
    sys.exit(1)

try:
    from pypdf import PdfReader
except ImportError:
    print("[ERROR] pypdf is not installed. Install with 'pip install pypdf'.")
    sys.exit(1)

try:
    from fhir.resources.bundle import Bundle
except ImportError:
    print("[ERROR] fhir.resources is not installed. Install with 'pip install fhir.resources'.")
    sys.exit(1)

FRONTEND_BASE = "http://localhost:3000"
BACKEND_BASE = "http://127.0.0.1:8000"
TARGET_CASE_ID = sys.argv[1] if len(sys.argv) > 1 else "8efa8854-3391-4ab1-ae76-6ae3cfb47d86"

passed = 0
total = 0

def check(category: str, title: str, condition: bool, detail: str = ""):
    global passed, total
    total += 1
    tag = "PASS" if condition else "FAIL"
    print(f"[{tag}] [{category:<25}] {title} - {detail}")
    if not condition:
        print(f"       --> FAILED ASSERTION!")
    else:
        passed += 1


def run_browser_e2e_tests():
    print("\n" + "=" * 80)
    print("PHASE 1: BROWSER AUTOMATION E2E TESTING (Playwright Chromium Headless)")
    print("=" * 80)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()

        # 1. Load active case page
        case_url = f"{FRONTEND_BASE}/cases/{TARGET_CASE_ID}"
        response = page.goto(case_url, wait_until="networkidle", timeout=30000)
        check("Browser E2E", "1. Active Case Page Load", response.status == 200, f"HTTP {response.status} at {case_url}")

        # 2. Page Title & Case Header
        content = page.content()
        check("Browser E2E", "2. UI Header & Case ID", TARGET_CASE_ID in content or TARGET_CASE_ID[:8] in content, f"Case {TARGET_CASE_ID[:8]} found in DOM")

        # 3. Critical Human Review Warning Banner
        has_review_banner = (
            "HUMAN REVIEW" in content.upper() or
            "ROUTED FOR HUMAN CLINICAL REVIEW" in content.upper() or
            "requires_human_review = true" in content
        )
        check("Browser E2E", "3. Human Review Safety Banner", has_review_banner, "Human review enforcement visible in UI")

        # 4. Conflicting Vision Predictions
        has_burn = "burn" in content.lower()
        has_bruise = "bruise" in content.lower()
        check("Browser E2E", "4. Concurrent Vision Hypotheses", has_burn and has_bruise, "Both Burn and Bruise visible concurrently")

        # 5. Strict Segmentation Mask Withheld Verification
        # Must verify EXACT badge and diagnostic strings, not permissive presence/absence
        has_gate_closed = "Reliability Gate: CLOSED (Output Withheld)" in content
        has_zero_pixels = "Segmentation non-detection (0 positive pixels)" in content
        has_na_area = "N/A" in content
        exact_seg_withheld = (has_gate_closed and has_zero_pixels and has_na_area)
        check(
            "Browser E2E",
            "5. Strict Segmentation Gate Safety",
            exact_seg_withheld,
            f"GateClosed={has_gate_closed}, ZeroPixelsTag={has_zero_pixels}, AreaNA={has_na_area}"
        )

        # 6. Tab Navigation: Switch to Audit Tab
        audit_tab = page.locator("#case-tab-audit")
        if audit_tab.count() > 0:
            audit_tab.click()
            page.wait_for_timeout(500)
            audit_content = page.content()
            has_audit_table = ("11-Issue" in audit_content or "Audit Matrix" in audit_content or "Verification" in audit_content)
            check("Browser E2E", "6. Tab Navigation: Audit Matrix", has_audit_table, "Audit & Governance matrix displayed upon click")
        else:
            check("Browser E2E", "6. Tab Navigation: Audit Matrix", False, "Audit tab button #case-tab-audit not located")

        # 7. Tab Navigation: Switch to Explainability (SHAP) Tab
        expl_tab = page.locator("#case-tab-explainability")
        if expl_tab.count() > 0:
            expl_tab.click()
            page.wait_for_timeout(500)
            expl_content = page.content()
            has_shap = ("SHAP" in expl_content or "peak_g_force" in expl_content or "attribution" in expl_content.lower())
            check("Browser E2E", "7. Tab Navigation: SHAP Breakdown", has_shap, "SHAP attribution visualization visible")
        else:
            check("Browser E2E", "7. Tab Navigation: SHAP Breakdown", False, "Explainability tab #case-tab-explainability not located")

        # 8. Strict Error State Handling: 404 Non-Existent Case Page
        # Must verify exact Error Card rendering and absence of normal case content
        bad_url = f"{FRONTEND_BASE}/cases/non-existent-case-id-12345"
        bad_resp = page.goto(bad_url, wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(1000)
        bad_body_text = page.locator("body").inner_text()
        has_error_title = "Error Retrieving Case" in bad_body_text
        has_return_btn = "Return to Dashboard" in bad_body_text
        normal_ui_blocked = ("Case sections" not in bad_body_text and "Full Case Overview" not in bad_body_text)
        strict_error_handled = (has_error_title and has_return_btn and normal_ui_blocked)
        check(
            "Browser E2E",
            "8. Strict Error State Rendering (404 Case)",
            strict_error_handled,
            f"ErrorTitle={has_error_title}, ReturnBtn={has_return_btn}, NormalUIBlocked={normal_ui_blocked}"
        )

        browser.close()


def run_export_inspection_tests():
    print("\n" + "=" * 80)
    print("PHASE 2: DEEP MULTIMODAL EXPORT INSPECTION (JSON, PDF, FHIR)")
    print("=" * 80)

    # 1. JSON Export Inspection
    json_url = f"{BACKEND_BASE}/api/cases/{TARGET_CASE_ID}/report/json"
    with urllib.request.urlopen(json_url, timeout=10) as resp:
        json_data = json.loads(resp.read().decode("utf-8"))

    check("JSON Export", "1. HTTP 200 & Valid Structure", isinstance(json_data, dict), f"Keys: {len(json_data.keys())}")
    check("JSON Export", "2. Human Review Flag Enforced", json_data.get("requires_human_review") is True, "requires_human_review=True")
    
    vis = json_data.get("vision", {})
    check("JSON Export", "3. Conflicting Vision Hypotheses", vis.get("yolo_finding") == "Burn" and vis.get("classifier_finding") == "Bruise", f"YOLO={vis.get('yolo_finding')}, Classifier={vis.get('classifier_finding')}")
    check("JSON Export", "4. Vision Disagreement Flag", vis.get("vision_disagreement") is True, "vision_disagreement=True")
    
    area_withheld = (vis.get("segmentation_reliable") is False and (vis.get("affected_ratio") is None or vis.get("affected_ratio") == "N/A"))
    check("JSON Export", "5. Segmentation Area Withheld", area_withheld, f"reliable={vis.get('segmentation_reliable')}, area={vis.get('affected_ratio')}")

    # 2. PDF Clinical Export Inspection
    pdf_url = f"{BACKEND_BASE}/api/cases/{TARGET_CASE_ID}/report/pdf"
    with urllib.request.urlopen(pdf_url, timeout=10) as resp:
        pdf_bytes = resp.read()

    check("PDF Export", "6. Binary PDF Stream", pdf_bytes.startswith(b"%PDF") and len(pdf_bytes) > 5000, f"Size: {len(pdf_bytes)} bytes")
    
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_pdf_text = "\n".join([page.extract_text() for page in reader.pages])

    check("PDF Export", "7. PDF Page Count", len(reader.pages) >= 2, f"Total pages: {len(reader.pages)}")
    check("PDF Export", "8. Case ID Preserved in PDF", TARGET_CASE_ID in full_pdf_text, f"Target case ID found in PDF text")
    check("PDF Export", "9. Human Review Warning in PDF", "ROUTED FOR HUMAN CLINICAL REVIEW" in full_pdf_text, "Mandatory human review warning present in PDF")
    check("PDF Export", "10. Vision Disagreement in PDF", "Discordant Vision Hypotheses Detected" in full_pdf_text, "Discordant vision findings detailed in PDF")
    check("PDF Export", "11. Data Provenance in PDF", "SYNTHETIC" in full_pdf_text, "Synthetic demonstration provenance clearly labeled")

    # 3. HL7 FHIR R4 Export Inspection
    fhir_url = f"{BACKEND_BASE}/api/cases/{TARGET_CASE_ID}/fhir"
    with urllib.request.urlopen(fhir_url, timeout=10) as resp:
        fhir_bundle = json.loads(resp.read().decode("utf-8"))

    try:
        validated_bundle = Bundle.model_validate(fhir_bundle)
        check("FHIR Export", "12. Pydantic R4 Schema Conformance", len(validated_bundle.entry) == 5, f"{len(validated_bundle.entry)} entries formally validated")
    except Exception as e:
        check("FHIR Export", "12. Pydantic R4 Schema Conformance", False, f"Validation error: {e}")

    entries = {e.get("resource", {}).get("resourceType"): e.get("resource") for e in fhir_bundle.get("entry", [])}
    diag_rep = entries.get("DiagnosticReport", {})
    
    # Extract obs-injury specifically (not overwritten by obs-triage)
    obs_inj = next((e.get("resource", {}) for e in fhir_bundle.get("entry", []) if e.get("resource", {}).get("id", "").startswith("obs-injury")), {})
    
    check("FHIR Export", "13. DiagnosticReport Status Preliminary", diag_rep.get("status") == "preliminary", f"status={diag_rep.get('status')}")
    check("FHIR Export", "14. Human Review Conclusion in FHIR", "ROUTED FOR HUMAN CLINICAL REVIEW" in diag_rep.get("conclusion", ""), "Conclusion includes human review instruction")
    check("FHIR Export", "15. Dual Hypotheses in FHIR Components", len(obs_inj.get("component", [])) >= 2, f"Components count: {len(obs_inj.get('component', []))}")


def main():
    print("=" * 80)
    print("AI-QTRIAGE END-TO-END BROWSER & MULTIMODAL EXPORT VERIFICATION")
    print(f"Target Active Screen Case ID: {TARGET_CASE_ID}")
    print("=" * 80)

    run_browser_e2e_tests()
    run_export_inspection_tests()

    print("\n" + "=" * 80)
    print(f"E2E BROWSER & EXPORT SUMMARY: {passed} / {total} CHECKS PASSED")
    print("=" * 80)

    if passed != total:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
