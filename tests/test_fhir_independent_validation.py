"""
Independent HL7 FHIR R4 Conformance & Interoperability Validator
================================================================
Validates exported FHIR bundles against the official HL7 FHIR R4 standard
using the authoritative `fhir.resources` (v8.3.0) Pydantic schema model.

Evaluates:
  1. Full Pydantic HL7 FHIR R4 model validation (fhir.resources.bundle.Bundle)
  2. Resource profiles: Bundle, Patient, Device, Observation, DiagnosticReport
  3. Standard Terminology: LOINC (72170-4, 11283-8, 11526-1), SNOMED (86184003), HL7 Categories
  4. Temporal conformance: ISO-8601 with timezone (Z)
  5. Cross-resource reference graph integrity
  6. Human-review governance state: 'preliminary' vs 'final' status transitions
  7. Dual concurrent vision hypothesis preservation
  8. Missing-data preservation: unavailable segmentation is withheld, not zero-measured
"""

import sys
import json
import urllib.request
from typing import Dict, Any, List

try:
    from fhir.resources.bundle import Bundle
    from fhir.resources.patient import Patient
    from fhir.resources.device import Device
    from fhir.resources.observation import Observation
    from fhir.resources.diagnosticreport import DiagnosticReport
except ImportError:
    print("[ERROR] fhir.resources is not installed. Install with 'pip install fhir.resources'.")
    sys.exit(1)

BACKEND_BASE = "http://127.0.0.1:8000"
ACTIVE_SCREEN_CASE_ID = (
    sys.argv[1] if len(sys.argv) > 1 and len(sys.argv[1]) > 10
    else "8efa8854-3391-4ab1-ae76-6ae3cfb47d86"
)
FIXTURE_CASE_ID = "concordant-test-fixture-001"

passed_checks = 0
total_checks = 0

def check(category: str, title: str, condition: bool, detail: str = ""):
    global passed_checks, total_checks
    total_checks += 1
    tag = "PASS" if condition else "FAIL"
    print(f"[{tag}] [{category:<25}] {title} - {detail}")
    if not condition:
        print(f"       --> FAILED: Condition not met!")
    else:
        passed_checks += 1


def fetch_bundle(case_id: str) -> Dict[str, Any]:
    url = f"{BACKEND_BASE}/api/cases/{case_id}/fhir"
    req = urllib.request.Request(url, headers={"User-Agent": "FHIR-Independent-Validator/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} fetching FHIR bundle from {url}")
        return json.loads(resp.read().decode("utf-8"))


def validate_case_bundle(case_id: str, expects_human_review: bool):
    print("\n" + "=" * 80)
    print(f"INDEPENDENT FHIR R4 VALIDATION FOR CASE: {case_id}")
    print(f"Expects Human Review / Preliminary Status: {expects_human_review}")
    print("=" * 80)

    raw_bundle = fetch_bundle(case_id)

    # 1. Official Pydantic FHIR R4 Model Validation
    try:
        validated = Bundle.model_validate(raw_bundle)
        check("Pydantic R4 Schema", f"Bundle.model_validate({case_id[:8]})", True, f"Parsed {len(validated.entry)} entries without schema error")
    except Exception as e:
        check("Pydantic R4 Schema", f"Bundle.model_validate({case_id[:8]})", False, f"Validation error: {e}")
        return

    # 2. Bundle Profile & Meta
    bundle_type = raw_bundle.get("type")
    check("Bundle Profile", "Bundle resourceType & type", raw_bundle.get("resourceType") == "Bundle" and bundle_type in ["collection", "document"], f"type={bundle_type}")
    
    profiles = raw_bundle.get("meta", {}).get("profile", [])
    has_bundle_profile = "http://hl7.org/fhir/StructureDefinition/Bundle" in profiles
    check("Profile Conformance", "meta.profile HL7 Bundle StructureDefinition", has_bundle_profile, f"profiles={profiles}")

    timestamp = raw_bundle.get("timestamp", "")
    has_tz = timestamp.endswith("Z") or ("+" in timestamp[10:] or "-" in timestamp[10:])
    check("Temporal Spec", "Bundle timestamp ISO timezone", bool(timestamp and has_tz), f"timestamp={timestamp}")

    # Extract resource entries by resourceType
    resources: Dict[str, List[Dict[str, Any]]] = {}
    all_codings = []
    for entry in raw_bundle.get("entry", []):
        res = entry.get("resource", {})
        rtype = res.get("resourceType")
        resources.setdefault(rtype, []).append(res)
        # Collect codings
        for k in ["code", "category", "type"]:
            val = res.get(k)
            if isinstance(val, dict) and "coding" in val:
                all_codings.extend(val.get("coding", []))
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "coding" in item:
                        all_codings.extend(item.get("coding", []))

    # Comprehensive Coding Integrity Check (system, code, display)
    valid_codings = all(
        c.get("system", "").startswith("http") and bool(c.get("code")) and bool(c.get("display"))
        for c in all_codings
    )
    check("Terminology Spec", "All codings contain valid system URI, code & display", valid_codings and len(all_codings) >= 5, f"Checked {len(all_codings)} codings across bundle entries")

    # 3. Patient Resource
    pts = resources.get("Patient", [])
    check("Patient Resource", "Patient exists & valid identifier", len(pts) == 1 and pts[0].get("identifier", [{}])[0].get("system") == "http://ai-qtriage.org/patients", f"id={pts[0].get('id') if pts else 'None'}")
    patient_id = pts[0].get("id") if pts else None

    # 4. Device Resource
    devs = resources.get("Device", [])
    dev_type_ok = False
    if devs:
        dev = devs[0]
        dev_types = dev.get("type", [])
        if isinstance(dev_types, list) and len(dev_types) > 0:
            coding = dev_types[0].get("coding", [{}])[0]
            dev_type_ok = (coding.get("system") == "http://snomed.info/sct" and coding.get("code") == "86184003")
    check("Device Resource", "Device SNOMED CT 86184003", dev_type_ok, f"dev_type_ok={dev_type_ok}")

    # 5. Observation Resources (Injury & Acuity)
    obs_list = resources.get("Observation", [])
    check("Observation Resources", "Two Observation resources present", len(obs_list) == 2, f"count={len(obs_list)}")

    obs_injury = next((o for o in obs_list if o.get("id", "").startswith("obs-injury")), None)
    obs_triage = next((o for o in obs_list if o.get("id", "").startswith("obs-triage")), None)

    if obs_injury:
        coding = obs_injury.get("code", {}).get("coding", [{}])[0]
        check("LOINC Terminology", "Injury LOINC 72170-4", coding.get("code") == "72170-4" and coding.get("system") == "http://loinc.org", f"code={coding.get('code')}")
        
        status_expected = "preliminary" if expects_human_review else "final"
        check("Governance Status", f"Injury status matches review status ({status_expected})", obs_injury.get("status") == status_expected, f"status={obs_injury.get('status')}")

        if expects_human_review:
            # Check dual concurrent hypotheses components
            components = obs_injury.get("component", [])
            comp_texts = [c.get("code", {}).get("text", "") for c in components]
            has_yolo = any("YOLO" in t for t in comp_texts)
            has_eff = any("EfficientNet" in t for t in comp_texts)
            has_policy = any("Disagreement" in t for t in comp_texts)
            check("Dual Hypotheses", "Both YOLO & EfficientNet preserved in components", has_yolo and has_eff and has_policy, f"components={comp_texts}")

            # Check withheld segmentation preservation
            val_str = obs_injury.get("valueString", "")
            check("Missing Data", "Segmentation area withheld (UNAVAILABLE_WITHHELD preserved)", "UNAVAILABLE_WITHHELD" in val_str, f"val={val_str[:60]}...")

    if obs_triage:
        coding = obs_triage.get("code", {}).get("coding", [{}])[0]
        check("LOINC Terminology", "Triage LOINC 11283-8", coding.get("code") == "11283-8" and coding.get("system") == "http://loinc.org", f"code={coding.get('code')}")
        check("Acuity Value", "Rule-Derived triage category valid", obs_triage.get("valueString") in ["LOW", "MODERATE", "HIGH"], f"category={obs_triage.get('valueString')}")

    # 6. DiagnosticReport Resource
    reps = resources.get("DiagnosticReport", [])
    if reps:
        rep = reps[0]
        coding = rep.get("code", {}).get("coding", [{}])[0]
        check("LOINC Terminology", "DiagnosticReport LOINC 11526-1", coding.get("code") == "11526-1" and coding.get("system") == "http://loinc.org", f"code={coding.get('code')}")
        
        rep_status_expected = "preliminary" if expects_human_review else "final"
        check("Governance Status", f"Report status matches review status ({rep_status_expected})", rep.get("status") == rep_status_expected, f"status={rep.get('status')}")

        # Check cross-resource references
        subject_ref = rep.get("subject", {}).get("reference", "")
        patient_ref_ok = (subject_ref == f"Patient/{patient_id}")
        
        results = rep.get("result", [])
        result_refs = [r.get("reference") for r in results]
        obs_injury_id = obs_injury.get("id") if obs_injury else ""
        obs_triage_id = obs_triage.get("id") if obs_triage else ""
        refs_ok = (f"Observation/{obs_injury_id}" in result_refs and f"Observation/{obs_triage_id}" in result_refs)
        
        check("Reference Graph", "Patient and Observation reference graph integrity", patient_ref_ok and refs_ok, f"subject={subject_ref}, results={result_refs}")

        if expects_human_review:
            conclusion = rep.get("conclusion", "")
            check("Human Review Note", "DiagnosticReport conclusion includes Review Notice", "ROUTED FOR HUMAN CLINICAL REVIEW" in conclusion, f"conclusion={conclusion[:80]}...")


def main():
    print("=" * 80)
    print("INDEPENDENT HL7 FHIR R4 VALIDATION SUITE (fhir.resources 8.3.0)")
    print("=" * 80)
    
    # Validate Case A: Active screen case (discordant vision hypotheses, human review required)
    validate_case_bundle(ACTIVE_SCREEN_CASE_ID, expects_human_review=True)
    
    # Validate Case B: Secondary fixture case (concordant)
    validate_case_bundle(FIXTURE_CASE_ID, expects_human_review=False)
    
    print("\n" + "=" * 80)
    print(f"INDEPENDENT FHIR VALIDATION SUMMARY: {passed_checks} / {total_checks} CHECKS PASSED")
    print("=" * 80)
    
    if passed_checks != total_checks:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
