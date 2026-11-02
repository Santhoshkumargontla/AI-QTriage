import os
import sys
import json
import urllib.request
import io
import pypdf
from fhir.resources.bundle import Bundle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.database.connection import get_database

TARGET_ID = "a4a30e57-a232-4ca9-9fc4-3f135f7eff8d"

print("=" * 80)
print(f"CHECKLIST 4: DEEP EXPORT FIDELITY VALIDATION (CASE {TARGET_ID})")
print("=" * 80)

db = get_database()
db_case = db["cases"].find_one({"case_id": TARGET_ID})

# 1. JSON Export
json_url = f"http://127.0.0.1:8000/api/cases/{TARGET_ID}/report/json"
req_json = urllib.request.urlopen(json_url, timeout=5)
assert req_json.getcode() == 200
json_report = json.loads(req_json.read().decode("utf-8"))

# 2. FHIR Export
fhir_url = f"http://127.0.0.1:8000/api/cases/{TARGET_ID}/fhir"
req_fhir = urllib.request.urlopen(fhir_url, timeout=5)
assert req_fhir.getcode() == 200
fhir_bundle_dict = json.loads(req_fhir.read().decode("utf-8"))

# 3. PDF Export
pdf_url = f"http://127.0.0.1:8000/api/cases/{TARGET_ID}/report/pdf"
req_pdf = urllib.request.urlopen(pdf_url, timeout=5)
assert req_pdf.getcode() == 200
pdf_bytes = req_pdf.read()

checks = []

# --- JSON CHECKS ---
checks.append(("JSON: Case ID match", json_report.get("case_id") == TARGET_ID, f"ID={TARGET_ID}"))
checks.append(("JSON: requires_human_review is True", json_report.get("requires_human_review") is True, "True"))
checks.append(("JSON: Triage category HIGH", json_report.get("rule_derived_category") == "HIGH", "HIGH"))
checks.append(("JSON: XGBoost class HIGH", json_report.get("xgboost", {}).get("class") == "HIGH", "HIGH"))
xgb_prob = json_report.get("xgboost", {}).get("probability", 0.0)
checks.append(("JSON: XGBoost prob ~0.665", abs(xgb_prob - 0.6649) < 1e-3, f"{xgb_prob:.4f}"))
checks.append(("JSON: Provenance SYNTHETIC", json_report.get("xgboost", {}).get("data_provenance") == "SYNTHETIC", "SYNTHETIC"))
checks.append(("JSON: Paired clinical count 0", json_report.get("xgboost", {}).get("paired_clinical_samples") == 0, "0"))
checks.append(("JSON: Vision dual finding present", json_report.get("vision", {}).get("yolo_finding") == "Burn" and json_report.get("vision", {}).get("classifier_finding") == "Bruise", "YOLO=Burn vs EffNet=Bruise"))
checks.append(("JSON: Segmentation area is None/N/A", json_report.get("vision", {}).get("affected_area_ratio") is None, "None (withheld)"))

# --- FHIR CHECKS ---
try:
    bundle_obj = Bundle.model_validate(fhir_bundle_dict)
    checks.append(("FHIR: Pydantic R4 model validation", True, "5 resources conform to HL7 FHIR R4 schema"))
except Exception as e:
    checks.append(("FHIR: Pydantic R4 model validation", False, str(e)))

res_types = [entry.resource.get_resource_type() for entry in bundle_obj.entry]
checks.append(("FHIR: Resource types present", res_types == ["Patient", "Device", "DiagnosticReport", "Observation", "Observation"], str(res_types)))

diag_report = next(e.resource for e in bundle_obj.entry if e.resource.get_resource_type() == "DiagnosticReport")
checks.append(("FHIR: DiagnosticReport status preliminary", diag_report.status == "preliminary", "preliminary"))
checks.append(("FHIR: DiagnosticReport subject reference", diag_report.subject.reference == f"Patient/patient-{TARGET_ID[:8]}", diag_report.subject.reference))
result_refs = [r.reference for r in diag_report.result]
checks.append(("FHIR: DiagnosticReport observation refs", len(result_refs) == 2 and f"Observation/obs-injury-{TARGET_ID[:8]}" in result_refs and f"Observation/obs-triage-{TARGET_ID[:8]}" in result_refs, str(result_refs)))
checks.append(("FHIR: DiagnosticReport human review text", "ROUTED FOR HUMAN CLINICAL REVIEW" in diag_report.conclusion, "Contains review notice"))

obs_injury = next(e.resource for e in bundle_obj.entry if e.resource.get_resource_type() == "Observation" and "injury" in e.resource.id)
checks.append(("FHIR: Injury Observation status preliminary", obs_injury.status == "preliminary", "preliminary"))
components = [c.code.text for c in obs_injury.component]
checks.append(("FHIR: Dual hypothesis components", "YOLO11 Region Lesion Detection" in components and "EfficientNetV2 Whole-Image Classification" in components, str(components)))

# --- PDF CHECKS ---
reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
pdf_pages = len(reader.pages)
checks.append(("PDF: Valid PDF stream & length", len(pdf_bytes) > 5000 and pdf_bytes.startswith(b"%PDF"), f"{len(pdf_bytes)} bytes"))
checks.append(("PDF: Total page count", pdf_pages == 2, f"{pdf_pages} pages"))

full_pdf_text = "".join(page.extract_text() or "" for page in reader.pages)
checks.append(("PDF: Contains Target Case ID", TARGET_ID in full_pdf_text, TARGET_ID))
checks.append(("PDF: Contains Human Review warning", "MANDATORY CLINICAL REVIEW REQUIRED" in full_pdf_text or "ROUTED FOR HUMAN CLINICAL REVIEW" in full_pdf_text, "Review warning verified"))
checks.append(("PDF: Vision hypotheses present", "Burn" in full_pdf_text and "Bruise" in full_pdf_text, "Burn & Bruise present"))
checks.append(("PDF: Synthetic research disclosure", "SYNTHETIC" in full_pdf_text or "Synthetic" in full_pdf_text, "Synthetic disclosure verified"))

all_passed = True
for name, passed, detail in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"[{status}] {name:40s}: {detail}")

print("=" * 80)
if all_passed:
    print(f"ALL {len(checks)} EXPORT INTEGRITY CHECKS PASSED.")
else:
    print("SOME EXPORT CHECKS FAILED.")
    sys.exit(1)
