import os
import sys
import json
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.database.connection import get_database

TARGET_ID = "a4a30e57-a232-4ca9-9fc4-3f135f7eff8d"

print("=" * 80)
print(f"CHECKLIST 3: CROSS-CHECK API AND DATABASE (CASE {TARGET_ID})")
print("=" * 80)

db = get_database()
db_case = db["cases"].find_one({"case_id": TARGET_ID})
assert db_case is not None, f"Case {TARGET_ID} not found in MongoDB."

url = f"http://127.0.0.1:8000/api/cases/{TARGET_ID}"
req = urllib.request.urlopen(url, timeout=5)
assert req.getcode() == 200, f"Expected 200 from API, got {req.getcode()}"
api_case = json.loads(req.read().decode("utf-8"))

checks = [
    ("Case ID equality", db_case["case_id"] == api_case["case_id"], f"{db_case['case_id']}"),
    ("Requires human review flag", db_case.get("requires_human_review") is True and api_case.get("requires_human_review") is True, f"DB={db_case.get('requires_human_review')}, API={api_case.get('requires_human_review')}"),
    ("Triage category (rule-derived)", db_case.get("rule_derived_category") == api_case.get("rule_derived_category") == "HIGH", f"{db_case.get('rule_derived_category')}"),
    ("Safety level", db_case.get("safety_guidance_level") == api_case.get("safety_guidance_level") == "HIGH", f"{db_case.get('safety_guidance_level')}"),
    ("XGBoost prediction class", db_case["xgboost_prediction"]["class"] == api_case["xgboost_prediction"]["class"] == "HIGH", "HIGH"),
    ("XGBoost prediction probability", abs(db_case["xgboost_prediction"]["probability"] - api_case["xgboost_prediction"]["probability"]) < 1e-6, f"{db_case['xgboost_prediction']['probability']:.6f}"),
    ("Data provenance class", db_case["xgboost_prediction"]["data_provenance"] == api_case["xgboost_prediction"]["data_provenance"] == "SYNTHETIC", "SYNTHETIC"),
    ("Paired clinical samples count", db_case["xgboost_prediction"].get("paired_clinical_samples") == api_case["xgboost_prediction"].get("paired_clinical_samples") == 0, "0"),
    ("Clinical claim blocked flag", db_case["xgboost_prediction"].get("clinical_claim_blocked") is True and api_case["xgboost_prediction"].get("clinical_claim_blocked") is True, "True"),
    ("YOLO primary finding", db_case["visible_injury"].get("yolo_finding") == api_case["visible_injury"].get("yolo_finding") == "Burn", "Burn"),
    ("EfficientNet classifier finding", db_case["visible_injury"].get("classifier_finding") == api_case["visible_injury"].get("classifier_finding") == "Bruise", "Bruise"),
    ("Affected area unavailable/withheld", db_case["visible_injury"].get("affected_area_ratio") is None and api_case["visible_injury"].get("affected_area_ratio") is None, "None (withheld)"),
    ("Segmentation reliable flag", db_case["visible_injury"].get("segmentation_reliable") is False and api_case["visible_injury"].get("segmentation_reliable") is False, "False"),
    ("Created_at timestamp matches", (db_case.get("created_at").isoformat() if hasattr(db_case.get("created_at"), "isoformat") else str(db_case.get("created_at"))) == api_case.get("created_at"), f"{api_case.get('created_at')}"),
    ("SHAP explanations length", len(db_case.get("shap_explanations", [])) == len(api_case.get("shap_explanations", [])) == 23, "23 features"),
]

all_passed = True
for name, passed, detail in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"[{status}] {name:38s}: {detail}")

print("=" * 80)
if all_passed:
    print(f"ALL {len(checks)} API-DATABASE CORRESPONDENCE CHECKS PASSED.")
else:
    print("SOME CHECKS FAILED.")
    sys.exit(1)
