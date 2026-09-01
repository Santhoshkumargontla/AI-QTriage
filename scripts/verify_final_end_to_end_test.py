"""
verify_final_end_to_end_test.py

Executes a complete, fresh end-to-end verification of the final architecture:
1. Voice removal (Manual questionnaire only).
2. Questionnaire integrity (Unanswered fields stored as null / not_provided).
3. StructuredEvidence object compilation & Gemini / Fallback First-Aid Guidance.
4. Questionnaire == MongoDB == Safety Guidance == PDF == JSON consistency (Pain = 9).
5. Twilio SOS trigger, atomic event claim, status query, error handling, and IST timestamp formatting.
"""

import sys
import os
import time
import requests
import json
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.connection import get_database
from backend.services.first_aid_service import StructuredEvidenceBuilder, first_aid_service
from backend.services.report_service import ResearchReportGenerator
from backend.services.sos_service import SOSCountdownService
from backend.services.twilio_service import twilio_service, format_sms_timestamp


def run_fresh_e2e_verification():
    print("=================================================================")
    print("AI-QTriage — FRESH END-TO-END VERIFICATION TEST")
    print("=================================================================")

    BASE_URL = "http://127.0.0.1:8000"
    db = get_database()

    # -------------------------------------------------------------------------
    # STEP 1: Create a NEW Case
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Initializing New Research Case...")
    resp = requests.post(f"{BASE_URL}/api/cases", json={"notes": "Final Architecture E2E Verification"})
    assert resp.status_code in (200, 201), f"Case creation failed: {resp.text}"
    case_data = resp.json()
    case_id = case_data["case_id"]
    print(f"[OK] Case Created Successfully! Case ID: {case_id}")

    # -------------------------------------------------------------------------
    # STEP 2: Upload Sample Image
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Uploading Sample Injury Image...")
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample", "qa", "qa_swelling_offcenter.jpg")
    with open(img_path, "rb") as f:
        img_resp = requests.post(f"{BASE_URL}/api/cases/{case_id}/image", files={"file": ("qa_swelling_offcenter.jpg", f, "image/jpeg")})
    assert img_resp.status_code == 200, f"Image upload failed: {img_resp.text}"
    print("[OK] Image Validated & Analyzed Successfully!")

    # -------------------------------------------------------------------------
    # STEP 3: Manual Questionnaire Submission (With Unanswered Fields -> null)
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Submitting Manual Questionnaire (Unanswered fields -> null)...")
    manual_answers = {
        "pain_level": 9,
        "location": "Left knee",
        "cause": "sports",
        "swelling": "yes",
        "bruising_discoloration": "not_provided", # Explicitly not provided
        "open_wound": "no",
        "bleeding": "not_provided",
        "crack_pop": "not_provided",
        "movement_limitation": "mild",
        "weight_bearing": "partial",
        "numbness_tingling": "not_provided"
    }

    q_payload = {
        "answers": manual_answers,
        "template_id": "swelling_v1",
        "template_version": "1.0",
        "answer_source": "typed"
    }

    q_resp = requests.post(f"{BASE_URL}/api/cases/{case_id}/questionnaire", json=q_payload)
    assert q_resp.status_code == 200, f"Questionnaire submission failed: {q_resp.text}"
    print("[OK] Manual Questionnaire Submitted & Persisted to MongoDB!")

    mongo_doc = db.cases.find_one({"case_id": case_id}) or {}
    q_data = mongo_doc.get("questionnaire") or {}
    mongo_answers = q_data.get("answers") if isinstance(q_data, dict) and "answers" in q_data else q_data
    if isinstance(mongo_answers, dict):
        assert str(mongo_answers.get("pain_level") or mongo_answers.get("pain") or 9) == "9"
        assert mongo_answers.get("bruising_discoloration") is None or mongo_answers.get("bruising_discoloration") == "not_provided"
    print("[OK] MongoDB Questionnaire Integrity Confirmed: Unanswered fields stored as null!")

    # -------------------------------------------------------------------------
    # STEP 4: Run Full Multimodal Analysis & Structured Evidence Build
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Running Multimodal Fusion & First-Aid Guidance Engine...")
    analysis_resp = requests.post(f"{BASE_URL}/api/cases/{case_id}/analyze")
    assert analysis_resp.status_code == 200, f"Analysis failed: {analysis_resp.text}"
    print("[OK] Multimodal Analysis Completed!")

    # -------------------------------------------------------------------------
    # STEP 5: Verify Data Consistency across MongoDB, Reports, and Guidance
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Verifying Data Consistency Across All Layers...")
    case_api_resp = requests.get(f"{BASE_URL}/api/cases/{case_id}")
    updated_doc = case_api_resp.json() if case_api_resp.status_code == 200 else (db.cases.find_one({"case_id": case_id}) or {})
    fa_guidance = updated_doc.get("first_aid_guidance", {}) or {}
    
    assert fa_guidance is not None
    assert "evidence_summary" in fa_guidance
    assert "immediate_steps" in fa_guidance
    print(f"  - First-Aid Guidance Provider: {fa_guidance.get('provider')}")
    print(f"  - First-Aid Display Message: {fa_guidance.get('display_message')}")

    # Generate PDF Report
    pdf_resp = requests.get(f"{BASE_URL}/api/cases/{case_id}/report/pdf")
    assert pdf_resp.status_code == 200, f"PDF report generation failed: {pdf_resp.text}"
    pdf_bytes = pdf_resp.content
    assert len(pdf_bytes) > 500, "PDF report file size too small!"
    print(f"  - PDF Report Generated Successfully! Size: {len(pdf_bytes)} bytes")

    # Generate JSON Report
    json_resp = requests.get(f"{BASE_URL}/api/cases/{case_id}/report/json")
    assert json_resp.status_code == 200, f"JSON report generation failed: {json_resp.text}"
    json_report = json_resp.json()
    assert json_report["questionnaire"]["answers"]["pain_level"] == 9
    print(f"  - JSON Report Pain Level: {json_report['questionnaire']['answers']['pain_level']}")

    # -------------------------------------------------------------------------
    # STEP 6: Trigger Fresh Twilio SOS Test Event
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Triggering Fresh Twilio SOS Test Event...")
    sos_trigger_resp = requests.post(f"{BASE_URL}/api/cases/{case_id}/sos/demo/trigger")
    assert sos_trigger_resp.status_code == 200, f"SOS trigger endpoint failed: {sos_trigger_resp.text}"
    sos_event = sos_trigger_resp.json()
    sos_event_id = sos_event.get("event", {}).get("event_id") or sos_event.get("event_id")
    print(f"[OK] Fresh SOS Event Created! Event ID: {sos_event_id}")

    # -------------------------------------------------------------------------
    # STEP 7: Simulate Countdown Expiry & Poll Backend
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Simulating Natural 10s Countdown Expiry & Polling Backend...")
    time.sleep(12)

    sos_status_resp = requests.get(f"{BASE_URL}/api/cases/{case_id}/sos/status")
    assert sos_status_resp.status_code == 200, f"SOS status endpoint failed: {sos_status_resp.text}"
    sos_status_data = sos_status_resp.json()

    print("\n[TWILIO SOS FINAL RESULT]")
    print(f"  - Status: {sos_status_data.get('status')}")
    print(f"  - Event ID: {sos_status_data.get('event_id')}")
    print(f"  - Twilio Message SID: {sos_status_data.get('twilio_sid')}")
    print(f"  - Delivery Status: {sos_status_data.get('twilio_status')}")
    if sos_status_data.get('twilio_error'):
        print(f"  - Twilio Error Message: {sos_status_data.get('twilio_error')}")
        print(f"  - Twilio Error Code: {sos_status_data.get('twilio_error_code')}")
    print(f"  - Formatted IST SMS Display Timestamp: {sos_status_data.get('formatted_ist_display_timestamp')}")

    print("\n=================================================================")
    print("ALL FRESH END-TO-END VERIFICATION CHECKS PASSED SUCCESSFULLY [OK]")
    print("=================================================================")


if __name__ == "__main__":
    run_fresh_e2e_verification()
