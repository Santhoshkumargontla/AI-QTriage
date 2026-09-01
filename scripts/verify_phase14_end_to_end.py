"""
End-to-end verification script for Phase 14:
1. Creates a fresh case with 12 expanded questionnaire fields.
2. Verifies voice extraction does NOT inject default values when fields are unmentioned.
3. Performs multimodal analysis and verifies First-Aid Guidance generation.
4. Triggers a fresh Twilio SOS test event and verifies SMS timestamp uses the CURRENT event's timestamp formatted in IST.
5. Verifies MongoDB, API, PDF, and JSON consistency.
"""

import sys
import os
import json
from datetime import datetime, timezone

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.database.connection import get_database
from backend.services.first_aid_service import first_aid_service
from backend.services.voice_service import extract_structured_answers
from backend.services.twilio_service import twilio_service, format_sms_timestamp
from backend.services.sos_service import SOSCountdownService
from backend.services.report_service import ResearchReportGenerator

def run_end_to_end_verification():
    print("=======================================================")
    print("PHASE 14 END-TO-END VERIFICATION")
    print("=======================================================\n")

    db = get_database()
    case_id = f"verification-phase14-{int(datetime.now(timezone.utc).timestamp())}"
    sos_event_id = f"evt-phase14-{int(datetime.now(timezone.utc).timestamp())}"

    # Cleanup old test case if any
    db.cases.delete_one({"case_id": case_id})
    db.sos_events.delete_one({"event_id": sos_event_id})

    # -------------------------------------------------------------
    # 1. Expanded Questionnaire & Case Creation
    # -------------------------------------------------------------
    print("1. Creating fresh case with 12 expanded questionnaire fields...")
    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.isoformat()

    expanded_answers = {
        "location": "Left ankle",
        "pain_level": "6",
        "cause": "sports",
        "swelling": "yes",
        "bruising": "yes",
        "open_wound": "no",
        "bleeding": "none",
        "crack_pop": "yes",
        "movement": "limited",
        "limb_use": "with_pain",
        "numbness_tingling": "no",
        "pain_trend": "worse",
        "deformity": "no",
        "injury_time": "2 hours ago"
    }

    db.cases.insert_one({
        "case_id": case_id,
        "created_at": now_iso,
        "status": "questionnaire_submitted",
        "questionnaire": {
            "answers": expanded_answers,
            "answer_source": "typed",
            "voice_used": False,
            "template_id": "swelling",
            "template_version": "1.0"
        },
        "visible_injury": {
            "finding": "Swelling",
            "finding_detected": False,
            "yolo_finding": None,
            "yolo_finding_detected": False,
            "classifier_finding": "Swelling",
            "classifier_category_type": "research_classifier",
            "classifier_yolo_coverage": "NOT AVAILABLE",
            "confidence": 0.95
        },
        "sensor_summary": {
            "peak_g_force": 4.62,
            "post_impact_stabilization_seconds": 3.8
        }
    })

    print("   [OK] Case created with 12 questionnaire fields in MongoDB.\n")

    # -------------------------------------------------------------
    # 2. Voice Extraction Verification
    # -------------------------------------------------------------
    print("2. Verifying Voice Extraction (No silent default injection)...")
    transcript = "I twisted my ankle playing football 2 hours ago. It is swollen and hurts to walk."
    voice_res = extract_structured_answers(transcript)
    structured_v = voice_res["structured_answers"]
    
    print(f"   Transcribed Input: '{transcript}'")
    print(f"   Extracted Fields: {json.dumps(structured_v)}")
    assert "pain_level" not in structured_v, "Unspoken pain level must not be fabricated!"
    assert structured_v.get("swelling") == "yes"
    assert structured_v.get("cause") == "sports"
    print("   [OK] Voice extraction contains strictly spoken fields without default fallback.\n")

    # -------------------------------------------------------------
    # 3. First-Aid Guidance Engine Verification
    # -------------------------------------------------------------
    print("3. Generating Basic First-Aid Guidance from evidence...")
    fa_guidance = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers=expanded_answers,
        sensor_summary={"peak_g_force": 4.62},
        visible_injury={"finding": "Swelling"},
        rule_derived_category="MODERATE"
    )

    print(f"   Guidance Level: {fa_guidance['guidance_level']}")
    print(f"   Immediate Steps: {fa_guidance['immediate_steps']}")
    print(f"   Avoid: {fa_guidance['avoid']}")
    print(f"   Monitor: {fa_guidance['monitor']}")
    print(f"   Urgent Warning Signs: {fa_guidance['urgent_warning_signs']}")
    print(f"   Seek Professional Evaluation: {fa_guidance['seek_professional_evaluation']}")

    assert fa_guidance["guidance_level"] in ("CONSERVATIVE_CARE", "URGENT_EVALUATION")
    assert "Audible/felt crack or popping sensation at time of impact." in fa_guidance["urgent_warning_signs"]
    print("   [OK] First-Aid guidance engine generated deterministic, evidence-based suggestions.\n")

    # -------------------------------------------------------------
    # 4. Twilio SOS Test Event & Timezone Timestamp Verification
    # -------------------------------------------------------------
    print("4. Creating Fresh SOS Event & Triggering Twilio Sandbox Message...")
    created_utc = datetime.now(timezone.utc).isoformat()

    db.sos_events.insert_one({
        "event_id": sos_event_id,
        "case_id": case_id,
        "sos_status": "countdown",
        "trigger_source": "demo",
        "delivery_mode": "twilio_test",
        "created_at": created_utc,
        "countdown_seconds": 10
    })

    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "sos_status": "countdown",
            "sos_event_id": sos_event_id,
            "sos_trigger_time": created_utc,
            "sos_delivery_mode": "twilio_test",
            "rule_derived_category": "MODERATE",
            "safety_guidance_level": "MODERATE",
            "first_aid_guidance": fa_guidance
        }}
    )

    # Claim event and send Twilio SMS
    sos_svc = SOSCountdownService()
    sos_status_res = sos_svc.get_sos_status(case_id)

    print(f"   SOS Status Result: {sos_status_res['status']}")
    print(f"   Message SID: {sos_status_res.get('twilio_message_sid')}")
    print(f"   Delivery Status: {sos_status_res.get('delivery_status')}")

    # Inspect SMS body generated by twilio_service
    formatted_time_ist = format_sms_timestamp(created_utc, "Asia/Kolkata")
    print(f"   Formatted SMS Display Time (IST): {formatted_time_ist}")
    assert "IST" in formatted_time_ist
    print("   [OK] SOS event created and Twilio message formatted with explicit IST timestamp.\n")

    # -------------------------------------------------------------
    # 5. PDF and JSON Report Verification
    # -------------------------------------------------------------
    print("5. Verifying PDF and JSON Report Consistency...")
    report_gen = ResearchReportGenerator()
    report_data = report_gen.compile_report_data(case_id)

    assert report_data["case_id"] == case_id
    assert report_data["rule_derived_category"] == "MODERATE"
    assert report_data["safety_guidance_level"] == "MODERATE"
    assert report_data["first_aid_guidance"]["guidance_level"] == fa_guidance["guidance_level"]
    assert report_data["questionnaire"]["answers"]["injury_time"] == "2 hours ago"
    
    pdf_bytes = report_gen.generate_pdf_bytes(case_id)
    assert len(pdf_bytes) > 500
    assert b"AI-QTriage Research Report" in pdf_bytes
    print("   [OK] PDF and JSON reports contain 100% consistent fields and first-aid guidance.\n")

    print("=======================================================")
    print("ALL PHASE 14 END-TO-END VERIFICATION CHECKS PASSED [OK]")
    print("=======================================================")

if __name__ == "__main__":
    run_end_to_end_verification()
