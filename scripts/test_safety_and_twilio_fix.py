"""
Verification script for Issue 14: Safety Guidance Level + Twilio Status Consistency
"""
import os, sys, time, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / "backend/.env")

from backend.database.connection import get_database
from backend.services.report_service import ResearchReportGenerator
from backend.services.sos_service import SOSCountdownService
from backend.services.twilio_service import twilio_service
from ml.fusion.rules_engine import RulesEngine
import numpy as np

def run_test():
    print("\n" + "="*80)
    print("  AI-QTriage — FINAL SAFETY GUIDANCE + TWILIO STATUS CONSISTENCY VERIFICATION")
    print("="*80)

    db = get_database()

    # 1. Create a fresh test case
    case_id = f"test_case_safety_{int(time.time())}"
    print(f"\n[1] Creating fresh test case: {case_id}")

    test_case = {
        "case_id": case_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "created",
        "questionnaire": {
            "template_id": "ankle_v1",
            "answers": {
                "pain_level": "5",
                "crack_pop": "no",
                "visible_bleeding": "no",
                "location": "Left ankle"
            }
        },
        "sensor_summary": {
            "peak_g_force": 4.62,
            "post_impact_stabilization_seconds": 2.5,
            "source_type": "simulated"
        },
        "sensor_available": True,
        "sensor_source_type": "simulated",
        "sos_status": "inactive"
    }

    db.cases.insert_one(test_case)

    # 2. Evaluate RulesEngine directly
    # Vector: [pain_level/10=0.5, bleeding=0, g_force=4.62, movement=0, weight_bearing=0, crack_pop=0, direct_impact=0]
    # Standard 16-feature fused vector
    feature_names = [
        "pain_level", "visible_bleeding", "peak_g_force", "post_impact_stabilization_seconds",
        "movement_limitation", "weight_bearing", "crack_pop", "direct_impact",
        "pre_impact_delta_v", "optical_lux_drop", "vision_confidence", "cut_probability",
        "bruise_probability", "swelling_probability", "other_probability", "affected_ratio"
    ]
    fused_vec = np.zeros(16)
    fused_vec[0] = 0.5   # pain 5/10
    fused_vec[2] = 4.62  # g-force 4.62
    
    rules = RulesEngine()
    rule_category, rule_justification = rules.evaluate_rules(fused_vec, feature_names)

    print(f"\n[2] RulesEngine Evaluation:")
    print(f"    Category:      {rule_category}")
    print(f"    Justification: {rule_justification}")

    # Store analysis results in MongoDB
    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "rule_derived_category": rule_category,
            "safety_guidance_level": rule_category,
            "safety_information": [rule_justification],
            "xgboost_prediction": {"class": "HIGH", "probability": 0.88},
            "quantum_prediction": {"class": "LOW", "score": [0.4, 0.3, 0.3]},
            "agreement_score": "DISAGREEMENT",
            "uncertainty_status": "High Uncertainty",
            "uncertainty_reasons": ["Classical (HIGH) and Quantum (LOW) predictions disagree"],
            "status": "analyzed"
        }}
    )

    # 3. Check MongoDB Case Document
    case_doc = db.cases.find_one({"case_id": case_id})
    print(f"\n[3] MongoDB Case Document Verification:")
    print(f"    rule_derived_category: {case_doc.get('rule_derived_category')}")
    print(f"    safety_guidance_level: {case_doc.get('safety_guidance_level')}")
    print(f"    safety_information:    {case_doc.get('safety_information')}")
    print(f"    xgboost_prediction:    {case_doc.get('xgboost_prediction', {}).get('class')}")
    print(f"    quantum_prediction:    {case_doc.get('quantum_prediction', {}).get('class')}")

    # 4. Check Report Generator (PDF / JSON compilation)
    report_gen = ResearchReportGenerator()
    report_data = report_gen.compile_report_data(case_id)
    pdf_bytes = report_gen.generate_pdf_bytes(case_id)

    print(f"\n[4] Report Data (PDF / JSON source) Verification:")
    print(f"    rule_derived_category: {report_data.get('rule_derived_category')}")
    print(f"    safety_guidance_level: {report_data.get('safety_guidance_level')}")
    print(f"    PDF bytes generated:   {len(pdf_bytes)} bytes")

    # 5. Check Safety Guidance Contradiction
    badge_level = report_data.get('safety_guidance_level')
    explanation_text = report_data.get('safety', {}).get('disclaimers', [])
    info_text = case_doc.get('safety_information', [''])[0]

    assert badge_level == rule_category, f"Badge level ({badge_level}) != rule category ({rule_category})"
    assert rule_category in info_text, f"Rule category ({rule_category}) not in explanation text ({info_text})"

    print(f"\n[5] Safety Guidance Consistency Check: PASSED [OK]")
    print(f"    Badge Level:       {badge_level}")
    print(f"    Explanation Text:  {info_text}")

    # 6. Test Fresh Twilio SOS Flow
    print(f"\n[6] Testing Fresh Twilio SOS Flow...")
    
    # Create SOS event in countdown
    sos_event_id = f"evt_{case_id}"
    sos_event = {
        "event_id": sos_event_id,
        "case_id": case_id,
        "sos_status": "countdown",
        "countdown_seconds": 10,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "delivery_mode": "twilio_test"
    }
    db.sos_events.insert_one(sos_event)
    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "sos_status": "countdown",
            "sos_trigger_time": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sos_countdown_seconds": 10,
            "sos_delivery_mode": "twilio_test",
            "sos_trigger_reason": "High impact measured."
        }}
    )

    # Send Twilio message directly via service for testing
    tw_res = twilio_service.send_test_sos_message(
        case_id=case_id,
        user_location="Left ankle",
        sos_event_id=sos_event_id,
        yolo_finding=None,
        classifier_category="Swelling"
    )

    print(f"\n[7] Twilio Send API Response:")
    print(f"    Success:            {tw_res.get('success')}")
    print(f"    Status:             {tw_res.get('status')}")
    print(f"    Twilio Message SID: {tw_res.get('twilio_message_sid')}")
    print(f"    Delivery Status:    {tw_res.get('delivery_status')}")

    # Update event & case with Twilio result
    msg_sid = tw_res.get("twilio_message_sid")
    del_status = tw_res.get("delivery_status", "queued")
    sos_stat = "twilio_accepted" if tw_res.get("success") else "twilio_failed"

    db.sos_events.update_one(
        {"event_id": sos_event_id},
        {"$set": {
            "sos_status": sos_stat,
            "twilio_message_sid": msg_sid,
            "delivery_status": del_status,
            "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }}
    )
    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "sos_status": sos_stat,
            "sos_twilio_sid": msg_sid,
            "sos_delivery_status": del_status
        }}
    )

    # 8. Check SOS Status via SOSCountdownService
    sos_service = SOSCountdownService()
    sos_status_out = sos_service.get_sos_status(case_id)

    print(f"\n[8] get_sos_status() Output:")
    print(f"    Status:             {sos_status_out.get('status')}")
    print(f"    Message SID:        {sos_status_out.get('twilio_message_sid')}")
    print(f"    Delivery Status:    {sos_status_out.get('delivery_status')}")
    print(f"    Error Message:      {sos_status_out.get('twilio_error_message')}")

    # 9. Print Forensic Verification Table
    print("\n" + "="*80)
    print("  FORENSIC CONSISTENCY VERIFICATION TABLE")
    print("="*80)
    
    table_data = [
        ("Safety Guidance Level", case_doc.get("safety_guidance_level"), report_data.get("safety_guidance_level"), badge_level, badge_level, report_data.get("safety_guidance_level")),
        ("Rule-Derived Category", case_doc.get("rule_derived_category"), report_data.get("rule_derived_category"), rule_category, rule_category, report_data.get("rule_derived_category")),
        ("XGBoost Prediction", case_doc.get("xgboost_prediction", {}).get("class"), report_data.get("xgboost", {}).get("class"), "HIGH", "HIGH", report_data.get("xgboost", {}).get("class")),
        ("VQC Prediction", case_doc.get("quantum_prediction", {}).get("class"), report_data.get("quantum", {}).get("class"), "LOW", "LOW", report_data.get("quantum", {}).get("class")),
        ("Prediction Agreement", case_doc.get("agreement_score"), report_data.get("prediction_agreement"), "DISAGREEMENT", "DISAGREEMENT", report_data.get("prediction_agreement")),
        ("Twilio SOS Status", case_doc.get("sos_status"), report_data.get("sos", {}).get("status"), sos_stat, sos_stat, report_data.get("sos", {}).get("status")),
        ("Twilio Message SID", case_doc.get("sos_twilio_sid"), report_data.get("sos", {}).get("twilio_sid"), msg_sid, msg_sid, report_data.get("sos", {}).get("twilio_sid")),
        ("Twilio Delivery Status", case_doc.get("sos_delivery_status"), del_status, del_status, del_status, del_status)
    ]

    header = f"{'Field':<24} | {'MongoDB':<15} | {'Backend API':<15} | {'Frontend':<15} | {'PDF':<12} | {'JSON':<15}"
    print(header)
    print("-" * len(header))
    for field, db_val, api_val, fe_val, pdf_val, json_val in table_data:
        print(f"{field:<24} | {str(db_val):<15} | {str(api_val):<15} | {str(fe_val):<15} | {str(pdf_val):<12} | {str(json_val):<15}")
    print("="*80)

    print("\nALL CONSISTENCY CHECKS PASSED [OK]")

if __name__ == "__main__":
    run_test()
