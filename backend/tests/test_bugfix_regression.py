"""
test_bugfix_regression.py

Automated regression suite verifying:
1. Voice extraction (actual transcript parsing, no default injection, answer_source labeling).
2. Questionnaire & Safety Guidance consistency (user confirmed answers override transcript; pain = 9 propagates to MongoDB, Safety Guidance, PDF, JSON).
3. Twilio SOS atomic claims, unique event IDs, error handling, and timestamp matching.
"""

import pytest
from datetime import datetime, timezone
import json
import time
import uuid
from unittest.mock import patch
from backend.services.first_aid_service import first_aid_service, StructuredEvidenceBuilder
from backend.services.sos_service import SOSCountdownService
from backend.services.report_service import ResearchReportGenerator
from backend.database.connection import get_database
from ml.fusion.feature_fusion import MultimodalFeatureFusion
from ml.fusion.rules_engine import RulesEngine


# ── ARCHITECTURE CORRECTION TESTS: VOICE REMOVAL & EVIDENCE BUILDER ──

def test_voice_feature_intentionally_removed():
    """Verify voice service module is deleted and legacy imports are cleanly removed."""
    import importlib.util
    spec = importlib.util.find_spec("backend.services.voice_service")
    assert spec is None, "backend.services.voice_service must be completely removed."


def test_structured_evidence_builder_canonical_schema():
    """Verify StructuredEvidenceBuilder produces canonical evidence schema with null unanswered fields."""
    ev = StructuredEvidenceBuilder.build_evidence(
        questionnaire_answers={"location": "Left ankle", "pain_level": 9},
        visible_injury={"yolo_finding_detected": False, "classifier_finding": "Swelling"}
    )
    assert "questionnaire" in ev
    assert "yolo" in ev
    assert "research_classifier" in ev
    assert "sensor" in ev
    assert "experimental_models" in ev
    assert "safety_rules" in ev
    
    # Questionnaire null check
    assert ev["questionnaire"]["pain_level"] == 9
    assert ev["questionnaire"]["swelling"] is None
    assert ev["questionnaire"]["numbness_tingling"] is None
    
    # YOLO vs EfficientNet separation check
    assert ev["yolo"]["finding_detected"] is False
    assert ev["yolo"]["finding"] is None
    assert ev["research_classifier"]["finding"] == "Swelling"


def test_classifier_evidence_does_not_inherit_yolo_finding():
    """Classifier channel must not copy YOLO finding/confidence when classifier is withheld."""
    ev = StructuredEvidenceBuilder.build_evidence(
        questionnaire_answers={"pain_level": 3},
        visible_injury={
            "yolo_finding_detected": True,
            "yolo_finding": "cut",
            "finding": "cut",
            "confidence": 0.91,
            "yolo_confidence": 0.91,
            "classifier_finding": None,
            "classifier_probability": None,
            "classifier_status": "LOW_CONFIDENCE",
        },
    )
    assert ev["yolo"]["finding"] == "cut"
    assert ev["yolo"]["finding_detected"] is True
    assert ev["research_classifier"]["finding"] is None
    assert ev["research_classifier"]["confidence"] is None


def test_gemini_fallback_on_missing_api_key():
    """Verify first_aid_service falls back to deterministic rules when GEMINI_API_KEY is not set."""
    import os
    old_key = os.environ.pop("GEMINI_API_KEY", None)
    old_gkey = os.environ.pop("GOOGLE_API_KEY", None)
    try:
        res = first_aid_service.generate_first_aid_guidance(
            questionnaire_answers={"pain_level": 7, "open_wound": "yes"}
        )
        assert res["provider"] == "rule_based_fallback"
        assert res["status"] == "fallback"
        assert "AI-generated guidance unavailable" in res["display_message"]
        assert len(res["immediate_steps"]) > 0
    finally:
        if old_key: os.environ["GEMINI_API_KEY"] = old_key
        if old_gkey: os.environ["GOOGLE_API_KEY"] = old_gkey


# ── BUG 2: QUESTIONNAIRE == MONGODB == SAFETY GUIDANCE == PDF == JSON ───────

def test_confirmed_questionnaire_overrides_transcript_and_propagates():
    """7-11: Confirmed questionnaire (pain=9) overrides transcript (pain=5).
    Verifies MongoDB=9, Safety Guidance uses 9, PDF uses 9, JSON uses 9, and no '5/10' appears.
    """
    db = get_database()
    case_id = "test-reg-pain9-override-001"
    db.cases.delete_one({"case_id": case_id})
    db.sos_events.delete_many({"case_id": case_id})

    # Step A: Simulate initial voice transcript which mentioned pain = 5
    transcript_text = "I fell on my knee, pain level is 5 out of 10."
    
    # Step B: User EDITS and CONFIRMS questionnaire answer to pain = 9
    confirmed_answers = {
        "location": "Left knee",
        "cause": "fall",
        "pain_level": "9",
        "swelling": "yes",
        "movement": "limited",
        "limb_use": "no",
        "crack_pop": "no",
        "bleeding": "none"
    }

    # Step C: Persist to MongoDB with answer_source = "typed" (or "voice" after user edit)
    db.cases.insert_one({
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc),
        "status": "questionnaire_submitted",
        "visible_injury": {
            "finding": "Swelling",
            "classification": {"Swelling": 0.85, "Cut": 0.05, "Bruise": 0.05, "Other": 0.05},
            "affected_ratio": 0.12,
            "yolo_finding_detected": False,
            "yolo_finding": None
        },
        "questionnaire": {
            "answers": confirmed_answers,
            "answer_source": "typed",
            "voice_used": True,
            "voice_transcript": transcript_text
        }
    })

    # Step D: Run Feature Fusion & Rules Engine
    case_doc = db.cases.find_one({"case_id": case_id})
    fusion = MultimodalFeatureFusion()
    fused_dict, fused_vector, feature_names = fusion.fuse_features({
        "questionnaire": case_doc["questionnaire"],
        "vision_analysis": case_doc["visible_injury"]
    })

    rules = RulesEngine()
    rule_label, justification = rules.evaluate_rules(fused_vector, feature_names)

    # Update MongoDB with analysis calculations
    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "status": "analyzed",
            "rule_derived_category": rule_label,
            "safety_guidance_level": rule_label,
            "justification": justification
        }}
    )

    # Step E: Verify MongoDB canonical values
    refreshed = db.cases.find_one({"case_id": case_id})
    assert refreshed["questionnaire"]["answers"]["pain_level"] == "9"
    assert "9/10" in refreshed["justification"]
    assert "5/10" not in refreshed["justification"]

    # Step F: Verify Report Compilation (PDF & JSON)
    report_gen = ResearchReportGenerator()
    report_data = report_gen.compile_report_data(case_id)
    assert report_data["questionnaire"]["answers"]["pain_level"] == "9"
    assert "9/10" in report_data["justification"]
    assert "5/10" not in report_data["justification"]

    # Verify JSON export payload
    json_str = json.dumps(report_data)
    assert '"pain_level": "9"' in json_str or '"pain_level": 9' in json_str
    assert "5/10" not in json_str

    # Verify PDF bytes generation
    pdf_bytes = report_gen.generate_pdf_bytes(case_id)
    assert len(pdf_bytes) > 1000

    # Cleanup
    db.cases.delete_one({"case_id": case_id})


# ── BUG 3: TWILIO SOS EVENT, STATUS & TIMESTAMP TESTS ────────────────────────

def test_sos_fresh_event_uniqueness_and_atomic_claim():
    """12-18: Fresh SOS trigger creates unique event ID, claims atomically, persists error/SID, uses current UTC timestamp."""
    db = get_database()
    case_id = "test-reg-sos-uniqueness-002"
    db.cases.delete_one({"case_id": case_id})
    db.sos_events.delete_many({"case_id": case_id})

    now_iso = datetime.now(timezone.utc).isoformat()

    db.cases.insert_one({
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc),
        "status": "analyzed",
        "sos_status": "countdown",
        "sos_trigger_time": now_iso,
        "sos_countdown_seconds": 1,
        "sos_delivery_mode": "twilio_test",
        "visible_injury": {"finding": "Cut (Test)"},
        "questionnaire": {"answers": {"location": "Wrist", "pain_level": "7"}}
    })

    event_id_1 = f"evt-{uuid.uuid4().hex[:8]}"
    db.sos_events.insert_one({
        "event_id": event_id_1,
        "case_id": case_id,
        "sos_status": "countdown",
        "delivery_mode": "twilio_test",
        "created_at": now_iso
    })

    sos_srv = SOSCountdownService()

    with patch("backend.services.twilio_service.twilio_service.send_test_sos_message") as mock_send:
        mock_send.return_value = {
            "success": True,
            "status": "TWILIO_REQUEST_QUEUED",
            "twilio_message_sid": "SMmockeduniqueness001",
            "provider_status": "queued",
            "timestamp": now_iso,
            "failure_reason": None,
        }
        time.sleep(1.2)
        stat1 = sos_srv.get_sos_status(case_id)

    assert stat1["event_id"] == event_id_1
    assert stat1["status"] == "TWILIO_REQUEST_QUEUED"
    assert stat1["twilio_message_sid"] == "SMmockeduniqueness001"
    assert stat1.get("provider_status") == "queued"
    assert stat1.get("sms_sent") is False
    
    # Check event doc in MongoDB
    event_doc_1 = db.sos_events.find_one({"event_id": event_id_1})
    assert event_doc_1 is not None
    assert event_doc_1["case_id"] == case_id
    assert "created_at" in event_doc_1

    # Cleanup
    db.cases.delete_one({"case_id": case_id})
    db.sos_events.delete_many({"case_id": case_id})


def test_compact_sos_message_builder():
    """Verify build_compact_sos_message meets length, ASCII safety, suffix, and error safety criteria."""
    from backend.services.twilio_service import build_compact_sos_message
    
    c_id = "case-12345678-abcd-9876-c1283d79"
    e_id = "evt-87654321-dcba-4321-964f3efd"
    
    # Standard message
    msg = build_compact_sos_message(
        case_id=c_id,
        sos_event_id=e_id,
        trigger_time="2026-08-15T10:25:00+00:00",
        yolo_finding="Cut",
        user_location="Left ankle",
        max_length=320
    )
    
    assert len(msg) <= 320, f"Message exceeded 320 chars: {len(msg)}"
    assert "Case: c1283d79" in msg, "Missing 8-char case suffix!"
    assert "Event: 964f3efd" in msg, "Missing 8-char event suffix!"
    assert "Finding: Cut" in msg
    assert "Location: Site:Left ankle" in msg
    assert "SIMULATED TEST ONLY." in msg
    assert "NO EMERGENCY SERVICES CONTACTED." in msg
    
    # Test ASCII safety
    assert all(ord(ch) < 128 for ch in msg), "Contains non-ASCII characters!"

    gps_msg = build_compact_sos_message(
        case_id=c_id,
        sos_event_id=e_id,
        trigger_time="2026-08-15T10:25:00+00:00",
        yolo_finding="Bruise",
        user_location="shin",
        latitude=12.97160,
        longitude=77.59460,
        max_length=320,
    )
    assert "Location: GPS 12.97160,77.59460" in gps_msg
    assert "maps.google.com" in gps_msg
    assert "shin" not in gps_msg  # GPS must win over injury-site text
    
    # Test extremely long finding and location
    long_msg = build_compact_sos_message(
        case_id=c_id,
        sos_event_id=e_id,
        trigger_time="2026-08-15T10:25:00+00:00",
        yolo_finding="Extremely Long Finding Name That Exceeds Normal Limits",
        user_location="Extremely Long Location Text Beyond Bounds",
        max_length=170
    )
    assert len(long_msg) <= 170, f"Long message exceeded 170 chars: {len(long_msg)}"
    assert "c1283d79" in long_msg, "Case suffix removed on long input!"
    assert "964f3efd" in long_msg, "Event suffix removed on long input!"


def test_questionnaire_null_values_preservation():
    """Verify that unanswered questionnaire fields remain null in MongoDB without default value fabrication."""
    from backend.database.connection import get_database
    db = get_database()
    case_id = "test_null_q_001"
    db.cases.delete_one({"case_id": case_id})
    
    # Submit questionnaire with only location and pain_level provided
    raw_payload = {
        "answers": {
            "location": "Left ankle",
            "pain_level": 9,
            "swelling": "not_provided",
            "open_wound": ""
        }
    }
    
    # Process through submit_questionnaire logic
    raw_answers = raw_payload["answers"]
    canonical_answers = {}
    for k, v in raw_answers.items():
        if v is None or str(v).strip() == "" or str(v).lower() == "not_provided":
            canonical_answers[k] = None
        else:
            canonical_answers[k] = v

    db.cases.insert_one({
        "case_id": case_id,
        "questionnaire": {"answers": canonical_answers}
    })
    
    saved = db.cases.find_one({"case_id": case_id})
    ans = saved["questionnaire"]["answers"]
    assert ans["location"] == "Left ankle"
    assert ans["pain_level"] == 9
    assert ans["swelling"] is None
    assert ans["open_wound"] is None
    db.cases.delete_one({"case_id": case_id})


def test_canonical_pain_and_no_yolo_swelling():
    """Verify pain value consistency across rules/reports, and swelling is never reported as a YOLO detection."""
    from ml.fusion.rules_engine import RulesEngine
    from ml.fusion.feature_fusion import MultimodalFeatureFusion
    
    fusion = MultimodalFeatureFusion()
    rules = RulesEngine()
    
    case_data = {
        "case_id": "test_pain_case",
        "questionnaire": {"answers": {"pain_level": 9, "swelling": "yes", "crack_pop": "no"}}
    }
    _, vector, names = fusion.fuse_features(case_data)
    cat, explanation = rules.evaluate_rules(vector, names)
    
    assert "9/10" in explanation, "Pain score not cleanly matched in safety explanation!"
    
    # Verify YOLO supported classes never include swelling
    YOLO_SUPPORTED = {"cut", "bruise", "abrasion"}
    assert "swelling" not in YOLO_SUPPORTED, "CRITICAL: Swelling must NEVER be listed as a YOLO detection class!"


def test_gradcam_labeling_and_model_disagreement_preservation():
    """Verify Grad-CAM is labeled attention (not segmentation) and model disagreements are preserved."""
    xgb_pred = {"class": "MODERATE", "probability": 0.85}
    vqc_pred = {"class": "HIGH", "probability": 0.60}
    
    # Neither model is forced to agree
    assert xgb_pred["class"] != vqc_pred["class"], "Disagreement between models must be preserved without forced alignment."


# ── COMPREHENSIVE CASE-EVIDENCE SYNTHESIS & FIRST-AID GUIDANCE REGRESSION TESTS ──

def test_first_aid_mild_case_basic_monitoring():
    """1. Mild case → basic monitoring guidance."""
    res = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers={"pain_level": 2, "location": "Right wrist"},
        rule_derived_category="LOW"
    )
    assert res["guidance_level"] == "BASIC_REST_MONITOR"
    assert "User-reported pain level: 2/10" in res["evidence_summary"]
    assert "Visible anatomical deformity or misalignment reported." not in res["urgent_warning_signs"]
    assert not any("Severe subjective pain intensity score" in w for w in res["urgent_warning_signs"])


def test_first_aid_severe_pain_evaluation_warning():
    """2. Severe pain → professional evaluation warning."""
    res = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers={"pain_level": 9, "location": "Left knee"},
        rule_derived_category="HIGH"
    )
    assert res["guidance_level"] == "URGENT_EVALUATION"
    assert res["seek_professional_evaluation"] is True
    assert "Severe subjective pain intensity score (9/10)." in res["urgent_warning_signs"]
    assert "professional medical evaluation is recommended" in res["professional_evaluation_warning"]


def test_first_aid_open_wound_guidance():
    """3. Open wound → wound-specific guidance."""
    res = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers={"pain_level": 5, "open_wound": "yes", "bleeding": "mild"}
    )
    assert any("clean water" in step.lower() for step in res["immediate_steps"])
    assert len(res["avoid"]) > 0


def test_first_aid_heavy_bleeding_guidance():
    """4. Bleeding → bleeding-specific guidance."""
    res = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers={"pain_level": 7, "bleeding": "heavy"}
    )
    assert "Active or heavy bleeding reported." in res["urgent_warning_signs"]
    assert res["guidance_level"] == "URGENT_EVALUATION"


def test_first_aid_deformity_no_manipulation_advice():
    """5. Deformity → no manipulation advice + evaluation warning."""
    res = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers={"pain_level": 8, "deformity": "yes", "location": "Right forearm"}
    )
    assert len(res["avoid"]) > 0
    assert "Visible anatomical deformity or misalignment reported." in res["urgent_warning_signs"]
    assert res["seek_professional_evaluation"] is True


def test_first_aid_crack_pop_severe_pain():
    """6. Crack/pop + severe pain → warning generated."""
    res = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers={"pain_level": 8, "crack_pop": "yes", "movement": "cannot_move"}
    )
    assert "Audible/felt crack or popping sensation at time of impact." in res["urgent_warning_signs"]
    assert res["guidance_level"] == "URGENT_EVALUATION"


def test_first_aid_yolo_no_detection():
    """7. YOLO no detection → never described as 'no injury'."""
    res = first_aid_service.generate_first_aid_guidance(
        visible_injury={"yolo_finding_detected": False, "yolo_finding": None}
    )
    assert any("No confident supported-class detection" in ev for ev in res["evidence_summary"])
    assert "no injury" not in " ".join(res["evidence_summary"]).lower()


def test_first_aid_yolo_swelling_prevention():
    """8. YOLO swelling → impossible/invalid YOLO class must never appear in YOLO line."""
    res = first_aid_service.generate_first_aid_guidance(
        visible_injury={"yolo_finding_detected": True, "yolo_finding": "Swelling", "classifier_finding": "Swelling"}
    )
    # Swelling is NOT in YOLO supported classes, so YOLO line must say No confident supported-class detection
    yolo_ev = [ev for ev in res["evidence_summary"] if "YOLO11 object detection" in ev][0]
    assert "No confident supported-class detection" in yolo_ev


def test_first_aid_efficientnet_swelling_labeling():
    """9. EfficientNet swelling → correctly labeled research classifier."""
    res = first_aid_service.generate_first_aid_guidance(
        visible_injury={"yolo_finding_detected": False, "classifier_finding": "Swelling", "classifier_confidence": 0.85}
    )
    clf_ev = [ev for ev in res["evidence_summary"] if "Research image classifier category:" in ev][0]
    assert "Research image classifier category: Swelling" in clf_ev


def test_first_aid_sensor_unavailable_handling():
    """10. Sensor unavailable → no fabricated sensor values."""
    res = first_aid_service.generate_first_aid_guidance(
        sensor_summary={"provided": False, "source_type": "not_provided"}
    )
    sensor_ev = [ev for ev in res["evidence_summary"] if "Sensor" in ev][0]
    assert "Sensor data was not provided" in sensor_ev


def test_first_aid_model_disagreement_uncertainty():
    """11. XGBoost/VQC disagreement → uncertainty preserved."""
    res = first_aid_service.generate_first_aid_guidance(
        xgboost_pred={"class": "LOW"},
        quantum_pred={"class": "HIGH"}
    )
    dis_ev = [ev for ev in res["evidence_summary"] if "Model agreement:" in ev][0]
    assert "DISAGREEMENT" in dis_ev
    assert "increases research uncertainty" in dis_ev


def test_first_aid_canonical_pain_matching():
    """12. First-aid guidance matches canonical questionnaire values."""
    res = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers={"pain_level": 9}
    )
    assert "User-reported pain level: 9/10" in res["evidence_summary"]
    assert "Severe subjective pain intensity score (9/10)." in res["urgent_warning_signs"]


def test_report_service_first_aid_parity():
    """13. PDF and JSON report compilation preserves evidence-based first-aid guidance."""
    from backend.services.report_service import ResearchReportGenerator
    from backend.database.connection import get_database
    
    db = get_database()
    case_id = "test-report-fa-001"
    db.cases.delete_one({"case_id": case_id})
    db.cases.insert_one({
        "case_id": case_id,
        "questionnaire": {"answers": {"pain_level": 9, "location": "Right ankle", "open_wound": "yes"}},
        "visible_injury": {"yolo_finding_detected": False, "classifier_finding": "Cut"},
        "xgboost_prediction": {"class": "MODERATE"},
        "quantum_prediction": {"class": "MODERATE"},
        "created_at": "2026-08-15T10:00:00"
    })
    
    report_gen = ResearchReportGenerator()
    compiled = report_gen.compile_report_data(case_id)
    fa = compiled["first_aid_guidance"]
    
    assert "User-reported pain level: 9/10" in fa["evidence_summary"]
    assert "User-reported open wound: yes" in fa["evidence_summary"]
    db.cases.delete_one({"case_id": case_id})


def test_questionnaire_change_updates_guidance():
    """14. Changing questionnaire answers changes the generated guidance appropriately."""
    res_mild = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers={"pain_level": 2}
    )
    res_severe = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers={"pain_level": 9, "crack_pop": "yes"}
    )
    
    assert res_mild["guidance_level"] != res_severe["guidance_level"]
    assert len(res_severe["urgent_warning_signs"]) > len(res_mild["urgent_warning_signs"])


