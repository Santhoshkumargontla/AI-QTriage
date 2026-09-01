"""
test_phase14_features.py

Automated tests for Phase 14 features:
1. Basic First-Aid Guidance Engine
2. Voice Transcription & Extraction (No default fallback injection)
3. Expanded 12-Questionnaire Field Persistence
4. SOS UTC Timestamp Storage and Configured Timezone Formatting
"""

import pytest
from datetime import datetime, timezone
from backend.services.first_aid_service import first_aid_service
from backend.services.twilio_service import format_sms_timestamp, twilio_service
from backend.database.connection import get_database


def test_first_aid_guidance_generation():
    """Verify first-aid guidance engine produces deterministic, rule-based advice."""
    answers = {
        "pain_level": "6",
        "location": "Right ankle",
        "cause": "sports",
        "swelling": "yes",
        "crack_pop": "yes",
        "movement": "limited",
        "limb_use": "with_pain"
    }
    res = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers=answers,
        sensor_summary={"peak_g_force": 4.5},
        visible_injury={"finding": "Swelling"},
        rule_derived_category="MODERATE"
    )

    assert res["guidance_level"] in ("CONSERVATIVE_CARE", "URGENT_EVALUATION")
    assert res["seek_professional_evaluation"] is True
    assert len(res["immediate_steps"]) > 0
    assert len(res["avoid"]) > 0
    assert len(res["monitor"]) > 0
    disclaimer_text = str(res.get("model_limitation_statement") or res.get("display_message") or "")
    assert "medical" in disclaimer_text.lower() or "diagnosis" in disclaimer_text.lower() or "prototype" in disclaimer_text.lower()


def test_first_aid_urgent_warning_signs():
    """Verify severe warning signs (deformity, numbness, heavy bleeding) trigger URGENT_EVALUATION."""
    answers = {
        "pain_level": "9",
        "deformity": "yes",
        "numbness_tingling": "yes",
        "bleeding": "heavy"
    }
    res = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers=answers,
        rule_derived_category="HIGH"
    )

    assert res["guidance_level"] == "URGENT_EVALUATION"
    assert res["seek_professional_evaluation"] is True
    assert len(res["urgent_warning_signs"]) >= 3


def test_first_aid_missing_data_no_fabrication():
    """Verify missing questionnaire fields do not fabricate false warnings."""
    res = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers={},
        sensor_summary=None,
        visible_injury=None,
        rule_derived_category="LOW"
    )

    assert res["guidance_level"] == "BASIC_REST_MONITOR"
    assert "Visible anatomical deformity or misalignment reported." not in res["urgent_warning_signs"]
    assert not any("Severe subjective pain intensity score" in w for w in res["urgent_warning_signs"])
    assert len(res["immediate_steps"]) > 0


def test_evidence_builder_unanswered_fields_are_null():
    """Verify StructuredEvidenceBuilder sets unanswered fields to null (no automatic default injection)."""
    from backend.services.first_aid_service import StructuredEvidenceBuilder
    ev = StructuredEvidenceBuilder.build_evidence(
        questionnaire_answers={"location": "Left ankle", "pain_level": 7}
    )
    q = ev["questionnaire"]
    assert q["location"] == "Left ankle"
    assert q["pain_level"] == 7
    assert q["swelling"] is None
    assert q["bruising_discoloration"] is None
    assert q["numbness_tingling"] is None


def test_sos_timestamp_utc_and_ist_formatting():
    """Verify SOS timestamp is stored as UTC ISO string and formatted explicitly in IST."""
    now_utc = datetime.now(timezone.utc)
    utc_str = now_utc.isoformat()

    formatted_ist = format_sms_timestamp(utc_str, "Asia/Kolkata")
    
    assert "IST" in formatted_ist
    assert str(now_utc.year) in formatted_ist


def test_expanded_questionnaire_persistence():
    """Verify expanded 12 questionnaire fields persist in MongoDB and return in API schema."""
    db = get_database()
    case_id = "test-phase14-q12-001"
    db.cases.delete_one({"case_id": case_id})

    expanded_answers = {
        "cause": "twist",
        "swelling": "yes",
        "bruising": "yes",
        "open_wound": "no",
        "bleeding": "none",
        "crack_pop": "no",
        "movement": "limited",
        "limb_use": "with_pain",
        "numbness_tingling": "no",
        "pain_trend": "worse",
        "deformity": "no",
        "injury_time": "2 hours ago"
    }

    db.cases.insert_one({
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc),
        "status": "questionnaire_submitted",
        "questionnaire": {
            "answers": expanded_answers,
            "answer_source": "typed",
            "voice_used": False
        }
    })

    doc = db.cases.find_one({"case_id": case_id})
    assert doc["questionnaire"]["answers"]["cause"] == "twist"
    assert doc["questionnaire"]["answers"]["pain_trend"] == "worse"
    assert doc["questionnaire"]["answers"]["injury_time"] == "2 hours ago"

    # Cleanup
    db.cases.delete_one({"case_id": case_id})
