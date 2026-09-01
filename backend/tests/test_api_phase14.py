import pytest
from backend.services.safety_service import SafetyGuidanceService
from backend.database.connection import get_database

def test_safety_guidance_low_severity():
    """Verify that a LOW severity category retrieves static data and contains the base disclaimer."""
    service = SafetyGuidanceService()
    
    # Run retrieval
    res = service.get_safety_guidance(finding="bruise", severity_category="LOW")
    
    assert res["finding"] == "bruise"
    assert len(res["first_aid_steps"]) > 0
    assert len(res["red_flags"]) > 0
    
    # Disclaimers validation
    assert len(res["disclaimers"]) == 1
    assert "research prototype, not a medical diagnostic device" in res["disclaimers"][0]
    # Fracture warning must NOT be in a LOW risk case
    assert not any("cannot reliably determine a fracture" in d for d in res["disclaimers"])

def test_safety_guidance_high_severity_or_fracture_risk():
    """Verify that HIGH/MODERATE cases or fracture flags append the mandatory fracture warning."""
    service = SafetyGuidanceService()
    
    # 1. Test case: HIGH category
    res_high = service.get_safety_guidance(finding="swelling", severity_category="HIGH")
    assert len(res_high["disclaimers"]) == 2
    assert any("cannot reliably determine a fracture" in d for d in res_high["disclaimers"])
    assert any("research prototype, not a medical diagnostic device" in d for d in res_high["disclaimers"])

    # 2. Test case: LOW category but has_fracture_risk flagged (e.g. user heard a crack)
    res_risk = service.get_safety_guidance(finding="swelling", has_fracture_risk=True, severity_category="LOW")
    assert len(res_risk["disclaimers"]) == 2
    assert any("cannot reliably determine a fracture" in d for d in res_risk["disclaimers"])
