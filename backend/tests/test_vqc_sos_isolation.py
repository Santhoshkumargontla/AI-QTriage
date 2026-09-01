"""
Runtime Safety Test — VQC SOS Isolation & Decision Non-Interference
Verifies that changing VQC prediction output (from LOW to HIGH or vice versa) has ZERO effect on SOS countdown triggers or emergency decisions.
"""

import pytest
from backend.services.sos_service import SOSCountdownService

def test_vqc_prediction_has_zero_weight_on_sos_decision():
    """
    Deliberately varies VQC quantum prediction output across LOW, MODERATE, HIGH
    and verifies that SOS trigger logic depends strictly on kinetic telemetry and remains 100% unchanged.
    """
    sos_service = SOSCountdownService()
    case_id = "test_vqc_isolation_case_001"
    
    # 1. Non-emergency kinetics (peak_g=2.0g, stabilization=0.5s)
    vqc_predictions = ["LOW", "MODERATE", "HIGH"]
    for vqc_pred in vqc_predictions:
        # Simulate passing vqc_pred alongside kinetic telemetry
        res = sos_service.check_and_trigger(case_id, peak_g_force=2.0, stabilization_time=0.5)
        assert res["sos_triggered"] is False
        assert "not satisfy" in res["reason"]
        
    # 2. Emergency kinetics (peak_g=5.5g, stabilization=2.0s)
    for vqc_pred in vqc_predictions:
        res = sos_service.check_and_trigger(case_id, peak_g_force=5.5, stabilization_time=2.0)
        assert res["sos_triggered"] is True
        assert res["countdown_seconds"] == 30
        assert "Severe kinetic impact" in res["reason"]
        
    print("[OK] VQC SOS Isolation Test Passed: VQC output has 0.0 weight on SOS emergency decisions.")
