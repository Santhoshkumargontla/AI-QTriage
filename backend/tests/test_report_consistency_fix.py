import pytest
import os
import uuid
import json
from datetime import datetime, timezone
from backend.database.connection import get_database
from backend.services.report_service import ResearchReportGenerator
from ml.explainability.evidence_consistency import EvidenceConsistencyAnalyzer

def test_regression_model_agreement_and_disagreement():
    """
    Test A & Test B:
    - XGBoost=HIGH vs VQC=HIGH -> AGREEMENT
    - XGBoost=HIGH vs VQC=LOW -> DISAGREEMENT & elevated uncertainty
    """
    analyzer = EvidenceConsistencyAnalyzer()
    dummy_vec = [0.5] * 23
    dummy_names = [f"feat_{i}" for i in range(23)]

    # Test A: Agreement
    res_agree = analyzer.calculate_consistency(
        dummy_vec, dummy_names,
        xgb_prediction_class="HIGH",
        vqc_prediction_class="HIGH",
        segmentation_reliable=True,
        is_demo=False
    )
    assert res_agree["status"] == "Highly Consistent"
    assert any("Model Agreement" in a for a in res_agree["agreements"])

    # Test B: Disagreement
    res_disagree = analyzer.calculate_consistency(
        dummy_vec, dummy_names,
        xgb_prediction_class="HIGH",
        vqc_prediction_class="LOW",
        segmentation_reliable=True,
        is_demo=False
    )
    assert res_disagree["status"] != "Highly Consistent"
    assert res_disagree["status"] in ["Partially Consistent", "Conflicting Evidence Detected"]
    assert any("Model Disagreement" in c for c in res_disagree["conflicts"])

def test_regression_segmentation_unavailable_behavior():
    """
    Test C:
    When segmentation is unavailable or unreliable, affected_area_ratio must be None
    and affected_ratio display string must be "N/A".
    """
    db = get_database()
    case_id = f"test_seg_fix_{uuid.uuid4().hex[:8]}"
    
    case_doc = {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc),
        "status": "analyzed",
        "is_demo": False,
        "visible_injury": {
            "finding": "Swelling",
            "finding_detected": False,
            "confidence": None,
            "bounding_box": None,
            "affected_ratio": None,
            "affected_area_ratio": None,
            "segmentation_available": False,
            "segmentation_reliable": False,
            "segmentation_status": "insufficient",
            "segmentation_reason": "No reliable segmentation mask available (YOLO detected no target region)"
        },
        "xgboost_prediction": {"class": "HIGH", "probability": 0.95},
        "quantum_prediction": {"class": "LOW", "score": [0.2, 0.3, 0.5]},
        "agreement_score": "DISAGREEMENT",
        "uncertainty_status": "HIGH UNCERTAINTY",
        "uncertainty_level": "HIGH UNCERTAINTY",
        "uncertainty_reasons": ["Classical and quantum model predictions disagree."]
    }
    db.cases.insert_one(case_doc)

    try:
        generator = ResearchReportGenerator()
        report = generator.compile_report_data(case_id)

        assert report["prediction_agreement"] == "DISAGREEMENT"
        assert report["uncertainty_level"] == "HIGH UNCERTAINTY"
        assert report["vision"]["affected_ratio"] == "N/A"
        assert report["vision"]["affected_area_ratio"] is None
        assert report["vision"]["segmentation_reliable"] is False
        assert "No reliable segmentation mask available" in report["vision"]["segmentation_reason"]
    finally:
        db.cases.delete_one({"case_id": case_id})

def test_regression_synthetic_data_provenance_warning():
    """
    Test D:
    Synthetic or demo cases must clearly display data provenance as SYNTHETIC
    and include the explicit research/demo warning.
    """
    db = get_database()
    case_id = f"test_demo_fix_{uuid.uuid4().hex[:8]}"
    
    case_doc = {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc),
        "status": "analyzed",
        "is_demo": True,
        "sensor_source_type": "demo",
        "sensor_summary": {"source_type": "demo", "peak_g_force": 4.5},
        "visible_injury": {"finding": "Cut", "source_type": "demo"},
        "xgboost_prediction": {"class": "HIGH", "probability": 0.92},
        "quantum_prediction": {"class": "HIGH", "score": [0.1, 0.2, 0.7]},
        "agreement_score": "AGREEMENT",
        "uncertainty_status": "LOW UNCERTAINTY"
    }
    db.cases.insert_one(case_doc)

    try:
        generator = ResearchReportGenerator()
        report = generator.compile_report_data(case_id)

        assert report["data_provenance"] == "SYNTHETIC"
        assert report["research_demo_warning"] is not None
        assert "Research/Demo Data Warning" in report["research_demo_warning"]
    finally:
        db.cases.delete_one({"case_id": case_id})

def test_regression_full_report_parity():
    """
    Test E:
    Verify that API, DB, JSON report, and PDF report render identical normalized values for the same case.
    """
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)

    db = get_database()
    case_id = f"test_parity_{uuid.uuid4().hex[:8]}"
    
    case_doc = {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc),
        "status": "analyzed",
        "is_demo": True,
        "sensor_source_type": "demo",
        "sensor_available": True,
        "sensor_summary": {"source_type": "demo", "peak_g_force": 3.8, "post_impact_stabilization_seconds": 1.2},
        "visible_injury": {
            "finding": "Swelling",
            "finding_detected": False,
            "confidence": None,
            "bounding_box": None,
            "affected_ratio": None,
            "affected_area_ratio": None,
            "segmentation_available": False,
            "segmentation_reliable": False,
            "segmentation_status": "insufficient",
            "segmentation_reason": "No reliable segmentation mask available (YOLO detected no target region)"
        },
        "xgboost_prediction": {"class": "HIGH", "probability": 0.94},
        "quantum_prediction": {"class": "LOW", "score": [0.7, 0.2, 0.1]},
        "agreement_score": "DISAGREEMENT",
        "uncertainty_status": "HIGH UNCERTAINTY",
        "uncertainty_level": "HIGH UNCERTAINTY",
        "uncertainty_reasons": [
            "Classical and quantum model predictions disagree.",
            "Segmentation output was unavailable or below the reliability threshold.",
            "This case uses synthetic or demo data."
        ]
    }
    db.cases.insert_one(case_doc)

    try:
        generator = ResearchReportGenerator()

        # 1. Direct Python compilation
        compiled = generator.compile_report_data(case_id)
        
        # 2. JSON API endpoint
        resp_json = client.get(f"/api/cases/{case_id}/report/json")
        assert resp_json.status_code == 200
        json_data = resp_json.json()

        # 3. PDF API endpoint
        resp_pdf = client.get(f"/api/cases/{case_id}/report/pdf")
        assert resp_pdf.status_code == 200
        pdf_bytes = resp_pdf.content

        # Parity assertions
        assert compiled["prediction_agreement"] == json_data["prediction_agreement"] == "DISAGREEMENT"
        assert compiled["uncertainty_level"] == json_data["uncertainty_level"] == "HIGH UNCERTAINTY"
        assert compiled["vision"]["affected_ratio"] == json_data["vision"]["affected_ratio"] == "N/A"
        assert compiled["data_provenance"] == json_data["data_provenance"] == "SYNTHETIC"

        # Check PDF contains the matching strings
        pdf_str = pdf_bytes.decode("latin-1")
        assert "DISAGREEMENT" in pdf_str
        assert "HIGH UNCERTAINTY" in pdf_str
        assert "Affected Ratio: N/A" in pdf_str
        assert "SYNTHETIC" in pdf_str
    finally:
        db.cases.delete_one({"case_id": case_id})
