import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.report_service import ResearchReportGenerator
from backend.database.connection import get_database

@pytest.fixture
def api_client():
    return TestClient(app)

@pytest.fixture
def seed_case_data():
    db = get_database()
    case_id = "test_report_case_999"
    db.cases.delete_many({"case_id": case_id})
    
    case_doc = {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc),
        "status": "analyzed",
        "visible_injury": {
            "finding": "Bruise",
            "confidence": 0.92,
            "affected_ratio": 0.08,
            "bounding_box": [10, 20, 100, 100]
        },
        "questionnaire": {
            "answers": {
                "pain_level": 8,
                "injury_mechanism": "sharp_object",
                "visible_bleeding": "yes",
                "movement_limitation": "mild",
                "weight_bearing": "yes",
                "crack_pop": "no"
            }
        },
        "sensor_summary": {
            "peak_g_force": 1.4,
            "pre_impact_delta_v": 0.2,
            "post_impact_stabilization_seconds": 0.3
        },
        "xgboost_prediction": {
            "class": "HIGH",
            "probability": 0.81
        },
        "quantum_prediction": {
            "class": "HIGH",
            "score": [0.1, 0.1, 0.8]
        },
        "consistency_analysis": {
            "score": 80.0,
            "status": "Highly Consistent",
            "conflicts": [],
            "agreements": []
        }
    }
    
    db.cases.insert_one(case_doc)
    yield case_id
    db.cases.delete_many({"case_id": case_id})

def test_report_service_logic(seed_case_data):
    """Test report compilation and PDF stream generation."""
    generator = ResearchReportGenerator()
    
    # 1. Compile report
    report = generator.compile_report_data(seed_case_data)
    assert report["case_id"] == seed_case_data
    assert report["rule_derived_category"] == "HIGH"
    assert report["vision"]["finding"] == "Bruise"
    assert report["quantum"]["class"] == "HIGH"
    assert len(report["quantum"]["experimental_vqc_outputs"]) == 3
    assert len(report["safety"]["first_aid_steps"]) > 0
    assert len(report["safety"]["disclaimers"]) == 2 # Prototype disclaimer + Fracture warning due to High Severity
    
    # 2. PDF Stream checks
    pdf_bytes = generator.generate_pdf_bytes(seed_case_data)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert pdf_bytes.endswith(b"%%EOF")

def test_report_api_endpoints(api_client, seed_case_data):
    """Test FastAPI JSON and PDF export routes."""
    # 1. Base JSON report endpoint
    resp = api_client.get(f"/api/cases/{seed_case_data}/report")
    assert resp.status_code == 200
    report_json = resp.json()
    assert report_json["case_id"] == seed_case_data
    assert report_json["rule_derived_category"] == "HIGH"

    # 2. Downloader JSON report endpoint
    resp_json = api_client.get(f"/api/cases/{seed_case_data}/report/json")
    assert resp_json.status_code == 200
    assert "Content-Disposition" in resp_json.headers
    assert "attachment" in resp_json.headers["Content-Disposition"]
    assert resp_json.json()["case_id"] == seed_case_data

    # 3. Downloader PDF report endpoint
    resp_pdf = api_client.get(f"/api/cases/{seed_case_data}/report/pdf")
    assert resp_pdf.status_code == 200
    assert "Content-Disposition" in resp_pdf.headers
    assert "attachment" in resp_pdf.headers["Content-Disposition"]
    assert resp_pdf.headers["content-type"] == "application/pdf"
    assert resp_pdf.content.startswith(b"%PDF-1.4")
