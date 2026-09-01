import os
import json
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.connection import get_database

client = TestClient(app)

def test_static_json_templates():
    """Verify that predefined static questionnaire templates exist and are valid JSON."""
    template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "questionnaire_templates")
    
    for template_name in ["cut.json", "bruise.json", "swelling.json"]:
        path = os.path.join(template_dir, template_name)
        assert os.path.exists(path), f"Predefined template {template_name} is missing."
        
        with open(path, "r") as f:
            data = json.load(f)
            assert "template_id" in data
            assert "questions" in data
            assert len(data["questions"]) > 0


def test_voice_endpoint_intentionally_removed():
    """Verify that posting audio to the legacy voice endpoint returns 404 Not Found (Voice feature removed)."""
    response = client.post("/api/cases/test_case_id/voice")
    assert response.status_code == 404, "Voice endpoint must return 404 as Voice feature has been completely removed."
