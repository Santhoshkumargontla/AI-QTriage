import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.connection import get_database

client = TestClient(app)

def test_health_endpoint():
    """Test that the health endpoint returns a successful healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"

def test_case_creation_and_retrieval():
    """Test case initialization and subsequent retrieval by case ID."""
    # Create Case
    response = client.post("/api/cases", json={"notes": "Test case description"})
    assert response.status_code == 201
    case_data = response.json()
    assert "case_id" in case_data
    assert case_data["status"] == "created"
    
    # Retrieve Case
    case_id = case_data["case_id"]
    ret_response = client.get(f"/api/cases/{case_id}")
    assert ret_response.status_code == 200
    ret_data = ret_response.json()
    assert ret_data["case_id"] == case_id
    assert ret_data["status"] == "created"
    
    # Clean up test case in MongoDB
    db = get_database()
    db.cases.delete_one({"case_id": case_id})

def test_invalid_image_type_upload():
    """Verify that uploading an invalid file format returns a 400 Bad Request."""
    # Create a temporary case
    response = client.post("/api/cases", json={"notes": "Temporary test case for upload"})
    assert response.status_code == 201
    case_id = response.json()["case_id"]
    
    # Upload text file as image
    files = {"file": ("test.txt", b"dummy text content", "text/plain")}
    upload_response = client.post(f"/api/cases/{case_id}/image", files=files)
    assert upload_response.status_code == 400
    assert "Invalid image format" in upload_response.json()["detail"]
    
    # Clean up test case in MongoDB
    db = get_database()
    db.cases.delete_one({"case_id": case_id})

def test_get_models_list():
    """Ensure that get_models endpoint lists available system classifiers."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    # Check that keys are present
    assert "model_name" in data[0]
    assert "model_version" in data[0]
