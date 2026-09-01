import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.connection import get_database

@pytest.fixture
def test_client():
    return TestClient(app)

def test_missing_sensor_columns_error(test_client):
    """Verify that uploading an invalid CSV/JSON returns the exact missing columns list."""
    # Create a new case
    resp = test_client.post("/api/cases", json={})
    assert resp.status_code == 201
    case_id = resp.json()["case_id"]

    # Write a CSV missing gyroscope_z and speed
    invalid_csv_content = (
        "timestamp,accelerometer_x,accelerometer_y,accelerometer_z,gyroscope_x,gyroscope_y,latitude,longitude\n"
        "0.0,0.1,9.8,0.1,0.0,0.0,37.7,-122.4\n"
        "0.02,0.1,9.8,0.1,0.0,0.0,37.7,-122.4\n"
        "0.04,0.1,9.8,0.1,0.0,0.0,37.7,-122.4\n"
        "0.06,0.1,9.8,0.1,0.0,0.0,37.7,-122.4\n"
        "0.08,0.1,9.8,0.1,0.0,0.0,37.7,-122.4\n"
        "0.1,0.1,9.8,0.1,0.0,0.0,37.7,-122.4\n"
        "0.12,0.1,9.8,0.1,0.0,0.0,37.7,-122.4\n"
        "0.14,0.1,9.8,0.1,0.0,0.0,37.7,-122.4\n"
        "0.16,0.1,9.8,0.1,0.0,0.0,37.7,-122.4\n"
        "0.18,0.1,9.8,0.1,0.0,0.0,37.7,-122.4\n"
    )
    
    csv_filename = f"invalid_temp_{case_id}.csv"
    with open(csv_filename, "w") as f:
        f.write(invalid_csv_content)

    try:
        with open(csv_filename, "rb") as f:
            resp_upload = test_client.post(
                f"/api/cases/{case_id}/sensor",
                files={"file": ("sensor.csv", f, "text/csv")}
            )
        assert resp_upload.status_code == 422
        detail = resp_upload.json()["detail"]
        assert "Missing required sensor columns" in detail
        assert "gyroscope_z" in detail
    finally:
        if os.path.exists(csv_filename):
            os.remove(csv_filename)

def test_sensor_demo_load(test_client):
    """Verify loading the bundled football fall demo sensor log."""
    resp = test_client.post("/api/cases", json={})
    assert resp.status_code == 201
    case_id = resp.json()["case_id"]

    resp_demo = test_client.post(f"/api/cases/{case_id}/sensor/demo")
    assert resp_demo.status_code == 200
    res_data = resp_demo.json()
    assert "Demo sensor log loaded" in res_data["message"]
    assert res_data["summary"]["source_type"] == "demo"
    assert res_data["summary"]["peak_g_force"] > 1.0

def test_sensor_simulation_scenarios(test_client):
    """Verify generating each simulation scenario."""
    scenarios = ["football_fall", "sudden_fall", "sudden_impact", "normal_movement"]
    
    for scenario in scenarios:
        resp = test_client.post("/api/cases", json={})
        assert resp.status_code == 201
        case_id = resp.json()["case_id"]

        resp_sim = test_client.post(
            f"/api/cases/{case_id}/sensor/simulate",
            json={"scenario": scenario}
        )
        assert resp_sim.status_code == 200
        res_data = resp_sim.json()
        assert "Sensor simulation generated" in res_data["message"]
        assert res_data["summary"]["source_type"] == "simulated"
        
        if scenario == "normal_movement":
            assert res_data["summary"]["peak_g_force"] < 2.0
        else:
            assert res_data["summary"]["peak_g_force"] > 2.0

def test_one_click_complete_demo(test_client):
    """Verify that the one-click complete demo triggers the full analysis pipeline."""
    resp = test_client.post("/api/cases/demo")
    assert resp.status_code == 201
    res_data = resp.json()
    
    assert res_data["status"] == "analyzed"
    assert res_data["sensor_summary"]["source_type"] == "demo"
    assert "xgboost_prediction" in res_data
    assert "quantum_prediction" in res_data
    assert ("prediction_agreement" in res_data or "agreement_score" in res_data)

def test_model_registry_hash_consistency_with_disk():
    """Verify that model_registry.json SHA-256 hashes and file sizes match actual disk artifacts 100%."""
    import json, hashlib
    registry_path = os.path.join("ml", "models", "model_registry.json")
    assert os.path.exists(registry_path), "model_registry.json missing"

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    for key, info in registry.items():
        rel_path = info["artifact_path"]
        norm_path = os.path.normpath(rel_path)
        assert os.path.exists(norm_path), f"Artifact missing on disk: {norm_path}"
        
        with open(norm_path, "rb") as fp:
            data = fp.read()
            calculated_hash = hashlib.sha256(data).hexdigest()
            calculated_mb = round(len(data) / (1024 * 1024), 2)
            
        assert calculated_hash == info["artifact_sha256"], f"Hash mismatch for {key}: disk {calculated_hash} != reg {info['artifact_sha256']}"
        assert calculated_mb == info["artifact_size_mb"], f"Size mismatch for {key}: disk {calculated_mb} != reg {info['artifact_size_mb']}"

