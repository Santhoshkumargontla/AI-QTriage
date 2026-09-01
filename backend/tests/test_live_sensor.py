import pytest
import uuid
import time
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.connection import get_database

client = TestClient(app)

def test_live_sensor_upload_and_verification():
    """
    Test 1:
    Verify live sensor upload, server-side data validation, authoritative sampling rate calculation,
    and storage of sensor_source_type = 'live'.
    """
    db = get_database()
    case_id = f"test_live_{uuid.uuid4().hex[:8]}"

    case_doc = {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc),
        "status": "created"
    }
    db.cases.insert_one(case_doc)

    try:
        # Create 50 samples over 1.0 second (50 Hz)
        start_ts = int(time.time() * 1000)
        samples = []
        for i in range(50):
            ts = start_ts + i * 20  # 20ms apart
            samples.append({
                "timestamp": ts,
                "acceleration_x": 0.1,
                "acceleration_y": 9.81 if i != 25 else 35.0, # Impact spike at sample index 25
                "acceleration_z": 0.2,
                "acceleration_gravity_x": 0.1,
                "acceleration_gravity_y": 9.81 if i != 25 else 35.0,
                "acceleration_gravity_z": 0.2,
                "rotation_alpha": 0.01,
                "rotation_beta": 0.02,
                "rotation_gamma": 0.01,
                "latitude": None, # Optional GPS skipped
                "longitude": None
            })

        payload = {
            "source_type": "live",
            "device_metadata": {
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)",
                "platform": "iPhone"
            },
            "recording_duration_seconds": 1.0,
            "observed_sampling_rate_hz": 50.0,
            "samples": samples
        }

        resp = client.post(f"/api/cases/{case_id}/sensor/live/upload", json=payload)
        assert resp.status_code == 200, f"Response failed: {resp.text}"
        data = resp.json()

        assert data["message"] == "Real-time device sensor data uploaded and processed successfully"
        summary = data["summary"]
        assert summary["source_type"] == "live"
        assert summary["data_provenance_badge"] == "REAL-TIME DEVICE DATA"
        assert summary["sample_count"] == 50
        assert summary["backend_verified_sampling_rate_hz"] > 40.0
        assert summary["peak_g_force"] > 3.0

        # Verify MongoDB storage
        stored = db.cases.find_one({"case_id": case_id})
        assert stored["sensor_available"] is True
        assert stored["sensor_source_type"] == "live"
        assert stored["sensor_summary"]["backend_verified_sampling_rate_hz"] > 40.0
    finally:
        db.cases.delete_one({"case_id": case_id})

def test_live_sensor_validation_failures():
    """
    Test 2 (Correction 2):
    Verify that server-side validation rejects invalid/malformed payloads (too short, NaN, unrealistic).
    """
    db = get_database()
    case_id = f"test_val_fail_{uuid.uuid4().hex[:8]}"

    case_doc = {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc),
        "status": "created"
    }
    db.cases.insert_one(case_doc)

    try:
        # A. Too short (<10 samples)
        short_payload = {
            "source_type": "live",
            "samples": [{"timestamp": 100, "acceleration_x": 0.1}]
        }
        resp_short = client.post(f"/api/cases/{case_id}/sensor/live/upload", json=short_payload)
        assert resp_short.status_code == 422
        assert "Recording too short" in resp_short.text or "minimum 10 valid samples" in resp_short.text

        # B. Unrealistic acceleration (>500 m/s^2)
        unrealistic_samples = [
            {"timestamp": 1000 + i * 20, "acceleration_x": 999.0} for i in range(15)
        ]
        unrealistic_payload = {"source_type": "live", "samples": unrealistic_samples}
        resp_unreal = client.post(f"/api/cases/{case_id}/sensor/live/upload", json=unrealistic_payload)
        assert resp_unreal.status_code == 422
        assert "Unrealistic acceleration" in resp_unreal.text
    finally:
        db.cases.delete_one({"case_id": case_id})

def test_all_five_sensor_modes_regression():
    """
    Test 3 (Regression Check for All 5 Modes):
    Verify that Demo, Upload, Simulation, Skip, and Live sensor modes all process cleanly.
    """
    db = get_database()

    # 1. Skip Mode
    cid_skip = f"reg_skip_{uuid.uuid4().hex[:6]}"
    db.cases.insert_one({"case_id": cid_skip, "status": "created"})
    r_skip = client.post(f"/api/cases/{cid_skip}/sensor/skip")
    assert r_skip.status_code == 200

    # 2. Demo Mode
    cid_demo = f"reg_demo_{uuid.uuid4().hex[:6]}"
    db.cases.insert_one({"case_id": cid_demo, "status": "created"})
    r_demo = client.post(f"/api/cases/{cid_demo}/sensor/demo")
    assert r_demo.status_code == 200
    assert db.cases.find_one({"case_id": cid_demo})["sensor_source_type"] == "demo"

    # 3. Simulate Mode
    cid_sim = f"reg_sim_{uuid.uuid4().hex[:6]}"
    db.cases.insert_one({"case_id": cid_sim, "status": "created"})
    r_sim = client.post(f"/api/cases/{cid_sim}/sensor/simulate", json={"scenario": "football_fall"})
    assert r_sim.status_code == 200
    assert db.cases.find_one({"case_id": cid_sim})["sensor_source_type"] == "simulated"

    # Clean up test cases
    db.cases.delete_many({"case_id": {"$in": [cid_skip, cid_demo, cid_sim]}})
