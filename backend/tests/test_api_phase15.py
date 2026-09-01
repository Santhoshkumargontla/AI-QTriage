import os
import pytest
import time
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.sos_service import SOSCountdownService
from backend.database.connection import get_database

@pytest.fixture
def api_client():
    return TestClient(app)

@pytest.fixture
def test_case_id():
    db = get_database()
    case_id = "test_case_sos_123"
    db.cases.delete_many({"case_id": case_id})
    db.cases.insert_one({
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc),
        "status": "created"
    })
    yield case_id
    db.cases.delete_many({"case_id": case_id})

def test_sos_service_logic(test_case_id):
    """Test SOS service trigger thresholds, overrides, and auto-dispatch expirations."""
    service = SOSCountdownService()
    db = get_database()
    
    # 1. Below threshold checks
    res_low = service.check_and_trigger(test_case_id, peak_g_force=2.5, stabilization_time=1.0)
    assert res_low["sos_triggered"] is False
    
    # 2. Trigger check
    res_high = service.check_and_trigger(test_case_id, peak_g_force=4.5, stabilization_time=1.8)
    assert res_high["sos_triggered"] is True
    assert "Prolonged stabilization" in res_high["reason"] or "Severe kinetic impact" in res_high["reason"]
    
    # Verify status in DB
    status_doc = service.get_sos_status(test_case_id)
    assert status_doc["status"] == "triggered"
    assert status_doc["remaining_seconds"] <= 30.0
    
    # 3. Abort check
    abort_res = service.abort_sos(test_case_id)
    assert abort_res["status"] == "aborted"
    
    status_aborted = service.get_sos_status(test_case_id)
    assert status_aborted["status"] == "aborted"
    assert status_aborted["remaining_seconds"] == 0.0

    # 4. Auto-dispatch check (mock past trigger time)
    service.check_and_trigger(test_case_id, peak_g_force=5.0, stabilization_time=2.0)
    # Manually shift trigger time back by 35 seconds
    past_time = datetime.now(timezone.utc) - timedelta(seconds=35)
    db.cases.update_one(
        {"case_id": test_case_id},
        {"$set": {"sos_trigger_time": past_time.isoformat()}}
    )
    
    status_expired = service.get_sos_status(test_case_id)
    # In local_demo mode, natural expiry resolves to LOCAL_SIMULATION
    assert status_expired["status"] == "LOCAL_SIMULATION"
    assert status_expired["remaining_seconds"] == 0.0
    assert status_expired.get("twilio_message_sid") in (None, "")

def test_sos_api_integration(api_client, test_case_id, tmpdir):
    """Test FastAPI endpoint uploads, countdown checks, and cancellations."""
    sensor_path = os.path.join(str(tmpdir), "severe_sensor.csv")
    
    # Write severe kinetics CSV file
    # Columns: timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,optical_lux
    # Trigger requirement is average freq >= 40Hz, peak g >= 4.0, stabilization time >= 1.5
    # Let's write 100 rows sampled at 50Hz (20ms interval)
    rows = ["timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,optical_lux"]
    for i in range(100):
        t = i * 0.02
        if i == 10:
            # Impact point: accel_x = 4.5g -> in m/s^2 is 4.5 * 9.80665 = 44.13
            val = 4.5 * 9.80665
            rows.append(f"{t},{val},0.0,0.0,0.0,0.0,0.0,5.0")
        elif i > 10 and i < 90:
            # Unstable period (accel magnitude ~ 1.8g -> in m/s^2 is 1.8 * 9.80665 = 17.65) to delay stabilization for ~ 1.6s
            val = 1.8 * 9.80665
            rows.append(f"{t},{val},0.0,0.0,0.1,0.1,0.1,15.0")
        else:
            # Baseline gravity (1.0g -> in m/s^2 is 1.0 * 9.80665 = 9.80665)
            val = 1.0 * 9.80665
            rows.append(f"{t},{val},0.0,0.0,0.0,0.0,0.0,15.0")
            
    with open(sensor_path, "w") as f:
        f.write("\n".join(rows))

    # 1. Upload sensor log via API
    with open(sensor_path, "rb") as f:
        resp = api_client.post(
            f"/api/cases/{test_case_id}/sensor",
            files={"file": ("severe_sensor.csv", f, "text/csv")}
        )
        
    assert resp.status_code == 200
    res_json = resp.json()
    assert res_json["sos_triggered"] is True
    
    # 2. Get status via API
    status_resp = api_client.get(f"/api/cases/{test_case_id}/sos/status")
    assert status_resp.status_code == 200
    status_json = status_resp.json()
    assert status_json["status"] == "triggered"
    assert status_json["remaining_seconds"] > 0.0
    
    # 3. Abort SOS via API
    abort_resp = api_client.post(f"/api/cases/{test_case_id}/sos/abort")
    assert abort_resp.status_code == 200
    abort_json = abort_resp.json()
    assert abort_json["status"] == "aborted"
    
    # 4. Verify aborted status
    status_resp2 = api_client.get(f"/api/cases/{test_case_id}/sos/status")
    assert status_resp2.json()["status"] == "aborted"
