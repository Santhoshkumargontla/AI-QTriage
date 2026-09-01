import os
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app
from ml.sensor.sensor_processor import validate_sensor_csv, process_sensor_data, SensorValidationError
from backend.database.connection import get_database

client = TestClient(app)

@pytest.fixture
def temp_csv_dir(tmpdir):
    return str(tmpdir)

def test_sensor_validation_failures(temp_csv_dir):
    """Verify that schema discrepancies, short files, and low sampling rates are rejected."""
    # 1. Test missing columns
    path_missing = os.path.join(temp_csv_dir, "missing.csv")
    df_missing = pd.DataFrame({"timestamp": [1.0, 2.0], "accel_x": [0.0, 0.0]})
    df_missing.to_csv(path_missing, index=False)
    with pytest.raises(SensorValidationError) as exc:
        validate_sensor_csv(path_missing)
    assert "Missing required sensor columns" in str(exc.value)

    # 2. Test too few samples
    path_short = os.path.join(temp_csv_dir, "short.csv")
    df_short = pd.DataFrame({
        "timestamp": np.linspace(0, 0.1, 5),
        "accel_x": [0]*5, "accel_y": [9.8]*5, "accel_z": [0]*5,
        "gyro_x": [0]*5, "gyro_y": [0]*5, "gyro_z": [0]*5,
        "optical_lux": [100]*5
    })
    df_short.to_csv(path_short, index=False)
    with pytest.raises(SensorValidationError) as exc:
        validate_sensor_csv(path_short)
    assert "contains too few samples" in str(exc.value)

    # 3. Test low sampling rate (e.g. 10Hz)
    path_slow = os.path.join(temp_csv_dir, "slow.csv")
    df_slow = pd.DataFrame({
        "timestamp": np.arange(0, 2.0, 0.1), # 10Hz sampling
        "accel_x": [0]*20, "accel_y": [9.8]*20, "accel_z": [0]*20,
        "gyro_x": [0]*20, "gyro_y": [0]*20, "gyro_z": [0]*20,
        "optical_lux": [100]*20
    })
    df_slow.to_csv(path_slow, index=False)
    with pytest.raises(SensorValidationError) as exc:
        validate_sensor_csv(path_slow)
    assert "Sampling rate is insufficient" in str(exc.value)

def test_sensor_timeline_reconstruction(temp_csv_dir):
    """Test timeline event sequencing and physical kinetics analysis on compliant sensor streams."""
    path_compliant = os.path.join(temp_csv_dir, "compliant.csv")
    
    # Generate 2 seconds of 50Hz data (100 samples, dt = 0.02s)
    n_samples = 100
    timestamps = np.arange(0, n_samples) * 0.02 # seconds
    
    accel_x = np.zeros(n_samples)
    accel_y = np.ones(n_samples) * 9.8 # gravity
    accel_z = np.zeros(n_samples)
    
    gyro_x = np.zeros(n_samples)
    gyro_y = np.zeros(n_samples)
    gyro_z = np.zeros(n_samples)
    
    lux = np.ones(n_samples) * 200.0
    
    # Inject a peak impact of 45 m/s2 (approx 4.6g) at sample 50 (t = 1.0s)
    accel_x[50] = 45.0
    
    # Inject light drop below 10 lux at sample 50
    lux[50] = 5.0
    
    # Post-impact stabilization: decays back to baseline quickly (e.g. < 1.5g within 0.2s / 10 samples)
    # At sample 60 (t = 1.20s), it is stable
    for idx in range(51, 60):
        decay_factor = (60 - idx) / 10.0
        accel_x[idx] = 45.0 * decay_factor
        
    df = pd.DataFrame({
        "timestamp": timestamps,
        "accel_x": accel_x, "accel_y": accel_y, "accel_z": accel_z,
        "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z,
        "optical_lux": lux
    })
    df.to_csv(path_compliant, index=False)
    
    # Process
    summary = process_sensor_data(path_compliant)
    
    assert summary["peak_time_offset"] == 1.0
    assert summary["peak_acceleration"] == pytest.approx(np.sqrt(45.0**2 + 9.8**2), abs=1e-2)
    assert summary["optical_lux_drop"] is True
    
    # Stabilization happens at sample 58 (t = 1.16s), so post-impact stabilization duration is 1.16 - 1.0 = 0.16s
    assert summary["post_impact_stabilization_seconds"] == pytest.approx(0.16, abs=1e-2)
    
    # Verify sequence order of timeline events
    events = summary["events"]
    assert len(events) >= 4
    
    event_names = [e["event_name"] for e in events]
    assert "Sensor Log Started" in event_names
    assert "Ambient Light Drop" in event_names
    assert "Peak Impact Acceleration" in event_names
    assert "Physical Stabilization" in event_names

def test_sensor_upload_api_integration(temp_csv_dir):
    """Verify that compliant CSV uploads return HTTP 200 and save timeline logs to MongoDB."""
    # 1. Create a case record
    response = client.post("/api/cases", json={"notes": "Sensor integration case"})
    assert response.status_code == 201
    case_id = response.json()["case_id"]
    
    # 2. Build compliant CSV file content
    csv_path = os.path.join(temp_csv_dir, "api_test.csv")
    n_samples = 50
    timestamps = np.arange(0, n_samples) * 0.02
    df = pd.DataFrame({
        "timestamp": timestamps,
        "accel_x": [0]*n_samples,
        "accel_y": [9.8]*n_samples,
        "accel_z": [0]*n_samples,
        "gyro_x": [0]*n_samples,
        "gyro_y": [0]*n_samples,
        "gyro_z": [0]*n_samples,
        "optical_lux": [150.0]*n_samples
    })
    # Add minor peak at sample 25
    df.loc[25, "accel_x"] = 40.0
    df.to_csv(csv_path, index=False)
    
    # 3. Post to endpoint
    with open(csv_path, "rb") as f:
        files = {"file": ("sensor.csv", f, "text/csv")}
        response_sensor = client.post(f"/api/cases/{case_id}/sensor", files=files)
        
    assert response_sensor.status_code == 200
    body = response_sensor.json()
    assert body["message"] == "Sensor data uploaded and processed successfully"
    assert "summary" in body
    assert body["summary"]["peak_time_offset"] == 0.50
    
    # Verify MongoDB updated
    db = get_database()
    case_doc = db.cases.find_one({"case_id": case_id})
    assert case_doc["status"] == "sensor_submitted"
    assert case_doc["sensor_summary"]["peak_time_offset"] == 0.50
    
    # Clean up
    db.cases.delete_one({"case_id": case_id})
