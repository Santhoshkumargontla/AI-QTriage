"""Sensor pipeline: scenarios, FEATURE_MISSING, persistence of classifier output."""
import os

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.connection import get_database
from ml.classifiers.sensor_classifier import FEATURE_MISSING, MODEL_UNAVAILABLE, SensorClassifier
from ml.sensor.scenarios import SUPPORTED_SCENARIOS, resolve_scenario

client = TestClient(app)


def test_supported_scenario_names_are_canonical_only():
    assert SUPPORTED_SCENARIOS == (
        "football_fall",
        "sudden_fall",
        "sudden_impact",
        "normal_movement",
    )
    assert resolve_scenario("football") is None
    assert resolve_scenario("FOOTBALL_FALL") == "football_fall"


def test_four_simulation_scenarios_persist_classifier(tmp_path):
    db = get_database()
    expected_g = {
        "normal_movement": lambda g: g is not None and g < 2.0,
        "sudden_fall": lambda g: g is not None and g > 2.0,
        "football_fall": lambda g: g is not None and g > 2.0,
        "sudden_impact": lambda g: g is not None and g > 2.0,
    }
    results = {}
    for scenario in SUPPORTED_SCENARIOS:
        resp = client.post("/api/cases", json={"notes": f"sensor {scenario}"})
        assert resp.status_code == 201
        case_id = resp.json()["case_id"]
        sim = client.post(f"/api/cases/{case_id}/sensor/simulate", json={"scenario": scenario})
        assert sim.status_code == 200, sim.text
        summary = sim.json()["summary"]
        assert summary["scenario"] == scenario
        assert summary["source_type"] == "simulated"
        assert expected_g[scenario](summary["peak_g_force"])
        assert "predicted_motion_class" in summary
        assert summary["classifier_status"] in ("classified", FEATURE_MISSING, MODEL_UNAVAILABLE)
        if summary["classifier_status"] == "classified":
            assert summary["predicted_motion_class"] in ("normal_activity", "fall", "impact")
            assert summary["motion_probabilities"]
        else:
            assert summary["predicted_motion_class"] is None

        stored = db.cases.find_one({"case_id": case_id})
        assert stored["sensor_summary"]["predicted_motion_class"] == summary["predicted_motion_class"]
        assert stored["sensor_summary"]["classifier_status"] == summary["classifier_status"]
        got = client.get(f"/api/cases/{case_id}")
        assert got.status_code == 200
        assert got.json()["sensor_summary"]["classifier_status"] == summary["classifier_status"]
        results[scenario] = summary["classifier_status"]
        db.cases.delete_one({"case_id": case_id})
    assert results


def test_unknown_scenario_is_rejected():
    resp = client.post("/api/cases", json={})
    case_id = resp.json()["case_id"]
    bad = client.post(f"/api/cases/{case_id}/sensor/simulate", json={"scenario": "football"})
    assert bad.status_code == 400
    assert "football_fall" in bad.json()["detail"]


def test_classifier_missing_features_are_feature_missing():
    clf = SensorClassifier()
    out = clf.predict_from_summary({"peak_g_force": 4.2})
    assert out["status"] == FEATURE_MISSING or out["status"] == MODEL_UNAVAILABLE
    assert out["predicted_motion_class"] is None
    if out["status"] == FEATURE_MISSING:
        assert "peak_acceleration" in out["missing_features"]


def test_skipped_sensor_does_not_invent_kinetics():
    resp = client.post("/api/cases", json={})
    case_id = resp.json()["case_id"]
    skip = client.post(f"/api/cases/{case_id}/sensor/skip")
    assert skip.status_code == 200
    stored = get_database().cases.find_one({"case_id": case_id})
    assert stored["sensor_summary"] is None or stored.get("sensor_available") is False
    get_database().cases.delete_one({"case_id": case_id})


def test_nan_gyro_is_feature_missing(tmp_path):
    from ml.sensor.sensor_processor import process_sensor_data

    rows = ["timestamp,accelerometer_x,accelerometer_y,accelerometer_z,gyroscope_x,gyroscope_y,gyroscope_z\n"]
    for i in range(50):
        t = round(i * 0.02, 3)
        rows.append(f"{t},0.1,9.81,0.1,nan,nan,nan\n")
    csv_path = tmp_path / "nan_gyro.csv"
    csv_path.write_text("".join(rows), encoding="utf-8")
    summary = process_sensor_data(str(csv_path))
    assert summary["gyro_variance"] is None
    assert summary["classifier_status"] == FEATURE_MISSING
    assert summary["predicted_motion_class"] is None
    assert "gyro_variance" in summary["motion_classification"]["missing_features"]


def test_missing_artifact_is_model_unavailable():
    clf = SensorClassifier(model_path="ml/models/does_not_exist.json", scaler_path="ml/models/does_not_exist.pkl")
    out = clf.predict_from_summary({
        "peak_g_force": 4.2,
        "peak_acceleration": 41.0,
        "peak_jerk_gs": 10.0,
        "accel_variance": 1.0,
        "gyro_variance": 0.2,
        "sma": 12.0,
        "post_impact_stabilization_seconds": 0.4,
    })
    assert out["status"] == MODEL_UNAVAILABLE
    assert out["predicted_motion_class"] is None


def test_analyze_keeps_persisted_classifier():
    db = get_database()
    img = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample", "image", "football_injury.jpg")
    resp = client.post("/api/cases", json={"notes": "sensor analyze persist"})
    case_id = resp.json()["case_id"]
    with open(img, "rb") as f:
        up = client.post(f"/api/cases/{case_id}/image", files={"file": ("football_injury.jpg", f, "image/jpeg")})
    assert up.status_code == 200
    q = client.post(f"/api/cases/{case_id}/questionnaire", json={
        "answers": {"pain_level": "4", "cause": "fall"},
        "answer_source": "typed",
    })
    assert q.status_code == 200
    sim = client.post(f"/api/cases/{case_id}/sensor/simulate", json={"scenario": "sudden_impact"})
    assert sim.status_code == 200
    before = sim.json()["summary"]
    analyzed = client.post(f"/api/cases/{case_id}/analyze")
    assert analyzed.status_code == 200, analyzed.text
    stored = db.cases.find_one({"case_id": case_id})
    got = client.get(f"/api/cases/{case_id}").json()
    assert stored["sensor_summary"]["predicted_motion_class"] == before["predicted_motion_class"]
    assert stored["sensor_summary"]["classifier_status"] == before["classifier_status"]
    assert got["sensor_summary"]["predicted_motion_class"] == before["predicted_motion_class"]
    db.cases.delete_one({"case_id": case_id})


def test_live_upload_missing_gyro_is_feature_missing():
    resp = client.post("/api/cases", json={})
    case_id = resp.json()["case_id"]
    samples = []
    t0 = 1_700_000_000_000
    for i in range(20):
        samples.append({
            "timestamp": t0 + i * 20,
            "acceleration_gravity_x": 0.1,
            "acceleration_gravity_y": 9.81,
            "acceleration_gravity_z": 0.1,
        })
    live = client.post(f"/api/cases/{case_id}/sensor/live/upload", json={
        "source_type": "live",
        "samples": samples,
        "recording_duration_seconds": 0.4,
        "observed_sampling_rate_hz": 50.0,
    })
    assert live.status_code == 422
    assert "FEATURE_MISSING" in str(live.json()["detail"])
    get_database().cases.delete_one({"case_id": case_id})
