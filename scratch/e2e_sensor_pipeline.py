"""E2E sensor pipeline: simulate -> MongoDB -> GET -> analyze. Measured numbers only."""
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient
from backend.main import app
from backend.database.connection import get_database
from ml.sensor.scenarios import SUPPORTED_SCENARIOS, SCENARIO_ALIASES

client = TestClient(app)
out = {
    "started": datetime.now(timezone.utc).isoformat(),
    "supported_scenarios": list(SUPPORTED_SCENARIOS),
    "aliases": dict(SCENARIO_ALIASES),
    "scenarios": {},
}

img = os.path.join(ROOT, "data", "sample", "image", "football_injury.jpg")
db = get_database()

for scenario in SUPPORTED_SCENARIOS:
    cr = client.post("/api/cases", json={"notes": f"e2e_sensor_{scenario}"})
    case_id = cr.json()["case_id"]
    with open(img, "rb") as f:
        client.post(f"/api/cases/{case_id}/image", files={"file": ("football_injury.jpg", f, "image/jpeg")})
    client.post(
        f"/api/cases/{case_id}/questionnaire",
        json={"answers": {"pain_level": "5", "cause": "fall"}, "answer_source": "typed"},
    )
    sim = client.post(f"/api/cases/{case_id}/sensor/simulate", json={"scenario": scenario})
    rec = {"simulate_http": sim.status_code, "case_id": case_id}
    if sim.status_code != 200:
        rec["error"] = sim.text[:400]
        out["scenarios"][scenario] = rec
        continue
    body = sim.json()
    s = body.get("summary") or {}
    rec.update({
        "sos_triggered": body.get("sos_triggered"),
        "peak_g_force": s.get("peak_g_force"),
        "peak_acceleration": s.get("peak_acceleration"),
        "peak_jerk_gs": s.get("peak_jerk_gs"),
        "gyro_variance": s.get("gyro_variance"),
        "post_impact_stabilization_seconds": s.get("post_impact_stabilization_seconds"),
        "optical_lux_drop": s.get("optical_lux_drop"),
        "lux_feature_available": s.get("lux_feature_available"),
        "predicted_motion_class": s.get("predicted_motion_class"),
        "motion_confidence": s.get("motion_confidence"),
        "motion_probabilities": s.get("motion_probabilities"),
        "classifier_status": s.get("classifier_status"),
        "source_type": s.get("source_type"),
        "scenario": s.get("scenario"),
        "sample_count": s.get("sample_count"),
        "events": [e.get("event_name") for e in (s.get("events") or [])],
    })
    stored = db.cases.find_one({"case_id": case_id})
    mongo_s = (stored or {}).get("sensor_summary") or {}
    rec["mongo_matches_simulate"] = (
        mongo_s.get("predicted_motion_class") == s.get("predicted_motion_class")
        and mongo_s.get("classifier_status") == s.get("classifier_status")
        and mongo_s.get("peak_g_force") == s.get("peak_g_force")
    )
    got = client.get(f"/api/cases/{case_id}").json()
    gs = got.get("sensor_summary") or {}
    rec["get_matches_mongo"] = (
        gs.get("predicted_motion_class") == mongo_s.get("predicted_motion_class")
        and gs.get("classifier_status") == mongo_s.get("classifier_status")
    )
    an = client.post(f"/api/cases/{case_id}/analyze")
    rec["analyze_http"] = an.status_code
    after = db.cases.find_one({"case_id": case_id})
    after_s = (after or {}).get("sensor_summary") or {}
    rec["classifier_survives_analyze"] = (
        after_s.get("predicted_motion_class") == s.get("predicted_motion_class")
        and after_s.get("classifier_status") == s.get("classifier_status")
    )
    rec["xgboost_class"] = ((after or {}).get("xgboost_prediction") or {}).get("class")
    rec["frontend_motion_display"] = (
        gs.get("predicted_motion_class")
        or gs.get("classifier_status")
        or "FEATURE_MISSING"
    )
    db.cases.delete_one({"case_id": case_id})
    out["scenarios"][scenario] = rec
    print(scenario, rec["classifier_status"], rec["predicted_motion_class"], rec["peak_g_force"])

# Invalid name
cr = client.post("/api/cases", json={"notes": "bad_scenario"})
cid = cr.json()["case_id"]
bad = client.post(f"/api/cases/{cid}/sensor/simulate", json={"scenario": "football"})
out["invalid_scenario"] = {"http": bad.status_code, "detail": bad.json().get("detail")}
db.cases.delete_one({"case_id": cid})

# Direct FEATURE_MISSING / MODEL_UNAVAILABLE
from ml.classifiers.sensor_classifier import FEATURE_MISSING, MODEL_UNAVAILABLE, SensorClassifier
from ml.sensor.sensor_processor import process_sensor_data

missing = SensorClassifier().predict_from_summary({"peak_g_force": 4.2})
out["feature_missing_direct"] = {
    "status": missing.get("status"),
    "predicted_motion_class": missing.get("predicted_motion_class"),
    "missing_features": missing.get("missing_features"),
}
unavail = SensorClassifier(model_path="nope.json", scaler_path="nope.pkl").predict_from_summary({
    "peak_g_force": 4.2,
    "peak_acceleration": 41.0,
    "peak_jerk_gs": 10.0,
    "accel_variance": 1.0,
    "gyro_variance": 0.2,
    "sma": 12.0,
    "post_impact_stabilization_seconds": 0.4,
})
out["model_unavailable_direct"] = {
    "status": unavail.get("status"),
    "predicted_motion_class": unavail.get("predicted_motion_class"),
}

dest = os.path.join(ROOT, "scratch", "e2e_sensor_pipeline.json")
with open(dest, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
print("wrote", dest)
print("aliases empty", SCENARIO_ALIASES == {})
