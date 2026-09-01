"""E2E API forensic: upload -> analyze -> MongoDB vs direct YOLO."""
import os, sys, json
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient
from backend.main import app
from backend.database.connection import get_database
from ml.vision.yolo_wrapper import YOLO11Detector

client = TestClient(app)
out = {"started": datetime.now(timezone.utc).isoformat()}

def dump(name, resp):
    rec = {"status": resp.status_code}
    try:
        rec["json"] = resp.json()
    except Exception:
        rec["text"] = resp.text[:500]
    out[name] = rec
    print(f"{name}: HTTP {resp.status_code}")
    return rec

# Health / models / registry / evaluation / sos config
dump("health", client.get("/api/health"))
dump("models", client.get("/api/models"))
dump("registry", client.get("/api/models/registry"))
dump("evaluation", client.get("/api/evaluation"))
dump("sos_config", client.get("/api/sos/config"))

# Direct YOLO on football_injury
img = os.path.join(ROOT, "data", "sample", "image", "football_injury.jpg")
det = YOLO11Detector()
direct = det.detect(img)
out["direct_yolo"] = {
    "model_path": det.model_path,
    "n": len(direct),
    "findings": direct,
}
print("DIRECT YOLO n=", len(direct), direct)

# Create case, upload image, questionnaire, skip sensor, analyze
r = client.post("/api/cases", json={"notes": "forensic_e2e_audit"})
dump("create_case", r)
case_id = r.json()["case_id"]
print("case_id", case_id)

with open(img, "rb") as f:
    ru = client.post(f"/api/cases/{case_id}/image", files={"file": ("football_injury.jpg", f, "image/jpeg")})
dump("upload_image", ru)

rq = client.post(f"/api/cases/{case_id}/questionnaire", json={
    "answers": {"pain_level": "6", "cause": "fall", "bleeding": "mild", "movement": "limited", "limb_use": "with_pain", "location": "left ankle"},
    "answer_source": "typed"
})
dump("questionnaire", rq)

rs = client.post(f"/api/cases/{case_id}/sensor/skip")
dump("sensor_skip", rs)

ra = client.post(f"/api/cases/{case_id}/analyze")
dump("analyze", ra)

rg = client.get(f"/api/cases/{case_id}")
dump("get_case", rg)
case = rg.json()
vi = case.get("visible_injury") or {}
out["layer_compare_demo_image"] = {
    "direct_n": len(direct),
    "direct_class": [d.get("finding") for d in direct],
    "direct_conf": [d.get("confidence") for d in direct],
    "direct_bbox": [d.get("bounding_box") for d in direct],
    "api_yolo_finding_detected": vi.get("yolo_finding_detected"),
    "api_yolo_finding": vi.get("yolo_finding"),
    "api_yolo_confidence": vi.get("yolo_confidence"),
    "api_yolo_bbox": vi.get("yolo_bounding_box"),
    "api_finding": vi.get("finding"),
    "api_finding_detected": vi.get("finding_detected"),
    "api_bbox": vi.get("bounding_box"),
    "api_classifier_finding": vi.get("classifier_finding"),
    "api_classifier_probability": vi.get("classifier_probability"),
    "api_overlay_url": vi.get("overlay_url"),
    "api_mask_url": vi.get("mask_url"),
    "api_affected_ratio": vi.get("affected_ratio"),
    "api_model_path": vi.get("model_path"),
    "mongo_status": case.get("status"),
    "xgboost": case.get("xgboost_prediction"),
    "vqc": case.get("quantum_prediction"),
}
print("LAYER COMPARE", json.dumps(out["layer_compare_demo_image"], indent=2, default=str))

# Second case: dataset val image that had 0 detections at 0.10
img2 = os.path.join(ROOT, "data", "datasets", "yolo_real_wound", "images", "val", "public_sample_0058.jpg")
if os.path.exists(img2):
    direct2 = det.detect(img2)
    r2 = client.post("/api/cases", json={"notes": "forensic_e2e_val_image"})
    cid2 = r2.json()["case_id"]
    with open(img2, "rb") as f:
        client.post(f"/api/cases/{cid2}/image", files={"file": ("public_sample_0058.jpg", f, "image/jpeg")})
    client.post(f"/api/cases/{cid2}/questionnaire", json={"answers": {"pain_level": "3", "cause": "other"}, "answer_source": "typed"})
    client.post(f"/api/cases/{cid2}/sensor/skip")
    a2 = client.post(f"/api/cases/{cid2}/analyze")
    g2 = client.get(f"/api/cases/{cid2}")
    vi2 = (g2.json().get("visible_injury") or {})
    out["layer_compare_val_image"] = {
        "direct_n": len(direct2),
        "direct_findings": direct2,
        "analyze_status": a2.status_code,
        "api_yolo_finding_detected": vi2.get("yolo_finding_detected"),
        "api_yolo_finding": vi2.get("yolo_finding"),
        "api_yolo_confidence": vi2.get("yolo_confidence"),
        "api_bbox": vi2.get("bounding_box"),
        "api_classifier_finding": vi2.get("classifier_finding"),
        "api_classifier_probability": vi2.get("classifier_probability"),
    }
    print("VAL IMAGE COMPARE", json.dumps(out["layer_compare_val_image"], indent=2, default=str))
    db = get_database()
    db.cases.delete_one({"case_id": cid2})

# Sensor simulate paths
for scenario in ("normal_movement", "football_fall", "sudden_impact"):
    rc = client.post("/api/cases", json={"notes": f"sensor_{scenario}"})
    sid = rc.json()["case_id"]
    sim = client.post(f"/api/cases/{sid}/sensor/simulate", json={"scenario": scenario})
    rec = {"status": sim.status_code}
    if sim.status_code == 200:
        s = sim.json().get("summary") or {}
        rec["peak_g_force"] = s.get("peak_g_force")
        rec["delta_v"] = s.get("pre_impact_delta_v")
        rec["stabilization"] = s.get("post_impact_stabilization_seconds")
        rec["lux_drop"] = s.get("optical_lux_drop")
        rec["source_type"] = s.get("source_type")
        rec["sos_triggered"] = sim.json().get("sos_triggered")
        rec["has_motion_class"] = "predicted_motion_class" in s
    else:
        rec["body"] = sim.text[:300]
    out[f"sensor_{scenario}"] = rec
    print(f"sensor {scenario}:", rec)
    get_database().cases.delete_one({"case_id": sid})

# SOS trigger local_demo (no Twilio)
sos = client.post(f"/api/cases/{case_id}/sos/demo/trigger", json={"mode": "local_demo"})
dump("sos_trigger_local", sos)
sos_t = client.post(f"/api/cases/{case_id}/sos/demo/trigger", json={"mode": "twilio_test"})
dump("sos_trigger_twilio_attempt", sos_t)

# cleanup primary case? keep for mongo compare then delete
db = get_database()
stored = db.cases.find_one({"case_id": case_id}, {"_id": 0, "visible_injury": 1, "xgboost_prediction": 1, "quantum_prediction": 1, "sos_status": 1})
if stored:
    vi_m = stored.get("visible_injury") or {}
    out["mongodb_stored"] = {
        "yolo_finding_detected": vi_m.get("yolo_finding_detected"),
        "yolo_finding": vi_m.get("yolo_finding"),
        "yolo_confidence": vi_m.get("yolo_confidence"),
        "yolo_bbox": vi_m.get("yolo_bounding_box"),
        "xgboost": stored.get("xgboost_prediction"),
        "vqc": stored.get("quantum_prediction"),
        "sos_status": stored.get("sos_status"),
    }
    print("MONGO", json.dumps(out["mongodb_stored"], indent=2, default=str))
db.cases.delete_one({"case_id": case_id})
db.sos_events.delete_many({"case_id": case_id})

dest = os.path.join(ROOT, "scratch", "e2e_api_forensic.json")
with open(dest, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
print("wrote", dest)
