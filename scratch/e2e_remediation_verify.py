"""Live E2E verification against a running FastAPI backend. Does not fabricate detections."""
import os
import sys
import json
import hashlib
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
from ml.models.canonical_paths import YOLO_CANONICAL
from ml.vision.yolo_wrapper import YOLO11Detector

API = os.environ.get("AIQTRIAGE_API", "http://127.0.0.1:8000")
OUT = os.path.join("scratch", "e2e_remediation_verify.json")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def direct_yolo(path, conf=0.10):
    model = YOLO(YOLO_CANONICAL)
    result = model(path, conf=conf, verbose=False)[0]
    boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            cid = int(box.cls[0].item())
            boxes.append({
                "class": str(model.names.get(cid, cid)),
                "confidence": round(float(box.conf[0].item()), 4),
                "bounding_box": [round(float(v), 2) for v in box.xyxy[0].cpu().numpy().tolist()],
            })
    wrapper = YOLO11Detector()
    wrapped = wrapper.detect(path, conf=conf)
    return boxes, wrapped


def workflow(image_path, label, answers=None, scenario="normal"):
    rec = {"label": label, "image": image_path}
    r = requests.post(f"{API}/api/cases", json={"notes": f"e2e {label}"}, timeout=30)
    rec["create_status"] = r.status_code
    rec["create"] = r.json() if r.ok else r.text[:300]
    if not r.ok:
        return rec
    case_id = r.json()["case_id"]
    rec["case_id"] = case_id
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            up = requests.post(f"{API}/api/cases/{case_id}/image", files={"file": (os.path.basename(image_path), f, "image/jpeg")}, timeout=60)
        rec["image_status"] = up.status_code
    answers = answers or {"pain_level": 4, "swelling": "no", "crack_pop": "no", "mechanism": "sports"}
    q = requests.post(f"{API}/api/cases/{case_id}/questionnaire", json={"answers": answers}, timeout=30)
    rec["questionnaire_status"] = q.status_code
    s = requests.post(f"{API}/api/cases/{case_id}/sensor/simulate", json={"scenario": scenario}, timeout=30)
    rec["sensor_status"] = s.status_code
    rec["sensor"] = s.json() if s.ok else s.text[:400]
    a = requests.post(f"{API}/api/cases/{case_id}/analyze", timeout=180)
    rec["analyze_status"] = a.status_code
    rec["analyze"] = a.json() if a.ok else a.text[:800]
    g = requests.get(f"{API}/api/cases/{case_id}", timeout=30)
    rec["get_status"] = g.status_code
    case = g.json() if g.ok else {}
    vis = case.get("visible_injury") or {}
    rec["stored_detection"] = {
        "finding": vis.get("finding"),
        "yolo_finding": vis.get("yolo_finding"),
        "confidence": vis.get("yolo_confidence") or vis.get("confidence"),
        "bounding_box": vis.get("yolo_bounding_box") or vis.get("bounding_box"),
        "finding_detected": vis.get("yolo_finding_detected") if vis.get("yolo_finding_detected") is not None else vis.get("finding_detected"),
    }
    rec["xgboost"] = case.get("xgboost_prediction")
    rec["vqc"] = case.get("quantum_prediction")
    rec["sensor_motion"] = {
        "predicted_motion_class": (case.get("sensor_summary") or {}).get("predicted_motion_class"),
        "classifier_status": (case.get("sensor_summary") or {}).get("classifier_status"),
    }
    if image_path and os.path.exists(image_path):
        direct, wrapped = direct_yolo(image_path)
        rec["direct_yolo"] = direct
        rec["wrapper_yolo"] = wrapped
        if direct and rec["stored_detection"]["bounding_box"]:
            rec["box_match"] = (
                direct[0]["class"] == (rec["stored_detection"]["yolo_finding"] or rec["stored_detection"]["finding"] or "").lower()
            )
    return rec


def sos_cases():
    out = {}
    cfg = requests.get(f"{API}/api/sos/config", timeout=15)
    out["twilio_config"] = cfg.json() if cfg.ok else {"status": cfg.status_code, "body": cfg.text[:300]}
    return out


def main():
    report = {
        "api": API,
        "yolo_canonical": YOLO_CANONICAL,
        "yolo_sha256": _sha256(YOLO_CANONICAL) if os.path.exists(YOLO_CANONICAL) else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "health": None,
        "workflows": [],
        "sos": None,
    }
    try:
        h = requests.get(f"{API}/docs", timeout=5)
        report["health"] = h.status_code
    except requests.RequestException as e:
        report["health"] = str(e)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Backend not reachable:", e)
        return report

    images = [
        ("demo", os.path.join("data", "sample", "image", "football_injury.jpg"), {"pain_level": 6, "swelling": "yes", "crack_pop": "no", "mechanism": "sports"}, "fall"),
    ]
    val_dir = os.path.join("data", "datasets", "yolo_real_wound", "images", "val")
    if os.path.isdir(val_dir):
        names = [n for n in sorted(os.listdir(val_dir)) if n.lower().endswith((".jpg", ".png"))]
        if names:
            images.append(("val_first", os.path.join(val_dir, names[0]), {"pain_level": 3, "swelling": "no", "crack_pop": "no"}, "normal"))
    blank = os.path.join("scratch", "blank_gray.jpg")
    import cv2
    import numpy as np
    cv2.imwrite(blank, np.full((256, 256, 3), 180, dtype=np.uint8))
    images.append(("blank_gray", blank, {"pain_level": 1, "swelling": "no", "crack_pop": "no"}, "normal"))

    for label, path, answers, scenario in images:
        print("workflow", label)
        report["workflows"].append(workflow(path, label, answers, scenario))

    report["sos"] = sos_cases()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print("Wrote", OUT)
    return report


if __name__ == "__main__":
    main()
