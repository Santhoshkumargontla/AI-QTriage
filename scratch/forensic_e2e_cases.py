"""End-to-end forensic cases via TestClient: upload → questionnaire → sensor → analyze → GET."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("PYTHONPATH", str(ROOT))

from backend.main import app
from backend.database.connection import get_database

client = TestClient(app)
IMG_DIR = ROOT / "scratch" / "forensic_suite_2026_08_29" / "test_images"
OUT = ROOT / "scratch" / "forensic_suite_2026_08_29" / "e2e_cases.json"

CASES = [
    ("synth_cut.png", "synth_cut", {"pain_level": 7, "location": "Left Hand", "open_wound": "yes", "bruising_discoloration": "no"}),
    ("synth_bruise.png", "synth_bruise", {"pain_level": 5, "location": "Arm", "open_wound": "no", "bruising_discoloration": "yes"}),
    ("blank_black.png", "blank_black", {"pain_level": 1, "location": "Other"}),
    ("normal_skin.png", "normal_skin", {"pain_level": 2, "location": "Arm", "open_wound": "no"}),
    ("unrelated_scene.png", "unrelated", {"pain_level": 0, "location": "Other"}),
]


def run_one(img_name: str, tag: str, answers: dict, sensor_mode: str) -> dict:
    create = client.post("/api/cases", json={"notes": f"forensic_{tag}"})
    assert create.status_code in (200, 201), create.text
    case_id = create.json()["case_id"]
    img_path = IMG_DIR / img_name
    raw = img_path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    up = client.post(
        f"/api/cases/{case_id}/image",
        files={"file": (img_name, raw, "image/png")},
    )
    upload_ok = up.status_code == 200
    upload_detail = up.json() if up.headers.get("content-type", "").startswith("application/json") else up.text

    q = client.post(
        f"/api/cases/{case_id}/questionnaire",
        json={"answers": answers},
    )

    sensor = None
    if sensor_mode == "demo":
        sensor = client.post(f"/api/cases/{case_id}/sensor/demo")
    elif sensor_mode == "simulate_fall":
        sensor = client.post(
            f"/api/cases/{case_id}/sensor/simulate",
            json={"scenario": "football_fall"},
        )
    elif sensor_mode == "simulate_normal":
        sensor = client.post(
            f"/api/cases/{case_id}/sensor/simulate",
            json={"scenario": "normal_movement"},
        )
    elif sensor_mode == "skip":
        sensor = client.post(f"/api/cases/{case_id}/sensor/skip")

    analyze = None
    get = None
    mongo = None
    if upload_ok:
        analyze = client.post(f"/api/cases/{case_id}/analyze")
        get = client.get(f"/api/cases/{case_id}")
        doc = get_database().cases.find_one({"case_id": case_id}) or {}
        vi = doc.get("visible_injury") or {}
        mongo = {
            "image_sha256": doc.get("image_sha256"),
            "yolo_finding": vi.get("yolo_finding"),
            "yolo_confidence": vi.get("yolo_confidence"),
            "yolo_bbox": vi.get("yolo_bounding_box"),
            "classifier_finding": vi.get("classifier_finding"),
            "classifier_model_status": vi.get("classifier_model_status"),
            "classifier_is_confident": vi.get("classifier_is_confident"),
            "classifier_abstention": vi.get("classifier_abstention_class"),
            "segmentation_reliable": vi.get("segmentation_reliable"),
            "source_type": vi.get("source_type"),
            "gradcam_status": vi.get("gradcam_explanation_status"),
            "xgb": (doc.get("xgboost_prediction") or {}).get("class"),
            "vqc": {
                "class": (doc.get("quantum_prediction") or {}).get("class"),
                "used": (doc.get("quantum_prediction") or {}).get("used_in_main_decision"),
                "status": (doc.get("quantum_prediction") or {}).get("status"),
            },
            "clinical_claim": doc.get("clinical_claim"),
            "sensor_source": doc.get("sensor_source_type"),
        }

    return {
        "tag": tag,
        "case_id": case_id,
        "file_sha": file_sha,
        "upload_status": up.status_code,
        "upload_ok": upload_ok,
        "upload_detail": upload_detail if not upload_ok else {"image_sha256": upload_detail.get("image_sha256")},
        "questionnaire_status": q.status_code,
        "sensor_mode": sensor_mode,
        "sensor_status": None if sensor is None else sensor.status_code,
        "analyze_status": None if analyze is None else analyze.status_code,
        "get_status": None if get is None else get.status_code,
        "mongo": mongo,
        "api_visible_injury": None
        if get is None or get.status_code != 200
        else {
            k: (get.json().get("visible_injury") or {}).get(k)
            for k in [
                "yolo_finding",
                "yolo_confidence",
                "yolo_bounding_box",
                "classifier_finding",
                "classifier_model_status",
                "classifier_is_confident",
                "classifier_abstention_class",
                "segmentation_reliable",
                "gradcam_explanation_status",
                "source_type",
                "image_sha256",
            ]
        },
    }


def main():
    rows = []
    # Mix sensor scenarios
    modes = ["demo", "simulate_fall", "simulate_normal", "skip", "demo"]
    for (img, tag, answers), mode in zip(CASES, modes):
        try:
            rows.append(run_one(img, tag, answers, mode))
        except Exception as exc:
            rows.append({"tag": tag, "error": f"{type(exc).__name__}: {exc}"})
    # SOS local simulation on last successful case
    sos = None
    for row in reversed(rows):
        if row.get("case_id") and row.get("analyze_status") == 200:
            cid = row["case_id"]
            trig = client.post(f"/api/cases/{cid}/sos/demo/trigger", json={"mode": "local_demo"})
            poll = client.get(f"/api/cases/{cid}")
            sos = {
                "case_id": cid,
                "trigger_status": trig.status_code,
                "trigger_body": trig.json() if trig.status_code < 500 else trig.text[:500],
                "sos_status": (poll.json() or {}).get("sos_status") if poll.status_code == 200 else None,
            }
            break

    # registry / models
    models = client.get("/api/models")
    registry = client.get("/api/models/registry")
    health = client.get("/api/health")

    out = {
        "cases": rows,
        "sos": sos,
        "health": health.json() if health.status_code == 200 else health.text,
        "models_status": models.status_code,
        "effnet_api": next(
            (m for m in (models.json() if models.status_code == 200 else []) if m.get("model_name") == "EfficientNetV2"),
            None,
        ),
        "registry_effnet": (registry.json() or {}).get("EfficientNetV2 Classification")
        if registry.status_code == 200
        else None,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("WROTE", OUT)
    for row in rows:
        print(
            row.get("tag"),
            "upload",
            row.get("upload_status"),
            "analyze",
            row.get("analyze_status"),
            "mongo",
            row.get("mongo"),
        )
    print("SOS", sos)


if __name__ == "__main__":
    main()
