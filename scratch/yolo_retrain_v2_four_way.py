"""Post-promotion 4-way YOLO check: direct vs API vs MongoDB. Frontend is separate."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient
from ml.models.canonical_paths import YOLO_CANONICAL, sha256_file
from ml.vision.yolo_wrapper import YOLO11Detector

IMG = os.path.join(ROOT, "data", "sample", "image", "football_injury.jpg")
BLANK = os.path.join(ROOT, "data", "datasets", "yolo_injury", "blank_skin.jpg")
SYN_CUT = os.path.join(
    ROOT,
    "data",
    "datasets",
    "yolo_retrain_v2",
    "images",
    "test",
    "raw_synthetic_wound__syn_wound_0185.jpg",
)

out = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "canonical_path": YOLO_CANONICAL.replace("\\", "/"),
    "canonical_sha256": sha256_file(YOLO_CANONICAL),
    "frontend_browser": "NOT_RUN",
}


def _findings(dets):
    return [
        {
            "finding": d.get("finding"),
            "confidence": d.get("confidence"),
            "bounding_box": d.get("bounding_box"),
        }
        for d in dets
    ]


def main():
    det = YOLO11Detector()
    info = det.get_info()
    out["direct_model"] = {
        "sha256": info.get("artifact_sha256"),
        "version": info.get("version"),
        "infer_conf": info.get("infer_conf"),
        "classes": info.get("classes"),
        "task": info.get("task"),
    }
    demo = det.detect(IMG)
    blank = det.detect(BLANK) if os.path.isfile(BLANK) else []
    out["direct_demo"] = {"n": len(demo), "findings": _findings(demo)}
    out["direct_blank"] = {"n": len(blank), "findings": _findings(blank)}
    if os.path.isfile(SYN_CUT):
        syn = det.detect(SYN_CUT)
        out["direct_heldout_synthetic"] = {
            "image": os.path.relpath(SYN_CUT, ROOT).replace("\\", "/"),
            "n": len(syn),
            "findings": _findings(syn),
        }

    from backend.main import app
    from backend.database.connection import get_database

    client = TestClient(app)
    health = client.get("/api/health")
    out["api_health"] = {"status": health.status_code, "json": health.json() if health.status_code == 200 else health.text[:300]}
    models = client.get("/api/models/registry")
    yolo_reg = None
    if models.status_code == 200:
        body = models.json()
        yolo_reg = (body.get("YOLO11 Detection") if isinstance(body, dict) else None) or body
        if isinstance(body, dict) and "YOLO11 Detection" in body:
            yolo_reg = body["YOLO11 Detection"]
        elif isinstance(body, dict) and "models" in body:
            for row in body.get("models") or []:
                if row.get("model_name") == "YOLO11 Detection":
                    yolo_reg = row
                    break
    out["api_registry_yolo_sha"] = (yolo_reg or {}).get("artifact_sha256") if isinstance(yolo_reg, dict) else None

    created = client.post("/api/cases", json={"notes": "yolo_retrain_v2_four_way"})
    if created.status_code not in (200, 201):
        out["api_error"] = {"create_case": created.status_code, "body": created.text[:400]}
        _write()
        return
    case_id = created.json()["case_id"]
    with open(IMG, "rb") as handle:
        up = client.post(
            f"/api/cases/{case_id}/image",
            files={"file": ("football_injury.jpg", handle, "image/jpeg")},
        )
    q = client.post(
        f"/api/cases/{case_id}/questionnaire",
        json={
            "answers": {
                "pain_level": "6",
                "cause": "fall",
                "bleeding": "mild",
                "movement": "limited",
                "limb_use": "with_pain",
                "location": "left ankle",
            },
            "answer_source": "typed",
        },
    )
    skip = client.post(f"/api/cases/{case_id}/sensor/skip")
    analyze = client.post(f"/api/cases/{case_id}/analyze")
    got = client.get(f"/api/cases/{case_id}")
    case = got.json() if got.status_code == 200 else {}
    vi = case.get("visible_injury") or {}
    out["api_demo"] = {
        "create": created.status_code,
        "upload": up.status_code,
        "questionnaire": q.status_code,
        "sensor_skip": skip.status_code,
        "analyze": analyze.status_code,
        "get": got.status_code,
        "yolo_finding_detected": vi.get("yolo_finding_detected"),
        "yolo_finding": vi.get("yolo_finding"),
        "yolo_confidence": vi.get("yolo_confidence"),
        "yolo_bbox": vi.get("yolo_bounding_box") or vi.get("bounding_box"),
        "finding": vi.get("finding"),
    }

    mongo = {"available": False}
    try:
        db = get_database()
        stored = db.cases.find_one(
            {"case_id": case_id},
            {"_id": 0, "visible_injury": 1},
        )
        vi_m = (stored or {}).get("visible_injury") or {}
        mongo = {
            "available": stored is not None,
            "yolo_finding_detected": vi_m.get("yolo_finding_detected"),
            "yolo_finding": vi_m.get("yolo_finding"),
            "yolo_confidence": vi_m.get("yolo_confidence"),
            "yolo_bbox": vi_m.get("yolo_bounding_box") or vi_m.get("bounding_box"),
        }
        db.cases.delete_one({"case_id": case_id})
    except Exception as exc:
        mongo = {"available": False, "error": str(exc)}
    out["mongodb_demo"] = mongo

    direct_n = out["direct_demo"]["n"]
    api_det = bool(out["api_demo"].get("yolo_finding_detected"))
    mongo_det = bool(mongo.get("yolo_finding_detected")) if mongo.get("available") else None
    out["agreement"] = {
        "direct_vs_api_detection_flag": (direct_n > 0) == api_det,
        "api_vs_mongo_detection_flag": (api_det == mongo_det) if mongo_det is not None else "MONGO_UNAVAILABLE",
        "direct_n": direct_n,
        "note": "Demo graphic is unlabeled and OOD vs synthetic drawings. Zero boxes at 0.25 is expected for the promoted candidate.",
    }
    _write()


def _write():
    dest = os.path.join(ROOT, "ml", "models", "yolo_retrain_v2", "FOUR_WAY_VERIFY.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
