"""Re-analyze forensic case and dump API honesty fields."""
from __future__ import annotations

import hashlib
import json
import os

from fastapi.testclient import TestClient

from backend.database.connection import get_database
from backend.main import app

CID = "ac69884f-7a50-48e5-b7cc-a9bea9b20313"


def main() -> None:
    client = TestClient(app)
    db = get_database()
    case = db.cases.find_one({"case_id": CID})
    path = case["image_reference"]
    with open(path, "rb") as handle:
        sha = hashlib.sha256(handle.read()).hexdigest()
    db.cases.update_one({"case_id": CID}, {"$set": {"image_sha256": sha}})
    print("backfill_sha", sha)

    analyze = client.post(f"/api/cases/{CID}/analyze")
    print("analyze", analyze.status_code, analyze.text[:500])

    get = client.get(f"/api/cases/{CID}")
    assert get.status_code == 200
    data = get.json()
    vi = data.get("visible_injury") or {}
    out = {
        "yolo": (
            vi.get("yolo_finding"),
            vi.get("yolo_confidence"),
            vi.get("yolo_bounding_box"),
        ),
        "source_type": vi.get("source_type"),
        "data_provenance": vi.get("data_provenance"),
        "display_message": vi.get("display_message"),
        "image_sha256": vi.get("image_sha256"),
        "gradcam": {
            "status": vi.get("gradcam_explanation_status"),
            "generated": vi.get("gradcam_overlay_generated"),
            "reason": vi.get("gradcam_withheld_reason"),
            "model_status": vi.get("gradcam_model_status"),
            "overlay_url": vi.get("overlay_url"),
        },
        "classifier": {
            "finding": vi.get("classifier_finding"),
            "model_status": vi.get("classifier_model_status"),
            "gate": vi.get("classifier_status"),
        },
        "seg": {
            "reliable": vi.get("segmentation_reliable"),
            "reason": vi.get("segmentation_reason"),
        },
        "xgb": data.get("xgboost_prediction"),
        "vqc": {
            k: (data.get("quantum_prediction") or {}).get(k)
            for k in ["class", "status", "experimental_only", "used_in_main_decision"]
        },
        "sensor": data.get("sensor_source_type"),
        "is_demo": data.get("is_demo"),
        "overlay_exists": os.path.exists(path.replace(".jpeg", "_overlay.jpg")),
        "disk_sha_match": vi.get("image_sha256") == sha,
    }
    print(json.dumps(out, indent=2, default=str))
    with open(
        os.path.join("scratch", "real_image_e2e_reanalyze.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(out, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
