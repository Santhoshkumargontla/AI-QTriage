"""Full-app forensic: YOLO/EffNet/UNet vs API vs Mongo. No metric edits."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fastapi.testclient import TestClient
from backend.main import app
from backend.database.connection import get_database
from ml.models.canonical_paths import (
    EFFNET_CANONICAL,
    UNET_CANONICAL,
    YOLO_CANONICAL,
    abs_path,
    sha256_file,
)
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
from ml.vision.unet_wrapper import UNetSegmenter
from ml.vision.yolo_wrapper import YOLO11Detector
from ml.vision.preprocess import preprocess_image_for_inference

OUT = os.path.join("scratch", "full_app_forensic.json")
THRESHOLDS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def load_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def yolo_raw(model, path, conf):
    res = model(path, conf=conf, verbose=False)[0]
    names = res.names if isinstance(res.names, dict) else {i: n for i, n in enumerate(res.names)}
    dets = []
    if res.boxes is not None:
        for box in res.boxes:
            dets.append({
                "class_id": int(box.cls[0]),
                "class_name": str(names.get(int(box.cls[0]), int(box.cls[0]))).lower(),
                "confidence": float(box.conf[0]),
                "xyxy": [float(v) for v in box.xyxy[0].tolist()],
            })
    return dets


def main():
    report = {"created_utc": datetime.now(timezone.utc).isoformat()}
    yolo = YOLO11Detector()
    info = yolo.get_info()
    report["yolo_runtime"] = {
        "canonical": YOLO_CANONICAL,
        "disk_sha": sha256_file(YOLO_CANONICAL),
        "wrapper_sha": info.get("artifact_sha256"),
        "names": info.get("classes"),
        "task": info.get("task"),
        "infer_conf": info.get("infer_conf"),
        "swelling_in_yolo_names": "swelling" in [str(c).lower() for c in (info.get("classes") or [])],
        "status": info.get("status"),
    }

    eff = EfficientNetV2Classifier()
    unet = UNetSegmenter()
    report["effnet_runtime"] = {
        "path": eff.model_path,
        "disk_sha": sha256_file(EFFNET_CANONICAL),
        "classes": eff.classes,
        "n_classes": len(eff.classes),
        "loaded": eff.is_loaded,
    }
    report["unet_runtime"] = {
        "path": unet.model_path if hasattr(unet, "model_path") else None,
        "disk_sha": sha256_file(UNET_CANONICAL),
        "loaded": getattr(unet, "is_loaded", None),
    }

    images = {
        "demo_football": os.path.join("data", "sample", "image", "football_injury.jpg"),
        "forensic_upload": os.path.join("data", "uploads", "3e0dbd17-7475-487d-9f10-e7f9d6800238.jpg"),
        "heldout_bruise": os.path.join(
            "data", "datasets", "yolo_retrain_v2", "images", "test",
            "raw_synthetic_wound__syn_wound_0185.jpg",
        ),
        "heldout_cut": os.path.join(
            "data", "datasets", "yolo_retrain_v2", "images", "test",
            "raw_synthetic_wound__syn_wound_0186.jpg",
        ),
        "blank_skin": os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg"),
    }

    probes = {}
    gray = np.full((224, 224, 3), 128, dtype=np.uint8)
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    white = np.full((224, 224, 3), 255, dtype=np.uint8)
    os.makedirs("data/test_suite", exist_ok=True)
    cv2.imwrite("data/test_suite/forensic_gray.jpg", gray)
    cv2.imwrite("data/test_suite/forensic_black.jpg", black)
    cv2.imwrite("data/test_suite/forensic_white.jpg", white)
    images["gray"] = "data/test_suite/forensic_gray.jpg"
    images["black"] = "data/test_suite/forensic_black.jpg"
    images["white"] = "data/test_suite/forensic_white.jpg"

    per_image = {}
    for label, path in images.items():
        if not os.path.isfile(path):
            per_image[label] = {"missing": True, "path": path}
            continue
        rgb = load_rgb(path)
        h, w = rgb.shape[:2]
        exif_orient = None
        wrapper = yolo.detect(path)
        raw_025 = yolo_raw(yolo.model, path, 0.25)
        raw_001 = yolo_raw(yolo.model, path, 0.01)
        sweep = {}
        for thr in THRESHOLDS:
            dets = yolo_raw(yolo.model, path, thr)
            sweep[str(thr)] = {
                "n": len(dets),
                "max": max((d["confidence"] for d in dets), default=None),
                "classes": sorted({d["class_name"] for d in dets}),
            }
        gated = eff.predict(rgb)
        raw_clf = eff.predict_raw(rgb)
        mask, pix, ratio, debug = unet.segment(rgb)
        pos = float(np.mean(mask > 0)) if mask is not None else None
        tensor, letterboxed, meta = preprocess_image_for_inference(path)
        per_image[label] = {
            "path": path.replace("\\", "/"),
            "width": w,
            "height": h,
            "wrapper_detect": wrapper,
            "raw_at_0_25": raw_025,
            "raw_at_0_01_n": len(raw_001),
            "raw_at_0_01_max": max((d["confidence"] for d in raw_001), default=None),
            "raw_at_0_01_classes": sorted({d["class_name"] for d in raw_001}),
            "threshold_sweep": sweep,
            "effnet_gated": {
                "winner": gated.get("__winner"),
                "raw_winner": gated.get("__raw_winner"),
                "raw_max": gated.get("__raw_max_prob"),
                "status": gated.get("__status"),
                "reason": gated.get("__reason"),
                "confident": gated.get("__is_confident"),
            },
            "effnet_raw": raw_clf,
            "unet_positive_ratio": pos,
            "unet_pixel_count": int(pix) if pix is not None else None,
            "unet_affected_ratio": ratio,
            "unet_debug_status": (debug or {}).get("status") if isinstance(debug, dict) else debug,
            "preprocess_meta": meta,
            "swelling_in_yolo_raw": any(d["class_name"] == "swelling" for d in raw_001 + raw_025),
        }
        print(
            label,
            "yolo_n@0.25", len(wrapper),
            "eff_raw", raw_clf.get("winner"), round(raw_clf.get("max_prob") or 0, 3),
            "eff_gated", gated.get("__winner"),
            "unet_pos", None if pos is None else round(pos, 4),
        )

    report["per_image"] = per_image

    client = TestClient(app)
    e2e = {}
    db = get_database()
    for key, path in (
        ("demo_football", images["demo_football"]),
        ("heldout_bruise", images["heldout_bruise"]),
    ):
        create = client.post("/api/cases", json={"notes": f"full_forensic_{key}"})
        case_id = create.json()["case_id"]
        with open(path, "rb") as handle:
            client.post(
                f"/api/cases/{case_id}/image",
                files={"file": (os.path.basename(path), handle, "image/jpeg")},
            )
        client.post(
            f"/api/cases/{case_id}/questionnaire",
            json={"answers": {"pain_level": "4", "cause": "direct_blow"}, "answer_source": "typed"},
        )
        client.post(f"/api/cases/{case_id}/sensor/skip")
        analyze = client.post(f"/api/cases/{case_id}/analyze")
        gotten = client.get(f"/api/cases/{case_id}")
        api = gotten.json()
        vi = api.get("visible_injury") or {}
        mongo = db.cases.find_one({"case_id": case_id}, {"_id": 0, "visible_injury": 1, "classical_prediction": 1, "quantum_prediction": 1})
        mvi = (mongo or {}).get("visible_injury") or {}
        direct = per_image[key]["wrapper_detect"]
        e2e[key] = {
            "case_id": case_id,
            "analyze_status": analyze.status_code,
            "direct_n": len(direct),
            "direct_class": [d.get("finding") for d in direct],
            "direct_conf": [d.get("confidence") for d in direct],
            "direct_bbox": [d.get("bounding_box") for d in direct],
            "api_yolo_detected": vi.get("yolo_finding_detected"),
            "api_yolo_finding": vi.get("yolo_finding"),
            "api_yolo_conf": vi.get("yolo_confidence"),
            "api_yolo_bbox": vi.get("yolo_bounding_box"),
            "api_finding_legacy": vi.get("finding"),
            "api_classifier_finding": vi.get("classifier_finding"),
            "api_classifier_status": vi.get("classifier_status"),
            "api_bbox": vi.get("bounding_box"),
            "api_overlay_wh": [vi.get("overlay_width"), vi.get("overlay_height")],
            "api_original_wh": [vi.get("original_width"), vi.get("original_height")],
            "mongo_yolo_finding": mvi.get("yolo_finding"),
            "mongo_yolo_detected": mvi.get("yolo_finding_detected"),
            "mongo_yolo_bbox": mvi.get("yolo_bounding_box"),
            "mongo_classifier_finding": mvi.get("classifier_finding"),
            "direct_equals_api_bbox": (direct[0]["bounding_box"] if direct else None) == vi.get("yolo_bounding_box"),
            "api_equals_mongo_bbox": vi.get("yolo_bounding_box") == mvi.get("yolo_bounding_box"),
            "api_equals_mongo_class": vi.get("yolo_finding") == mvi.get("yolo_finding"),
            "xgb": (api.get("classical_prediction") or {}).get("severity_category")
            or (api.get("classical_prediction") or {}).get("predicted_class"),
        }
        print("E2E", key, e2e[key]["api_yolo_finding"], "clf", e2e[key]["api_classifier_finding"], "bbox_match", e2e[key]["direct_equals_api_bbox"])

    report["e2e"] = e2e
    sos = client.get("/api/sos/config")
    report["sos_config"] = sos.json() if sos.status_code == 200 else {"status": sos.status_code}
    models = client.get("/api/models")
    report["api_models_yolo_names"] = (models.json().get("yolo") if isinstance(models.json(), dict) else None)
    report["health"] = client.get("/api/health").json()

    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
