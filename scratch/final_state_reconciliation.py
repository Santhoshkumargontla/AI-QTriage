"""Final-state reconciliation probes. Evidence only — no mutations."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.abspath("."))

from ml.models.canonical_paths import (
    YOLO_CANONICAL,
    EFFNET_CANONICAL,
    UNET_CANONICAL,
    XGB_CANONICAL,
    VQC_DIR,
    VQC_WEIGHTS,
    SENSOR_MODEL,
    REGISTRY_PATH,
    MANIFEST_PATH,
    sha256_file,
    resolve_existing,
    read_json,
)

OUT = os.path.join("scratch", "final_state_reconciliation.json")
os.makedirs("scratch", exist_ok=True)


def probe_yolo():
    from ml.vision.yolo_wrapper import YOLO11Detector
    from ultralytics import YOLO

    det = YOLO11Detector()
    info = det.get_info()
    disk = sha256_file(YOLO_CANONICAL)
    direct = YOLO(resolve_existing(YOLO_CANONICAL))
    images = {}
    # blank
    blank = os.path.join("scratch", "_recon_blank.jpg")
    import cv2

    cv2.imwrite(blank, np.full((256, 256, 3), 180, dtype=np.uint8))
    for label, path, conf in [
        ("blank", blank, 0.10),
        ("demo", os.path.join("data", "sample", "image", "football_injury.jpg"), 0.10),
    ]:
        if not os.path.exists(path):
            images[label] = {"missing": True}
            continue
        wrap = det.detect(path, conf=conf) if det.model is not None else []
        raw = direct(path, conf=conf, verbose=False)[0]
        raw_boxes = []
        if raw.boxes is not None:
            for b in raw.boxes:
                cid = int(b.cls[0].item())
                raw_boxes.append(
                    {
                        "class": str(direct.names.get(cid, cid)),
                        "conf": round(float(b.conf[0].item()), 4),
                        "xyxy": [round(float(v), 2) for v in b.xyxy[0].cpu().numpy().tolist()],
                    }
                )
        images[label] = {
            "wrapper_n": len(wrap),
            "direct_n": len(raw_boxes),
            "wrapper_top": wrap[:3],
            "direct_top": raw_boxes[:3],
            "match_top": (
                wrap
                and raw_boxes
                and wrap[0]["finding"] == raw_boxes[0]["class"]
                and abs(wrap[0]["confidence"] - raw_boxes[0]["conf"]) < 1e-3
            ),
        }
    return {
        "wrapper_path": info.get("model_path"),
        "wrapper_sha_claimed": info.get("sha256") or info.get("artifact_sha256"),
        "disk_sha": disk,
        "task": getattr(det.model, "task", None) if det.model else None,
        "names": dict(det.model.names) if det.model else None,
        "supported_classes": sorted(det.supported_classes),
        "untrained_classes": getattr(det, "untrained_classes", None),
        "status": det.status,
        "images": images,
    }


def probe_effnet():
    from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
    import cv2

    clf = EfficientNetV2Classifier()
    disk = sha256_file(EFFNET_CANONICAL)
    probes = {}
    for name, val in [("gray", 180), ("black", 0), ("white", 255)]:
        img = np.full((224, 224, 3), val, dtype=np.uint8)
        out = clf.predict(img)
        probes[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    demo = os.path.join("data", "sample", "image", "football_injury.jpg")
    if os.path.exists(demo):
        bgr = cv2.imread(demo)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = clf.predict(rgb)
        probes["demo"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    return {
        "disk_sha": disk,
        "classes": list(clf.classes),
        "is_loaded": clf.is_loaded,
        "probes": probes,
        "metadata": read_json("ml/models/vision/efficientnetv2_metadata.json").get("status"),
        "meta_sha": read_json("ml/models/vision/efficientnetv2_metadata.json").get("artifact_sha256"),
        "meta_classes": read_json("ml/models/vision/efficientnetv2_metadata.json").get("classes"),
    }


def probe_unet():
    from ml.vision.unet_wrapper import UNetSegmenter

    seg = UNetSegmenter()
    disk = sha256_file(UNET_CANONICAL)
    probes = {}
    for name, val in [("gray", 180), ("black", 0), ("white", 255)]:
        img = np.full((128, 128, 3), val, dtype=np.uint8)
        mask, count, ratio, info = seg.segment(img)
        probes[name] = {
            "positive_pixels": int(count),
            "ratio": float(ratio),
            "trust": info.get("trust_status"),
            "message": info.get("display_message"),
            "raw_mean": info.get("raw_output_mean"),
            "raw_max": info.get("raw_output_max"),
        }
    return {
        "disk_sha": disk,
        "is_loaded": seg.is_loaded,
        "probes": probes,
        "meta_status": read_json("ml/models/vision/unet_metadata.json").get("status"),
        "meta_sha": read_json("ml/models/vision/unet_metadata.json").get("artifact_sha256"),
    }


def probe_xgb_vqc():
    from ml.classifiers.xgboost_classifier import XGBoostClassifier
    from ml.classifiers.vqc_classifier import VQCClassifier
    from ml.training.train_xgboost import generate_multimodal_dataset
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    xgb = XGBoostClassifier(XGB_CANONICAL)
    vqc = VQCClassifier(VQC_DIR)
    X, y = generate_multimodal_dataset(num_samples=200, seed=42)
    X_test, y_test = X[170:], y[170:]
    xgb_preds = []
    vqc_preds = []
    vqc_errors = 0
    for i in range(len(X_test)):
        xi, _ = xgb.predict(X_test[i])
        xgb_preds.append(xi)
        try:
            vi, _ = vqc.predict(X_test[i])
            vqc_preds.append(vi)
        except Exception as e:
            vqc_errors += 1
            vqc_preds.append(-1)
    xgb_preds = np.array(xgb_preds)
    vqc_ok = np.array([p for p in vqc_preds if p >= 0])
    y_vqc = y_test[[i for i, p in enumerate(vqc_preds) if p >= 0]]
    # inspect predict source for fake fallback
    import inspect
    from ml.classifiers import vqc_classifier as vc

    src = inspect.getsource(vc.VQCClassifier.predict)
    return {
        "xgb_sha": sha256_file(XGB_CANONICAL),
        "xgb_n_features": int(xgb.model.n_features_in_),
        "xgb_acc": float(accuracy_score(y_test, xgb_preds)),
        "xgb_correct": f"{int((xgb_preds == y_test).sum())} / {len(y_test)}",
        "vqc_sha": sha256_file(VQC_WEIGHTS),
        "vqc_trained": vqc.is_trained,
        "vqc_acc": float(accuracy_score(y_vqc, vqc_ok)) if len(vqc_ok) else None,
        "vqc_correct": f"{int((vqc_ok == y_vqc).sum())} / {len(y_vqc)}" if len(vqc_ok) else None,
        "vqc_errors": vqc_errors,
        "vqc_predict_has_fake_015": "[0.15, 0.70, 0.15]" in src,
        "vqc_predict_has_except_Exception": "except Exception" in src,
        "vqc_macro_f1": float(f1_score(y_vqc, vqc_ok, average="macro", zero_division=0)) if len(vqc_ok) else None,
        "vqc_mcc": float(matthews_corrcoef(y_vqc, vqc_ok)) if len(vqc_ok) else None,
    }


def probe_sensor():
    from ml.classifiers.sensor_classifier import SensorClassifier
    from ml.sensor.sensor_processor import process_sensor_data
    import inspect
    from ml.sensor import sensor_processor as sp

    src = inspect.getsource(sp.process_sensor_data)
    clf = SensorClassifier()
    empty = clf.predict_from_summary({})
    demo = os.path.join("data", "sample", "sensor", "football_fall.csv")
    demo_out = None
    if os.path.exists(demo):
        # process_sensor_data signature may vary — try common patterns
        try:
            demo_out = process_sensor_data(demo)
        except TypeError:
            try:
                demo_out = process_sensor_data(csv_path=demo)
            except Exception as e:
                demo_out = {"error": str(e)}
        except Exception as e:
            demo_out = {"error": str(e)}
    return {
        "sha": sha256_file(SENSOR_MODEL) if os.path.exists(SENSOR_MODEL) else None,
        "is_trained": clf.is_trained,
        "empty": empty,
        "processor_calls_classifier": "predict_from_summary" in src or "SensorClassifier" in src,
        "demo_summary_keys": list(demo_out.keys())[:30] if isinstance(demo_out, dict) else type(demo_out).__name__,
        "demo_motion": (demo_out or {}).get("predicted_motion_class") if isinstance(demo_out, dict) else None,
        "demo_status": (demo_out or {}).get("classifier_status") if isinstance(demo_out, dict) else None,
    }


def probe_twilio():
    import inspect
    from backend.services.twilio_service import TwilioService

    src = inspect.getsource(TwilioService)
    svc = TwilioService()
    configured, msg = svc.is_configured()
    info = svc.get_status_info()
    return {
        "configured": configured,
        "message": msg,
        "status_info": info,
        "has_FROM": "TWILIO_FROM_NUMBER" in src,
        "has_TO": "TWILIO_TO_NUMBER" in src,
        "has_PHONE_alias": "TWILIO_PHONE_NUMBER" in src,
        "has_hardcoded_SMS_SENT": "SMS_SENT" in src,
    }


def probe_api():
    """Prefer live HTTP to a running server; avoid TestClient lifespan hangs."""
    try:
        import urllib.request

        out = {}
        for key, path in (("health", "/api/health"), ("sos", "/api/sos/config"), ("models", "/api/models")):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:8000{path}", timeout=15) as resp:
                    out[key] = json.loads(resp.read().decode("utf-8"))
                    out[f"{key}_status"] = resp.status
            except Exception as e:
                out[key] = {"error": str(e)}
                out[f"{key}_status"] = None
        return out
    except Exception as e:
        return {"error": str(e)}


def five_way():
    reg = read_json(REGISTRY_PATH)
    man = read_json(MANIFEST_PATH)
    man_by = {m["model_name"]: m for m in man.get("models", [])}
    rows = []
    mapping = [
        ("YOLO11 Detection", YOLO_CANONICAL),
        ("EfficientNetV2 Classification", EFFNET_CANONICAL),
        ("ResNet34-UNet Segmentation", UNET_CANONICAL),
        ("XGBoost Multimodal", XGB_CANONICAL),
        ("Experimental 4-Qubit VQC", VQC_WEIGHTS),
        ("Sensor Motion Event Classifier", SENSOR_MODEL),
    ]
    for name, path in mapping:
        disk = sha256_file(path) if os.path.exists(resolve_existing(path)) else None
        r = reg.get(name) or {}
        m = man_by.get(name) or {}
        rows.append(
            {
                "model": name,
                "disk_sha": disk,
                "registry_sha": r.get("artifact_sha256"),
                "manifest_sha": m.get("sha256"),
                "disk_eq_registry": disk == r.get("artifact_sha256"),
                "disk_eq_manifest": disk == m.get("sha256"),
                "registry_status": r.get("status") or r.get("readiness_status"),
                "manifest_status": m.get("readiness_status"),
                "classes_registry": r.get("classes"),
                "classes_manifest": m.get("classes"),
            }
        )
    return rows


def search_swelling_frontend():
    hits = []
    for root, _, files in os.walk("frontend"):
        for f in files:
            if f.endswith((".ts", ".tsx", ".js", ".jsx")):
                p = os.path.join(root, f)
                try:
                    text = open(p, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                if "Swelling" in text or "swelling" in text:
                    hits.append(p.replace("\\", "/"))
    return hits


def main():
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "five_way": five_way(),
        "yolo": None,
        "effnet": None,
        "unet": None,
        "xgb_vqc": None,
        "sensor": None,
        "twilio": None,
        "api": None,
        "frontend_swelling_files": search_swelling_frontend(),
    }
    print("YOLO...")
    report["yolo"] = probe_yolo()
    print("EFFNET...")
    report["effnet"] = probe_effnet()
    print("UNET...")
    report["unet"] = probe_unet()
    print("XGB/VQC...")
    report["xgb_vqc"] = probe_xgb_vqc()
    print("SENSOR...")
    report["sensor"] = probe_sensor()
    print("TWILIO...")
    report["twilio"] = probe_twilio()
    print("API...")
    report["api"] = probe_api()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print("WROTE", OUT)
    # concise stdout
    print("FIVE_WAY", json.dumps(report["five_way"], indent=2))
    print("YOLO names", report["yolo"]["names"], "status", report["yolo"]["status"])
    print("EFFNET classes", report["effnet"]["classes"])
    print("EFFNET gray", report["effnet"]["probes"].get("gray"))
    print("UNET gray", report["unet"]["probes"].get("gray"))
    print("XGB", report["xgb_vqc"]["xgb_correct"], "VQC", report["xgb_vqc"]["vqc_correct"])
    print("SENSOR wired", report["sensor"]["processor_calls_classifier"], "demo", report["sensor"]["demo_motion"])
    print("TWILIO", report["twilio"]["configured"], report["twilio"]["message"])
    print("SWELLING FE files", len(report["frontend_swelling_files"]))


if __name__ == "__main__":
    main()
