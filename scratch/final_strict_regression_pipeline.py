"""Final regression pipeline: direct wrappers -> live API -> MongoDB.

Leaves the primary case in MongoDB for browser verification.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

API = os.environ.get("AIQTRIAGE_API", "http://127.0.0.1:8000")
IMG = os.path.join(ROOT, "data", "sample", "image", "football_injury.jpg")
OUT_PATH = os.path.join(ROOT, "scratch", "final_strict_regression_pipeline.json")

from backend.database.connection import get_database
from ml.classifiers.sensor_classifier import SensorClassifier
from ml.classifiers.vqc_classifier import VQCClassifier
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.fusion.feature_fusion import MultimodalFeatureFusion
from ml.models.canonical_paths import VQC_DIR, XGB_CANONICAL
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
from ml.vision.unet_wrapper import UNetSegmenter
from ml.vision.yolo_wrapper import YOLO11Detector

LABEL = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
ANSWERS = {
    "pain_level": "6",
    "cause": "fall",
    "bleeding": "mild",
    "movement": "limited",
    "limb_use": "with_pain",
    "location": "left ankle",
}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj


def nearly_equal(a, b, tol=1e-5):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def lists_close(a, b, tol=1e-5):
    if a is None and b is None:
        return True
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return False
    return all(nearly_equal(x, y, tol) for x, y in zip(a, b))


out = {
    "started": datetime.now(timezone.utc).isoformat(),
    "api_base": API,
    "image": IMG,
}

# --- Health / registry / evaluation / SOS ---
sess = requests.Session()
for name, path in [
    ("health", "/api/health"),
    ("models", "/api/models"),
    ("registry", "/api/models/registry"),
    ("evaluation", "/api/evaluation"),
    ("sos_config", "/api/sos/config"),
]:
    r = sess.get(API + path, timeout=120)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:400]
    out[name] = {"http": r.status_code, "body": body}
    print(name, r.status_code)

img_bgr = cv2.imread(IMG)
if img_bgr is None:
    raise SystemExit(f"cannot read {IMG}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --- Direct YOLO ---
yolo = YOLO11Detector()
direct_yolo = yolo.detect(IMG)
out["direct_yolo"] = {
    "model_path": yolo.model_path,
    "status": yolo.status if hasattr(yolo, "status") else None,
    "n": len(direct_yolo),
    "findings": direct_yolo,
}
print("DIRECT YOLO", len(direct_yolo), direct_yolo)

best_bbox = None
if direct_yolo:
    best = max(direct_yolo, key=lambda d: float(d.get("confidence") or 0))
    best_bbox = best.get("bounding_box")

# --- Direct EfficientNet (full image + YOLO ROI if present) ---
eff = EfficientNetV2Classifier()
full_pred = eff.predict(img_rgb)
full_raw = eff.predict_raw(img_rgb)
roi_pred = None
if best_bbox and len(best_bbox) == 4:
    x1, y1, x2, y2 = [int(v) for v in best_bbox]
    h, w = img_rgb.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 > x1 and y2 > y1:
        roi_pred = eff.predict(img_rgb[y1:y2, x1:x2])
out["direct_efficientnet"] = {
    "model_path": eff.model_path,
    "is_loaded": eff.is_loaded,
    "classes": list(eff.classes),
    "full_gated": {k: v for k, v in full_pred.items() if k.startswith("__") or k in ("Cut", "Bruise", "Swelling", "Other")},
    "full_raw": full_raw,
    "roi_gated": roi_pred,
}
print("DIRECT EFFNET full winner", full_pred.get("__winner"), "raw", full_raw.get("winner"), full_raw.get("max_prob"))

# --- Direct U-Net ---
unet = UNetSegmenter()
seg_full = unet.segment_raw(img_rgb)
seg_roi = unet.segment_raw(img_rgb, bbox=best_bbox) if best_bbox else None
gated_full = unet.segment(img_rgb)
gated_roi = unet.segment(img_rgb, bbox=best_bbox) if best_bbox else None


def _seg_summary(raw, gated):
    info = {}
    if isinstance(raw, dict):
        info["raw_positive_ratio"] = raw.get("positive_ratio")
        info["raw_mean_prob"] = raw.get("mean_prob")
        info["raw_max_prob"] = raw.get("max_prob")
        info["raw_threshold"] = raw.get("threshold")
    # segment() returns mask, pixel_count, affected_ratio, debug_info
    if isinstance(gated, tuple) and len(gated) >= 4:
        mask, pixel_count, affected_ratio, meta = gated[0], gated[1], gated[2], gated[3]
        info["gated_mask_sum"] = int(np.asarray(mask).sum()) if mask is not None else None
        info["gated_pixel_count"] = int(pixel_count) if pixel_count is not None else None
        info["gated_affected_ratio"] = float(affected_ratio) if affected_ratio is not None else None
        if isinstance(meta, dict):
            info["gated_status"] = meta.get("status")
            info["gated_reason"] = meta.get("reason") or meta.get("display_message")
    return info


out["direct_unet"] = {
    "model_path": unet.model_path,
    "is_loaded": unet.is_loaded,
    "full": _seg_summary(seg_full, gated_full),
    "roi": _seg_summary(seg_roi, gated_roi) if best_bbox else None,
}
print("DIRECT UNET full", out["direct_unet"]["full"])

# --- Live API case: image + questionnaire + skip sensor + analyze ---
cr = sess.post(API + "/api/cases", json={"notes": "final_strict_regression_pipeline"}, timeout=60)
cr.raise_for_status()
case_id = cr.json()["case_id"]
out["case_id"] = case_id
print("case_id", case_id)

with open(IMG, "rb") as f:
    ur = sess.post(
        API + f"/api/cases/{case_id}/image",
        files={"file": ("football_injury.jpg", f, "image/jpeg")},
        timeout=60,
    )
out["upload_image"] = {"http": ur.status_code}

qr = sess.post(
    API + f"/api/cases/{case_id}/questionnaire",
    json={"answers": ANSWERS, "answer_source": "typed"},
    timeout=60,
)
out["questionnaire"] = {"http": qr.status_code, "body": qr.json() if qr.status_code == 200 else qr.text[:400]}

sr = sess.post(API + f"/api/cases/{case_id}/sensor/skip", timeout=60)
out["sensor_skip"] = {"http": sr.status_code}

ar = sess.post(API + f"/api/cases/{case_id}/analyze", timeout=300)
out["analyze"] = {"http": ar.status_code, "body": ar.json() if ar.status_code == 200 else ar.text[:800]}
print("analyze", ar.status_code)

gr = sess.get(API + f"/api/cases/{case_id}", timeout=60)
api_case = gr.json()
out["api_get_http"] = gr.status_code
vi = api_case.get("visible_injury") or {}
xgb_api = api_case.get("xgboost_prediction") or {}
vqc_api = api_case.get("quantum_prediction") or {}

# Reconstruct fusion vector the same way analyze does, then direct XGB/VQC
fusion = MultimodalFeatureFusion()
case_data = {
    "vision_analysis": {
        "classification": vi.get("classification"),
        "segmentation": {"affected_ratio": vi.get("affected_ratio", 0.0)},
    } if vi else {},
    "questionnaire": api_case.get("questionnaire") or {},
    "sensor_summary": {},
}
_, vector, names = fusion.fuse_features(case_data)
xgb = XGBoostClassifier(XGB_CANONICAL)
pred_xgb, probs_xgb = xgb.predict(vector)
vqc = VQCClassifier(VQC_DIR)
pred_vqc, scores_vqc = vqc.predict(vector)

out["direct_xgboost"] = {
    "class_idx": int(pred_xgb),
    "class": LABEL[int(pred_xgb)],
    "probs": list(probs_xgb),
    "vector": vector.tolist(),
    "feature_names": names,
}
out["direct_vqc"] = {
    "class_idx": int(pred_vqc),
    "class": LABEL[int(pred_vqc)],
    "probs": list(scores_vqc),
    "status": getattr(vqc, "status", None),
}

db = get_database()
stored = db.cases.find_one({"case_id": case_id}, {"_id": 0})
vi_m = (stored or {}).get("visible_injury") or {}
xgb_m = (stored or {}).get("xgboost_prediction") or {}
vqc_m = (stored or {}).get("quantum_prediction") or {}

out["layer_compare"] = {
    "yolo": {
        "direct_n": len(direct_yolo),
        "direct_class": [d.get("finding") for d in direct_yolo],
        "direct_conf": [d.get("confidence") for d in direct_yolo],
        "direct_bbox": [d.get("bounding_box") for d in direct_yolo],
        "api_finding": vi.get("yolo_finding") or vi.get("finding"),
        "api_detected": vi.get("yolo_finding_detected", vi.get("finding_detected")),
        "api_conf": vi.get("yolo_confidence") or vi.get("confidence"),
        "api_bbox": vi.get("yolo_bounding_box") or vi.get("bounding_box"),
        "mongo_finding": vi_m.get("yolo_finding") or vi_m.get("finding"),
        "mongo_conf": vi_m.get("yolo_confidence") or vi_m.get("confidence"),
        "mongo_bbox": vi_m.get("yolo_bounding_box") or vi_m.get("bounding_box"),
        "direct_eq_api_bbox": lists_close(
            (direct_yolo[0].get("bounding_box") if direct_yolo else None),
            (vi.get("yolo_bounding_box") or vi.get("bounding_box")),
            1e-3,
        ) if (len(direct_yolo) == 1) else None,
        "api_eq_mongo_conf": nearly_equal(vi.get("yolo_confidence") or vi.get("confidence"), vi_m.get("yolo_confidence") or vi_m.get("confidence")),
    },
    "efficientnet": {
        "direct_full_winner": full_pred.get("__winner"),
        "direct_full_raw_winner": full_raw.get("winner"),
        "direct_full_raw_max": full_raw.get("max_prob"),
        "direct_roi_winner": (roi_pred or {}).get("__winner"),
        "direct_roi_raw_winner": (roi_pred or {}).get("__raw_winner"),
        "api_classifier_finding": vi.get("classifier_finding"),
        "api_classifier_probability": vi.get("classifier_probability"),
        "api_classification": vi.get("classification"),
        "mongo_classifier_finding": vi_m.get("classifier_finding"),
        "mongo_classification": vi_m.get("classification"),
        "api_eq_mongo": vi.get("classifier_finding") == vi_m.get("classifier_finding"),
    },
    "unet": {
        "direct_full": out["direct_unet"]["full"],
        "direct_roi": out["direct_unet"]["roi"],
        "api_affected_ratio": vi.get("affected_ratio"),
        "api_segmentation_status": vi.get("segmentation_status") or vi.get("segmentation_trust"),
        "api_segmentation_reliable": vi.get("segmentation_reliable"),
        "api_mask_url": vi.get("mask_url"),
        "mongo_affected_ratio": vi_m.get("affected_ratio"),
        "mongo_segmentation_status": vi_m.get("segmentation_status") or vi_m.get("segmentation_trust"),
        "api_eq_mongo_ratio": nearly_equal(vi.get("affected_ratio"), vi_m.get("affected_ratio")),
    },
    "xgboost": {
        "direct_class": LABEL[int(pred_xgb)],
        "direct_probs": list(probs_xgb),
        "api_class": xgb_api.get("class"),
        "api_score": xgb_api.get("score"),
        "mongo_class": xgb_m.get("class"),
        "mongo_score": xgb_m.get("score"),
        "direct_eq_api_class": LABEL[int(pred_xgb)] == xgb_api.get("class"),
        "direct_eq_api_probs": lists_close(list(probs_xgb), xgb_api.get("score")),
        "api_eq_mongo": xgb_api.get("class") == xgb_m.get("class") and lists_close(xgb_api.get("score"), xgb_m.get("score")),
    },
    "vqc": {
        "direct_class": LABEL[int(pred_vqc)],
        "direct_probs": list(scores_vqc),
        "api_class": vqc_api.get("class"),
        "api_score": vqc_api.get("score"),
        "api_status": vqc_api.get("status"),
        "mongo_class": vqc_m.get("class"),
        "mongo_score": vqc_m.get("score"),
        "direct_eq_api_class": LABEL[int(pred_vqc)] == vqc_api.get("class"),
        "direct_eq_api_probs": lists_close(list(scores_vqc), vqc_api.get("score")),
        "api_eq_mongo": vqc_api.get("class") == vqc_m.get("class") and lists_close(vqc_api.get("score"), vqc_m.get("score")),
        "used_in_main_decision": vqc_api.get("used_in_main_decision"),
    },
}

print("LAYER COMPARE", json.dumps(out["layer_compare"], indent=2, default=str))

# --- Sensor simulate case (keep separate; delete after mongo compare) ---
sensor_out = {}
clf = SensorClassifier()
for scenario in ("normal_movement", "football_fall", "sudden_impact"):
    rc = sess.post(API + "/api/cases", json={"notes": f"final_reg_sensor_{scenario}"}, timeout=60)
    sid = rc.json()["case_id"]
    sim = sess.post(API + f"/api/cases/{sid}/sensor/simulate", json={"scenario": scenario}, timeout=60)
    rec = {"http": sim.status_code, "case_id": sid}
    if sim.status_code == 200:
        body = sim.json()
        s = body.get("summary") or {}
        rec["api_summary"] = {
            "peak_g_force": s.get("peak_g_force"),
            "predicted_motion_class": s.get("predicted_motion_class"),
            "motion_confidence": s.get("motion_confidence"),
            "classifier_status": s.get("classifier_status"),
            "source_type": s.get("source_type"),
            "sos_triggered": body.get("sos_triggered"),
        }
        direct_s = clf.predict_from_summary(s)
        rec["direct_sensor"] = {
            "predicted_motion_class": direct_s.get("predicted_motion_class"),
            "confidence": direct_s.get("confidence"),
            "status": direct_s.get("status") or direct_s.get("classifier_status"),
            "probabilities": direct_s.get("probabilities"),
        }
        mongo = db.cases.find_one({"case_id": sid}, {"_id": 0, "sensor_summary": 1})
        ms = (mongo or {}).get("sensor_summary") or {}
        rec["mongo"] = {
            "predicted_motion_class": ms.get("predicted_motion_class"),
            "classifier_status": ms.get("classifier_status"),
            "peak_g_force": ms.get("peak_g_force"),
        }
        rec["direct_eq_api"] = direct_s.get("predicted_motion_class") == s.get("predicted_motion_class")
        rec["api_eq_mongo"] = ms.get("predicted_motion_class") == s.get("predicted_motion_class")
    else:
        rec["error"] = sim.text[:400]
    sensor_out[scenario] = rec
    print("sensor", scenario, rec.get("direct_eq_api"), rec.get("api_eq_mongo"), rec.get("http"))
    db.cases.delete_one({"case_id": sid})
    db.sos_events.delete_many({"case_id": sid})
out["sensor"] = sensor_out

# --- SOS on primary case ---
sos_local = sess.post(
    API + f"/api/cases/{case_id}/sos/demo/trigger",
    json={"mode": "local_demo"},
    timeout=60,
)
sos_twilio = sess.post(
    API + f"/api/cases/{case_id}/sos/demo/trigger",
    json={"mode": "twilio_test"},
    timeout=60,
)
out["sos"] = {
    "config": out["sos_config"]["body"],
    "local_demo": {"http": sos_local.status_code, "body": sos_local.json() if sos_local.status_code < 500 else sos_local.text[:400]},
    "twilio_test": {"http": sos_twilio.status_code, "body": sos_twilio.json() if sos_twilio.status_code < 500 else sos_twilio.text[:400]},
}
print("SOS local", sos_local.status_code, "twilio", sos_twilio.status_code)

# Do not delete primary case — browser verification uses it.
out["keep_case_for_browser"] = case_id
out["finished"] = datetime.now(timezone.utc).isoformat()

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(_jsonable(out), f, indent=2, default=str)
print("wrote", OUT_PATH)
print("BROWSER_CASE", case_id)
