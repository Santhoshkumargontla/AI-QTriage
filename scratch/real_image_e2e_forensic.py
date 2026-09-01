"""End-to-end forensic on the newest unique upload (hand injury candidate)."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ml.models.canonical_paths import (
    YOLO_CANONICAL,
    sha256_file,
    abs_path,
    exists,
)
from ml.vision.yolo_wrapper import YOLO11Detector
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier, interpret_prediction
from ml.vision.unet_wrapper import UNetSegmenter, interpret_segmentation
from ml.explainability.grad_cam import maybe_generate_gradcam

OUT = abs_path("scratch/real_image_e2e_forensic.json")
UPLOAD = abs_path("data/uploads")
CASE_ID = "ac69884f-7a50-48e5-b7cc-a9bea9b20313"
IMG = os.path.join(UPLOAD, f"{CASE_ID}.jpeg")
BACKUP = abs_path("ml/models/vision/yolo11_injury_best.pt.pre_retrain_v2_backup")


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_sha(path: str) -> str:
    bgr = cv2.imread(path)
    if bgr is None:
        return ""
    return hashlib.sha256(bgr.tobytes() + f"|{bgr.shape[1]}x{bgr.shape[0]}".encode()).hexdigest()


def main():
    assert os.path.exists(IMG), IMG
    bgr = cv2.imread(IMG)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    identity = {
        "case_id": CASE_ID,
        "path": IMG,
        "file_sha256": file_sha(IMG),
        "pixel_sha256": pixel_sha(IMG),
        "dimensions": [h, w],
        "nbytes": os.path.getsize(IMG),
        "mean": float(rgb.mean()),
        "std": float(rgb.std()),
    }

    det = YOLO11Detector()
    info = det.get_info()
    active_sha = info.get("artifact_sha256") or sha256_file(YOLO_CANONICAL)
    names = list(info.get("classes") or [])

    sweeps = {}
    for conf in (0.01, 0.05, 0.10, 0.25, 0.40, 0.50):
        boxes = det.detect(IMG, conf=conf)
        sweeps[str(conf)] = [
            {
                "finding": b.get("finding"),
                "confidence": float(b.get("confidence")),
                "bbox": b.get("bounding_box"),
                "class_id": b.get("class_id"),
            }
            for b in boxes
        ]

    # ultralytics direct for class ids
    from ultralytics import YOLO

    model = YOLO(abs_path(YOLO_CANONICAL))
    raw_ultra = {}
    for conf in (0.01, 0.25):
        res = model.predict(IMG, conf=conf, verbose=False)[0]
        rows = []
        if res.boxes is not None and len(res.boxes):
            for box in res.boxes:
                cls_id = int(box.cls.item())
                rows.append(
                    {
                        "cls_id": cls_id,
                        "name": model.names.get(cls_id) or model.names[cls_id],
                        "conf": float(box.conf.item()),
                        "xyxy": [float(x) for x in box.xyxy.cpu().numpy().reshape(-1).tolist()],
                    }
                )
        raw_ultra[str(conf)] = {"names": dict(model.names), "dets": rows}

    backup_compare = None
    if exists(BACKUP):
        import shutil
        import tempfile

        tmp_pt = os.path.join(tempfile.gettempdir(), "yolo_backup_compare.pt")
        shutil.copy2(BACKUP, tmp_pt)
        bmodel = YOLO(tmp_pt)
        backup_compare = {
            "sha": sha256_file(BACKUP),
            "names": dict(bmodel.names),
            "dets_0.25": [],
            "dets_0.01": [],
        }
        for conf_key, conf_val in (("dets_0.25", 0.25), ("dets_0.01", 0.01)):
            bres = bmodel.predict(IMG, conf=conf_val, verbose=False)[0]
            if bres.boxes is not None and len(bres.boxes):
                for box in bres.boxes:
                    cls_id = int(box.cls.item())
                    backup_compare[conf_key].append(
                        {
                            "cls_id": cls_id,
                            "name": bmodel.names.get(cls_id) or bmodel.names[cls_id],
                            "conf": float(box.conf.item()),
                            "xyxy": [float(x) for x in box.xyxy.cpu().numpy().reshape(-1).tolist()],
                        }
                    )

    # draw debug overlay for active 0.25
    debug = rgb.copy()
    for b in sweeps.get("0.25") or []:
        x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug,
            f"{b['finding']} {b['confidence']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    debug_path = abs_path(f"scratch/{CASE_ID}_yolo_debug.jpg")
    cv2.imwrite(debug_path, cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))

    # EffNet / U-Net
    clf = EfficientNetV2Classifier()
    unet = UNetSegmenter()
    raw_e = clf.predict_raw(rgb)
    gate_e = interpret_prediction(clf.predict(rgb))
    ov, cam = maybe_generate_gradcam(clf, rgb, gate_e)

    # U-Net full + with best yolo bbox if any
    mask, pc, ar, uinfo = unet.segment(rgb)
    uparsed = interpret_segmentation(mask, pc, ar, uinfo)
    unet_roi = None
    if sweeps.get("0.25"):
        bbox = sweeps["0.25"][0]["bbox"]
        m2, pc2, ar2, i2 = unet.segment(rgb, bbox)
        unet_roi = {
            "bbox": bbox,
            "raw_pos": i2.get("raw_positive_ratio"),
            "reliable": interpret_segmentation(m2, pc2, ar2, i2).get("is_reliable"),
            "withheld": interpret_segmentation(m2, pc2, ar2, i2).get("mask_withheld"),
            "status": i2.get("status"),
            "pos_px": int(np.asarray(m2).sum()) if m2 is not None else 0,
        }

    # Mongo case
    mongo_case = None
    try:
        from backend.database import get_database

        db = get_database()
        doc = db.cases.find_one({"case_id": CASE_ID})
        if doc:
            vi = doc.get("visible_injury") or {}
            mongo_case = {
                "status": doc.get("status"),
                "image_reference": doc.get("image_reference"),
                "yolo_finding": vi.get("yolo_finding"),
                "yolo_confidence": vi.get("yolo_confidence") or vi.get("confidence"),
                "yolo_bbox": vi.get("yolo_bounding_box") or vi.get("bounding_box"),
                "classifier_finding": vi.get("classifier_finding"),
                "classifier_status": vi.get("classifier_status"),
                "classifier_model_status": vi.get("classifier_model_status"),
                "classifier_is_confident": vi.get("classifier_is_confident"),
                "segmentation_reliable": vi.get("segmentation_reliable"),
                "gradcam_overlay_generated": vi.get("gradcam_overlay_generated"),
                "xgb": (doc.get("xgboost_prediction") or {}).get("class"),
                "vqc": (doc.get("quantum_prediction") or {}).get("class"),
                "clinical_claim_blocked": doc.get("clinical_claim_blocked"),
            }
            # hash image_reference if exists
            ref = doc.get("image_reference")
            if ref and os.path.exists(ref):
                mongo_case["image_reference_file_sha"] = file_sha(ref)
                mongo_case["image_reference_match"] = file_sha(ref) == identity["file_sha256"]
            elif ref:
                # maybe relative
                cand = abs_path(ref) if not os.path.isabs(ref) else ref
                if os.path.exists(cand):
                    mongo_case["image_reference_file_sha"] = file_sha(cand)
                    mongo_case["image_reference_match"] = file_sha(cand) == identity["file_sha256"]
    except Exception as e:
        mongo_case = {"error": str(e)}

    # class support / mapping
    support = None
    sp = abs_path("ml/models/vision/yolo11_class_support.json")
    if os.path.exists(sp):
        support = json.load(open(sp, encoding="utf-8"))

    payload = {
        "identity": identity,
        "yolo_active": {
            "sha": active_sha,
            "names": names,
            "wrapper_path": info.get("model_path") or info.get("canonical_path"),
            "sweeps": sweeps,
            "ultralytics": raw_ultra,
        },
        "backup_compare": backup_compare,
        "debug_overlay": debug_path,
        "efficientnet": {
            "raw": {"winner": raw_e.get("winner"), "max": raw_e.get("max_prob"), "probs": raw_e.get("probs")},
            "gated": gate_e,
            "gradcam": {"overlay": ov is not None, **{k: cam.get(k) for k in ("explanation_status", "withheld_reason", "model_status")}},
        },
        "unet_full": {
            "raw_pos": uinfo.get("raw_positive_ratio"),
            "raw_mean": uinfo.get("raw_output_mean"),
            "status": uinfo.get("status"),
            "withheld": uparsed.get("mask_withheld"),
            "reliable": uparsed.get("is_reliable"),
            "px": uparsed.get("pixel_count"),
        },
        "unet_roi": unet_roi,
        "mongo_case": mongo_case,
        "class_support": support,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print("WROTE", OUT)
    print("IDENTITY", identity["file_sha256"][:16], identity["dimensions"])
    print("ACTIVE_SHA", active_sha[:16], "names", names)
    print("SWEEP_0.25", sweeps.get("0.25"))
    print("SWEEP_0.01", sweeps.get("0.01")[:5] if sweeps.get("0.01") else None)
    print("BACKUP_0.25", None if not backup_compare else backup_compare["dets_0.25"])
    print("EFFNET", gate_e.get("status"), gate_e.get("winner"), "raw", raw_e.get("winner"), raw_e.get("max_prob"))
    print("UNET", uparsed.get("is_reliable"), uinfo.get("raw_positive_ratio"))
    print("MONGO", mongo_case)


if __name__ == "__main__":
    main()
