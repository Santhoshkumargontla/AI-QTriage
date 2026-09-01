"""Phase 0-1 YOLO baseline verification + forensic hand case audit."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

CANONICAL = "ml/models/vision/yolo11_injury_best.pt"
EXPECTED_SHA = "4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879"
HAND_CASE = "3f629ca8-dd98-427d-a708-f976e2042555"
HAND_IMG = ROOT / "data/uploads" / f"{HAND_CASE}.jpeg"
THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    from ml.models.canonical_paths import abs_path, read_json, MANIFEST_PATH, REGISTRY_PATH
    from ml.vision.yolo_wrapper import YOLO11Detector, DEFAULT_YOLO_INFER_CONF

    out = {"phase": "0-1 baseline and forensic verify"}

    disk_sha = sha256_file(Path(abs_path(CANONICAL)))
    det = YOLO11Detector()
    info = det.get_info()
    reg = read_json(REGISTRY_PATH).get("YOLO11 Detection", {})
    man = read_json(MANIFEST_PATH)
    man_yolo = next((m for m in (man.get("models") or []) if "YOLO" in m.get("model_name", "")), {})

    # Class support from yolo_retrain_v2 labels
    box_support = {"train": {}, "val": {}, "test": {}}
    for split in box_support:
        lab_dir = ROOT / "data/datasets/yolo_retrain_v2/labels" / split
        if not lab_dir.is_dir():
            continue
        from collections import Counter
        c = Counter()
        for lf in lab_dir.glob("*.txt"):
            for ln in lf.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    cid = int(float(ln.split()[0]))
                    name = {0: "cut", 1: "bruise", 2: "wound"}.get(cid, str(cid))
                    c[name] += 1
        box_support[split] = dict(c)

    out["baseline"] = {
        "canonical_path": CANONICAL,
        "disk_sha256": disk_sha,
        "expected_sha256": EXPECTED_SHA,
        "sha_match": disk_sha == EXPECTED_SHA,
        "wrapper_sha256": info.get("artifact_sha256"),
        "registry_sha256": reg.get("artifact_sha256"),
        "manifest_sha256": man_yolo.get("sha256"),
        "model_names": info.get("classes"),
        "keep_threshold": DEFAULT_YOLO_INFER_CONF,
        "class_support_yolo_retrain_v2": box_support,
        "wound_honest_support": box_support["train"].get("wound", 0) == 0,
    }

    # Hand forensic
    hand = {"path": str(HAND_IMG), "exists": HAND_IMG.is_file()}
    if HAND_IMG.is_file():
        hand["sha256"] = sha256_file(HAND_IMG)
        bgr = cv2.imread(str(HAND_IMG))
        hand["dimensions_hw"] = list(bgr.shape[:2])
        hand["threshold_sweep"] = {}
        for thr in THRESHOLDS:
            # raw ultralytics at multiple conf
            from ultralytics import YOLO
            model = YOLO(abs_path(CANONICAL))
            res = model.predict(str(HAND_IMG), conf=thr, verbose=False)
            dets = []
            if res and res[0].boxes is not None:
                names = res[0].names
                for box in res[0].boxes:
                    cid = int(box.cls[0].item())
                    cname = names.get(cid, str(cid))
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    dets.append({
                        "class": str(cname).lower(),
                        "confidence": round(float(box.conf[0].item()), 4),
                        "bounding_box": [round(float(v), 2) for v in xyxy],
                    })
            hand["threshold_sweep"][str(thr)] = dets
        wrapper_dets = det.detect(str(HAND_IMG))
        hand["wrapper_at_default_threshold"] = wrapper_dets

    try:
        import requests
        api = requests.get(f"http://127.0.0.1:8000/api/cases/{HAND_CASE}", timeout=30).json()
        vi = api.get("visible_injury") or {}
        hand["api"] = {
            "yolo_finding": vi.get("yolo_finding"),
            "yolo_confidence": vi.get("yolo_confidence"),
            "bounding_box": vi.get("bounding_box"),
            "original_width": vi.get("original_width"),
            "original_height": vi.get("original_height"),
        }
        hand["api_matches_direct"] = (
            vi.get("bounding_box") == (wrapper_dets[0]["bounding_box"] if wrapper_dets else None)
        )
    except Exception as exc:
        hand["api_error"] = str(exc)

    out["forensic_hand"] = hand
    dest = ROOT / "scratch" / "yolo_phase0_1_verify.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
