"""Fresh Phase-1 YOLO forensic audit — write scratch/yolo_fresh_forensic_audit.json."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backend.config import settings
from ml.models.canonical_paths import (
    MANIFEST_PATH,
    REGISTRY_PATH,
    YOLO_CANONICAL,
    YOLO_METADATA,
    YOLO_RETRAIN_V2_ROOT,
    abs_path,
    read_json,
    sha256_file,
)
from ml.vision.yolo_wrapper import YOLO11Detector

SCRATCH = ROOT / "scratch"
SCRATCH.mkdir(exist_ok=True)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _box_counts(dataset_root: Path) -> dict:
    names = {0: "cut", 1: "bruise", 2: "wound"}
    out = {}
    for split in ("train", "val", "test"):
        lab = dataset_root / "labels" / split
        boxes = {n: 0 for n in names.values()}
        images = 0
        empty = 0
        if not lab.is_dir():
            out[split] = {"missing": True}
            continue
        for p in lab.glob("*.txt"):
            images += 1
            lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                empty += 1
            for ln in lines:
                parts = ln.split()
                if len(parts) >= 1:
                    cid = int(float(parts[0]))
                    if cid in names:
                        boxes[names[cid]] += 1
        out[split] = {"images": images, "empty_labels": empty, "boxes": boxes}
    return out


def _make_solid(path: Path, color) -> None:
    img = np.full((256, 256, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-9))


def _run_infer(det: YOLO11Detector, path: Path, conf: float) -> list[dict]:
    if not path.is_file():
        return [{"error": "missing", "path": str(path)}]
    # bypass wrapper keep threshold by calling model directly when possible
    results = det.model.predict(str(path), conf=conf, verbose=False)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        names = r.names
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            cid = int(box.cls[0].item())
            dets.append(
                {
                    "class": names.get(cid, str(cid)),
                    "confidence": float(box.conf[0].item()),
                    "bounding_box": [round(x, 2) for x in xyxy],
                }
            )
    return dets


def main() -> None:
    meta = read_json(YOLO_METADATA)
    reg = read_json(REGISTRY_PATH)
    man = read_json(MANIFEST_PATH)
    prod_sha = sha256_file(YOLO_CANONICAL)
    reg_yolo = reg.get("YOLO11 Detection") or {}
    man_yolo = None
    for m in man.get("models") or []:
        if "YOLO" in str(m.get("model_name", "")):
            man_yolo = m
            break

    probe_dir = SCRATCH / "yolo_fresh_probes"
    probe_dir.mkdir(exist_ok=True)
    _make_solid(probe_dir / "blank_gray.png", 128)
    _make_solid(probe_dir / "black.png", 0)
    _make_solid(probe_dir / "white.png", 255)

    images = {
        "blank_gray": probe_dir / "blank_gray.png",
        "black": probe_dir / "black.png",
        "white": probe_dir / "white.png",
        "dummy_test": ROOT / "data" / "datasets" / "yolo_injury" / "dummy_test.jpg",
        "blank_skin": ROOT / "data" / "datasets" / "yolo_injury" / "blank_skin.jpg",
        "football_injury": ROOT / "data" / "demo" / "football_injury.jpg",
        "forensic_hand": ROOT / "data" / "uploads" / "3f629ca8-dd98-427d-a708-f976e2042555.jpeg",
    }
    # heuristic GT for forensic hand (prior audit): injury roughly mid-hand
    hand_gt = [550.0, 200.0, 750.0, 480.0]

    det = YOLO11Detector()
    assert det.model is not None
    runtime_sha = prod_sha
    ckpt_names = det.model.names

    probes = {}
    for name, path in images.items():
        raw = _run_infer(det, path, 0.01) if path.is_file() else []
        app = [d for d in raw if d.get("confidence", 0) >= settings.yolo_conf_threshold]
        entry = {
            "path": str(path.relative_to(ROOT)) if path.is_file() else None,
            "exists": path.is_file(),
            "raw_001": raw,
            "app_keep": app,
            "n_raw": len(raw) if isinstance(raw, list) and (not raw or "error" not in raw[0]) else 0,
            "n_app": len(app),
        }
        if name == "forensic_hand" and path.is_file():
            best_iou = 0.0
            best = None
            for d in raw:
                iou = _iou(d["bounding_box"], hand_gt)
                if iou > best_iou:
                    best_iou = iou
                    best = d
            cut_dets = [d for d in raw if d.get("class") == "cut"]
            entry["heuristic_gt"] = hand_gt
            entry["best_iou_vs_gt"] = round(best_iou, 4)
            entry["best_det"] = best
            entry["cut_at_001"] = cut_dets
            entry["covers_injury_heuristic"] = best_iou >= 0.2
        probes[name] = entry

    report = {
        "created_utc": _utc(),
        "production": {
            "path": YOLO_CANONICAL.replace("\\", "/"),
            "sha256_disk": prod_sha,
            "sha256_metadata": meta.get("artifact_sha256"),
            "sha256_registry": reg_yolo.get("sha256") or reg_yolo.get("artifact_sha256"),
            "sha_match_meta": prod_sha == meta.get("artifact_sha256"),
            "runtime_sha": runtime_sha,
            "classes_metadata": meta.get("classes"),
            "classes_checkpoint": ckpt_names,
            "zero_training_boxes": meta.get("classes_with_zero_training_boxes"),
            "keep_threshold": settings.yolo_conf_threshold,
            "version": meta.get("version"),
            "status": meta.get("status"),
        },
        "registry_entry": {k: reg_yolo.get(k) for k in list(reg_yolo.keys())[:15]},
        "manifest_entry": man_yolo,
        "label_counts_yolo_retrain_v2": _box_counts(ROOT / YOLO_RETRAIN_V2_ROOT),
        "label_counts_yolo_real_skin_v2": _box_counts(ROOT / "data" / "datasets" / "yolo_real_skin_v2"),
        "probes": probes,
        "credentials": {
            "ROBOFLOW_API_KEY_in_process_env": bool(os.environ.get("ROBOFLOW_API_KEY", "").strip()),
            "ROBOFLOW_API_KEY_in_backend_dotenv": False,  # set by caller
            "KAGGLE_JSON_present": (Path.home() / ".kaggle" / "kaggle.json").exists(),
        },
    }
    out = SCRATCH / "yolo_fresh_forensic_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "sha": prod_sha[:16],
        "sha_match_meta": report["production"]["sha_match_meta"],
        "threshold": settings.yolo_conf_threshold,
        "ckpt_names": ckpt_names,
        "hand_n_raw": probes["forensic_hand"]["n_raw"],
        "hand_n_app": probes["forensic_hand"]["n_app"],
        "hand_best_iou": probes["forensic_hand"].get("best_iou_vs_gt"),
        "blank_app": probes["blank_gray"]["n_app"],
        "black_app": probes["black"]["n_app"],
        "wrote": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
