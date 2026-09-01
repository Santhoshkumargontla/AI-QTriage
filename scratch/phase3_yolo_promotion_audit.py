"""Phase 3 verify-first: YOLO candidate vs previous checkpoint, negative-image audit."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from ml.models.canonical_paths import (  # noqa: E402
    MANIFEST_PATH,
    REGISTRY_PATH,
    YOLO_CANONICAL,
    YOLO_RETRAIN_V2_BEST,
    abs_path,
    posix,
    read_json,
    sha256_file,
)
from ml.vision.yolo_wrapper import DEFAULT_YOLO_INFER_CONF, YOLO11Detector  # noqa: E402

THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50]
CANDIDATE_SHA = "4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879"
PREV_SHA = "6cc84115e4cb85c8b82715211c3935200b815b76efbc95f83855c2cc988dce4f"
FORENSIC = os.path.join("data", "uploads", "3e0dbd17-7475-487d-9f10-e7f9d6800238.jpg")
OUT = os.path.join("scratch", "phase3_yolo_promotion_audit.json")


def file_meta(rel):
    path = abs_path(rel) if not os.path.isabs(rel) else rel
    if not os.path.isfile(path):
        return {"path": posix(rel), "exists": False}
    st = os.stat(path)
    digest = sha256_file(path)
    return {
        "path": posix(path),
        "rel": posix(os.path.relpath(path, ROOT)),
        "exists": True,
        "sha256": digest,
        "size_bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "runtime": posix(os.path.relpath(path, ROOT)).replace("\\", "/") == posix(YOLO_CANONICAL),
    }


def image_stats(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return {"error": "unreadable"}
    h, w = bgr.shape[:2]
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    mean = [float(x) for x in bgr.mean(axis=(0, 1))]
    std = [float(x) for x in bgr.std(axis=(0, 1))]
    unique = int(len(np.unique(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))))
    return {
        "path": posix(path),
        "width": w,
        "height": h,
        "channels": int(bgr.shape[2]) if bgr.ndim == 3 else 1,
        "size_bytes": os.path.getsize(path),
        "sha256": digest,
        "bgr_mean": mean,
        "bgr_std": std,
        "unique_gray_values": unique,
        "is_uniform": unique <= 2,
        "is_near_blank": max(std) < 5.0,
    }


def infer(model, path, conf):
    bgr = cv2.imread(path)
    h, w = (None, None) if bgr is None else bgr.shape[:2]
    res = model(path, conf=conf, verbose=False)[0]
    names = res.names if isinstance(res.names, dict) else {i: n for i, n in enumerate(res.names)}
    dets = []
    if res.boxes is not None:
        for box in res.boxes:
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            cid = int(box.cls[0])
            dets.append({
                "class": str(names.get(cid, cid)).lower(),
                "confidence": float(box.conf[0]),
                "xyxy": xyxy,
            })
    max_conf = max((d["confidence"] for d in dets), default=None)
    return {
        "threshold": conf,
        "n_detections": len(dets),
        "max_confidence": max_conf,
        "classes": sorted({d["class"] for d in dets}),
        "width": w,
        "height": h,
        "detections": dets,
    }


def sweep_model(weights, images):
    from ultralytics import YOLO
    model = YOLO(weights)
    out = {}
    for label, path in images.items():
        if not os.path.isfile(path):
            out[label] = {"missing": True, "path": posix(path)}
            continue
        rows = [infer(model, path, conf) for conf in THRESHOLDS]
        out[label] = {
            "path": posix(path),
            "by_threshold": rows,
            "max_at_0_01": rows[0]["max_confidence"],
            "n_at_0_25": rows[3]["n_detections"],
            "kept_at_application_0_25": rows[3]["detections"],
        }
    return out


def find_hash_in_dataset(image_sha):
    root = abs_path(os.path.join("data", "datasets", "yolo_retrain_v2", "images"))
    hits = []
    if not os.path.isdir(root):
        return hits
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in os.listdir(split_dir):
            path = os.path.join(split_dir, name)
            if not os.path.isfile(path):
                continue
            digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if digest == image_sha:
                hits.append({"split": split, "file": posix(os.path.relpath(path, ROOT))})
    return hits


def list_split_images(split, n=4):
    img_dir = abs_path(os.path.join("data", "datasets", "yolo_retrain_v2", "images", split))
    if not os.path.isdir(img_dir):
        return []
    names = sorted(os.listdir(img_dir))
    return [os.path.join(img_dir, name) for name in names[:n]]


def main():
    canonical = file_meta(YOLO_CANONICAL)
    backup = file_meta(YOLO_CANONICAL + ".pre_retrain_v2_backup")
    run_best = file_meta(YOLO_RETRAIN_V2_BEST)
    registry = read_json(REGISTRY_PATH)
    manifest = read_json(MANIFEST_PATH)
    yolo_reg = registry.get("YOLO11 Detection", {})
    yolo_man = next((m for m in manifest.get("models", []) if m.get("model_name") == "YOLO11 Detection"), {})

    detector = YOLO11Detector()
    info = detector.get_info()

    forensic_stats = image_stats(FORENSIC)
    dataset_hits = find_hash_in_dataset(forensic_stats.get("sha256", ""))

    images = {
        "forensic_upload": FORENSIC,
        "demo_football": os.path.join("data", "sample", "image", "football_injury.jpg"),
        "blank_skin": os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg"),
        "dummy_test": os.path.join("data", "datasets", "yolo_injury", "dummy_test.jpg"),
        "val_neg_blank": os.path.join("data", "datasets", "yolo_retrain_v2", "images", "val", "negative__blank_skin.jpg"),
        "test_neg_dummy": os.path.join("data", "datasets", "yolo_retrain_v2", "images", "test", "negative__dummy_test.jpg"),
    }
    for i, path in enumerate(list_split_images("val", 3)):
        images[f"val_{i}_{os.path.basename(path)}"] = path
    for i, path in enumerate(list_split_images("test", 3)):
        images[f"test_{i}_{os.path.basename(path)}"] = path

    gray_path = os.path.join("data", "test_suite", "phase3_uniform_gray.jpg")
    os.makedirs(os.path.dirname(gray_path), exist_ok=True)
    cv2.imwrite(gray_path, np.full((300, 300, 3), 200, dtype=np.uint8))
    images["synthetic_uniform_gray"] = gray_path

    cand_weights = abs_path(YOLO_CANONICAL)
    prev_src = abs_path(YOLO_CANONICAL + ".pre_retrain_v2_backup")
    assert sha256_file(cand_weights) == CANDIDATE_SHA, sha256_file(cand_weights)
    assert sha256_file(prev_src) == PREV_SHA, sha256_file(prev_src)
    # Ultralytics refuses non-.pt suffixes. Copy bytes only; do not overwrite canonical.
    prev_weights = abs_path(os.path.join("scratch", "phase3_previous_canonical.pt"))
    import shutil
    shutil.copy2(prev_src, prev_weights)
    assert sha256_file(prev_weights) == PREV_SHA

    candidate_sweep = sweep_model(cand_weights, images)
    previous_sweep = sweep_model(prev_weights, images)

    wrapper_forensic = detector.detect(FORENSIC)
    wrapper_blank = detector.detect(images["blank_skin"])
    wrapper_gray = detector.detect(gray_path)
    wrapper_demo = detector.detect(images["demo_football"])

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "default_infer_conf": DEFAULT_YOLO_INFER_CONF,
        "artifacts": {
            "canonical": canonical,
            "pre_retrain_v2_backup": backup,
            "retrain_run_best": run_best,
        },
        "agreement": {
            "disk_sha": canonical.get("sha256"),
            "wrapper_sha": info.get("artifact_sha256"),
            "registry_sha": yolo_reg.get("artifact_sha256"),
            "manifest_sha": yolo_man.get("sha256"),
            "wrapper_canonical_path": info.get("canonical_path"),
            "registry_path": yolo_reg.get("canonical_path"),
            "manifest_path": yolo_man.get("canonical_path"),
            "all_four_match_candidate": all(
                x == CANDIDATE_SHA
                for x in (
                    canonical.get("sha256"),
                    info.get("artifact_sha256"),
                    yolo_reg.get("artifact_sha256"),
                    yolo_man.get("sha256"),
                )
            ),
            "backup_is_previous": backup.get("sha256") == PREV_SHA,
            "run_best_equals_canonical": run_best.get("sha256") == canonical.get("sha256"),
        },
        "wrapper_info": {
            "infer_conf": info.get("infer_conf"),
            "classes": info.get("classes"),
            "task": info.get("task"),
            "status": info.get("status"),
            "model_path": info.get("model_path"),
        },
        "forensic_image": {
            **forensic_stats,
            "in_yolo_retrain_v2_dataset": dataset_hits,
            "classification": (
                "unlabeled_ood_photo_with_visible_injury_like_mark"
                if not forensic_stats.get("is_uniform")
                else "uniform"
            ),
        },
        "wrapper_application_threshold": {
            "threshold": detector.infer_conf,
            "forensic": wrapper_forensic,
            "blank_skin": wrapper_blank,
            "synthetic_gray": wrapper_gray,
            "demo": wrapper_demo,
        },
        "candidate_sweep": candidate_sweep,
        "previous_sweep": previous_sweep,
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote", OUT)
    print("disk", canonical.get("sha256"))
    print("wrapper", info.get("artifact_sha256"))
    print("registry", yolo_reg.get("artifact_sha256"))
    print("manifest", yolo_man.get("sha256"))
    print("backup", backup.get("sha256"))
    print("forensic", forensic_stats)
    print("dataset_hits", dataset_hits)
    print("wrapper forensic n", len(wrapper_forensic), wrapper_forensic)
    for label in ("forensic_upload", "demo_football", "blank_skin", "dummy_test", "synthetic_uniform_gray"):
        c = candidate_sweep.get(label, {})
        p = previous_sweep.get(label, {})
        c01 = (c.get("by_threshold") or [{}])[0]
        p01 = (p.get("by_threshold") or [{}])[0]
        print(
            label,
            "CAND max@0.01",
            c01.get("max_confidence"),
            "n@0.25",
            c.get("n_at_0_25"),
            "| PREV max@0.01",
            p01.get("max_confidence"),
            "n@0.25",
            p.get("n_at_0_25"),
        )


if __name__ == "__main__":
    main()
