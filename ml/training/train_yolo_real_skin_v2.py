"""Train YOLO real-skin v2 candidate. Never overwrites canonical without passing all gates."""
from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import YOLO_CANONICAL, abs_path, sha256_file
from ml.training.prepare_yolo_real_skin_v2 import OUT_ROOT, build as build_dataset
from ml.training.train_yolo_processed import NAMES, extract_metrics
from ml.vision.yolo_wrapper import DEFAULT_YOLO_INFER_CONF

PROJECT = os.path.join("ml", "models", "yolo_real_skin_candidate_v1")
RUN_NAME = "run_v1"
DATA_YAML = os.path.join(OUT_ROOT, "data.yaml")
BACKUP_SUFFIX = ".pre_real_skin_v2_backup"
FORENSIC_HAND = os.path.join("data", "uploads", "3f629ca8-dd98-427d-a708-f976e2042555.jpeg")
THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50]


def _predict(model_path: str, image_path: str, conf: float) -> list:
    from ultralytics import YOLO
    model = YOLO(abs_path(model_path))
    res = model.predict(abs_path(image_path), conf=conf, verbose=False)
    if not res or res[0].boxes is None:
        return []
    out = []
    names = res[0].names
    for box in res[0].boxes:
        cid = int(box.cls[0].item())
        cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        out.append({
            "class": str(cname).lower(),
            "confidence": float(box.conf[0].item()),
            "bounding_box": [round(float(v), 2) for v in xyxy],
        })
    return out


def _tn_probe(model_path: str, conf: float) -> dict:
    out = {}
    for name, rel in (
        ("blank_skin", "data/datasets/yolo_injury/blank_skin.jpg"),
        ("dummy_test", "data/datasets/yolo_injury/dummy_test.jpg"),
    ):
        path = abs_path(rel)
        dets = _predict(model_path, path, conf) if os.path.isfile(path) else []
        out[name] = {"exists": os.path.isfile(path), "n": len(dets), "conf": conf, "detections": dets}
    return out


def _injury_heuristic_bbox(bgr: np.ndarray) -> list | None:
    """Approximate red injury region for forensic IoU only — not training labels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([168, 60, 40]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _forensic_hand_eval(model_path: str, conf: float) -> dict:
    path = abs_path(FORENSIC_HAND)
    if not os.path.isfile(path):
        return {"available": False}
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    gt = _injury_heuristic_bbox(bgr)
    dets = _predict(model_path, path, conf)
    best = max(dets, key=lambda d: d["confidence"]) if dets else None
    best_iou = _iou(best["bounding_box"], gt) if best and gt else 0.0
    cut_like = [d for d in dets if d["class"] in ("cut", "wound")]
    best_cut = max(cut_like, key=lambda d: d["confidence"]) if cut_like else None
    cut_iou = _iou(best_cut["bounding_box"], gt) if best_cut and gt else 0.0
    return {
        "available": True,
        "image_wh": [w, h],
        "heuristic_gt_bbox": gt,
        "detections": dets,
        "best_detection": best,
        "best_iou_with_heuristic_gt": round(best_iou, 4),
        "best_cut_or_wound": best_cut,
        "best_cut_wound_iou": round(cut_iou, 4),
        "covers_injury_heuristic": best_iou >= 0.15 or cut_iou >= 0.15,
    }


def _test_box_counts() -> Counter:
    counts = Counter()
    lab_dir = abs_path(os.path.join(OUT_ROOT, "labels", "test"))
    for name in os.listdir(lab_dir):
        if not name.endswith(".txt"):
            continue
        for ln in open(os.path.join(lab_dir, name), encoding="utf-8"):
            if ln.strip():
                counts[NAMES[int(float(ln.split()[0]))]] += 1
    return counts


def train_and_evaluate(promote: bool = False):
    prep = build_dataset()
    yaml_path = abs_path(DATA_YAML)
    production = abs_path(YOLO_CANONICAL)
    production_sha = sha256_file(production)

    import torch
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"
    epochs = 40 if torch.cuda.is_available() else 10
    batch = 8 if torch.cuda.is_available() else 4

    # Fine-tune from production checkpoint (same taxonomy nc=3).
    model = YOLO(production)
    project = abs_path(PROJECT)
    os.makedirs(project, exist_ok=True)
    print(f"training real_skin_v2 epochs={epochs} device={device} init=production", flush=True)
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        project=project,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=False,
        patience=12,
        seed=42,
        verbose=True,
        lr0=0.001,
        cos_lr=True,
    )
    best = os.path.join(project, RUN_NAME, "weights", "best.pt")
    if not os.path.isfile(best):
        raise FileNotFoundError(best)
    candidate_sha = sha256_file(best)

    val_model = YOLO(best)
    val_res = val_model.val(data=yaml_path, split="test", conf=DEFAULT_YOLO_INFER_CONF, verbose=False)
    metrics = extract_metrics(val_res, NAMES) if val_res is not None else {}

    prod_metrics = {}
    prod_val = YOLO(production).val(data=yaml_path, split="test", conf=DEFAULT_YOLO_INFER_CONF, verbose=False)
    if prod_val is not None:
        prod_metrics = extract_metrics(prod_val, NAMES)

    test_boxes = _test_box_counts()
    baseline_tn = _tn_probe(production, DEFAULT_YOLO_INFER_CONF)
    candidate_tn = _tn_probe(best, DEFAULT_YOLO_INFER_CONF)

    threshold_sweep = {}
    for ckpt_name, ckpt_path in (("production", production), ("candidate", best)):
        threshold_sweep[ckpt_name] = {}
        for thr in THRESHOLDS:
            threshold_sweep[ckpt_name][str(thr)] = {
                "forensic_hand": _forensic_hand_eval(ckpt_path, thr),
                "blank_skin_n": _tn_probe(ckpt_path, thr)["blank_skin"]["n"],
                "dummy_test_n": _tn_probe(ckpt_path, thr)["dummy_test"]["n"],
            }

    prod_hand = _forensic_hand_eval(production, DEFAULT_YOLO_INFER_CONF)
    cand_hand = _forensic_hand_eval(best, DEFAULT_YOLO_INFER_CONF)

    tn_clear = all((row.get("n") or 0) == 0 for row in candidate_tn.values() if row.get("exists"))
    base_tn_clear = all((row.get("n") or 0) == 0 for row in baseline_tn.values() if row.get("exists"))

    cand_map50 = float(metrics.get("mAP50") or metrics.get("map50") or 0)
    prod_map50 = float(prod_metrics.get("mAP50") or prod_metrics.get("map50") or 0)

    gates = {
        "gate1_real_test_not_regressed": cand_map50 >= prod_map50 * 0.95,
        "gate2_wound_test_support": test_boxes.get("wound", 0) > 0,
        "gate3_negatives_at_025": tn_clear and (
            threshold_sweep["candidate"]["0.25"]["forensic_hand"]["detections"] == [] or
            len(threshold_sweep["candidate"]["0.25"]["forensic_hand"]["detections"]) <=
            len(threshold_sweep["production"]["0.25"]["forensic_hand"]["detections"])
        ),
        "gate4_hand_localization": (
            cand_hand.get("best_cut_wound_iou", 0) > prod_hand.get("best_iou_with_heuristic_gt", 0)
            or cand_hand.get("best_detection", {}) and cand_hand["best_detection"].get("class") in ("cut", "wound")
            and prod_hand.get("best_detection", {}) and prod_hand["best_detection"].get("class") not in ("cut", "wound")
        ),
        "gate5_blank_negatives": (
            threshold_sweep["candidate"]["0.25"]["blank_skin_n"] == 0
            and threshold_sweep["candidate"]["0.25"]["dummy_test_n"] == 0
        ),
        "gate6_leakage_free": prep.get("leakage_free", False),
        "gate7_sha_distinct": candidate_sha != production_sha,
        "gate8_no_fp_regression_on_forensic_at_025": (
            len(threshold_sweep["candidate"]["0.25"]["forensic_hand"].get("detections") or []) <=
            len(threshold_sweep["production"]["0.25"]["forensic_hand"].get("detections") or [])
        ),
    }

    # Strict: hand must improve localization OR class toward cut/wound AND no extra FP vs production on hand at 0.25
    hand_improved = (
        cand_hand.get("best_cut_wound_iou", 0) > prod_hand.get("best_iou_with_heuristic_gt", 0) + 0.05
    ) or (
        cand_hand.get("covers_injury_heuristic") and not prod_hand.get("covers_injury_heuristic")
    )
    gates["gate4_hand_localization"] = hand_improved

    # Block if candidate fires wound on hand when production didn't at same conf — prior revert reason
    prod_dets_025 = threshold_sweep["production"]["0.25"]["forensic_hand"].get("detections") or []
    cand_dets_025 = threshold_sweep["candidate"]["0.25"]["forensic_hand"].get("detections") or []
    gates["gate9_no_new_wound_fp_on_hand"] = not (
        len(cand_dets_025) > len(prod_dets_025)
        and any(d["class"] == "wound" for d in cand_dets_025)
    )

    all_pass = all(gates.values())
    recommendation = "PROMOTE" if all_pass else "KEEP_BASELINE"
    failure_reasons = [k for k, v in gates.items() if not v]

    promoted = False
    if promote and all_pass:
        backup = production + BACKUP_SUFFIX
        if os.path.isfile(production) and not os.path.isfile(backup):
            shutil.copy2(production, backup)
        shutil.copy2(best, production)
        promoted = True
    elif promote and not all_pass:
        recommendation = "PROMOTION_REJECTED"

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recommendation": recommendation,
        "promoted_to_production": promoted,
        "promotion_gates": gates,
        "failure_reasons": failure_reasons,
        "device": device,
        "epochs": epochs,
        "batch": batch,
        "init_checkpoint": "production_finetune",
        "dataset": prep,
        "test_box_counts": dict(test_boxes),
        "production_sha256": production_sha,
        "candidate_sha256": candidate_sha,
        "candidate_path": best.replace("\\", "/"),
        "production_metrics_test": prod_metrics,
        "candidate_metrics_test": metrics,
        "baseline_tn_025": baseline_tn,
        "candidate_tn_025": candidate_tn,
        "forensic_hand_production_025": prod_hand,
        "forensic_hand_candidate_025": cand_hand,
        "threshold_sweep": threshold_sweep,
        "keep_threshold": DEFAULT_YOLO_INFER_CONF,
        "status": "PROMOTION_VALIDATED" if promoted else ("PROMOTION_REJECTED" if not all_pass else "CANDIDATE_READY"),
    }
    os.makedirs(project, exist_ok=True)
    with open(os.path.join(project, "TRAIN_EVAL_REPORT.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(os.path.join(project, "PROMOTION.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "recommendation": recommendation,
            "promoted_to_production": promoted,
            "failure_reasons": failure_reasons,
            "production_sha256": production_sha,
            "candidate_sha256": candidate_sha,
        }, handle, indent=2)
    print(json.dumps({
        "recommendation": recommendation,
        "promoted": promoted,
        "gates": gates,
        "candidate_sha": candidate_sha[:16],
        "prod_map50": prod_map50,
        "cand_map50": cand_map50,
        "hand_prod": prod_hand,
        "hand_cand": cand_hand,
    }, indent=2))
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true", help="Promote only if all gates pass")
    args = parser.parse_args()
    train_and_evaluate(promote=args.promote)
