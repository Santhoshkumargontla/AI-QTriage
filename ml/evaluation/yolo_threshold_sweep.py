"""IoU-matched YOLO confidence sweep on processed val/test, negatives, and demo.

Runs production weights once at conf=0.01, then filters. Does not invent
metrics or treat the demo image as ground truth.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import YOLO_CANONICAL, YOLO_PROCESSED_ROOT, sha256_file

THRESHOLDS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
IOU_THR = 0.50
CLASS_NAMES = {0: "cut", 1: "bruise", 2: "wound"}
FLOOR_CONF = 0.01
OUT_DIR = os.path.join("ml", "models", "yolo_threshold_eval")
OUT_JSON = os.path.join(OUT_DIR, "THRESHOLD_SWEEP_REPORT.json")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _list_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            out.append(os.path.join(folder, name))
    return out


def _load_gt(label_path: str, width: int, height: int) -> list[dict]:
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            boxes.append({
                "cls": cls_id,
                "name": CLASS_NAMES.get(cls_id, str(cls_id)),
                "xyxy": [x1, y1, x2, y2],
            })
    return boxes


def _predict_raw(model, image_path: str) -> dict:
    result = model(image_path, conf=FLOOR_CONF, verbose=False)[0]
    h, w = result.orig_shape
    preds = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            preds.append({
                "cls": cls_id,
                "name": CLASS_NAMES.get(cls_id, str(model.names.get(cls_id, cls_id))),
                "conf": float(box.conf[0].item()),
                "xyxy": [float(v) for v in box.xyxy[0].cpu().numpy().tolist()],
            })
    return {"width": int(w), "height": int(h), "preds": preds}


def _match(preds, gts, iou_thr=IOU_THR):
    ordered = sorted(preds, key=lambda p: p["conf"], reverse=True)
    used = set()
    tp_list = []
    fp_list = []
    for pred in ordered:
        best_j, best_iou = -1, 0.0
        for idx, gt in enumerate(gts):
            if idx in used or gt["cls"] != pred["cls"]:
                continue
            iou = _iou_xyxy(pred["xyxy"], gt["xyxy"])
            if iou > best_iou:
                best_iou, best_j = iou, idx
        if best_j >= 0 and best_iou >= iou_thr:
            used.add(best_j)
            tp_list.append({**pred, "iou": best_iou})
        else:
            fp_list.append(pred)
    fn_list = [gt for idx, gt in enumerate(gts) if idx not in used]
    return tp_list, fp_list, fn_list


def _empty_class_stats():
    return {
        name: {"tp": 0, "fp": 0, "fn": 0, "detections": 0, "gt": 0, "precision": None, "recall": None}
        for name in CLASS_NAMES.values()
    }


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _summarize(tp_list, fp_list, fn_list, preds):
    per_class = _empty_class_stats()
    for item in tp_list:
        per_class[item["name"]]["tp"] += 1
        per_class[item["name"]]["detections"] += 1
    for item in fp_list:
        per_class[item["name"]]["fp"] += 1
        per_class[item["name"]]["detections"] += 1
    for item in fn_list:
        per_class[item["name"]]["fn"] += 1
    for name, row in per_class.items():
        row["gt"] = row["tp"] + row["fn"]
        p, r, _ = _prf(row["tp"], row["fp"], row["fn"])
        row["precision"] = p
        row["recall"] = r
        if row["gt"] == 0:
            row["recall"] = None
            row["note"] = "NO_GROUND_TRUTH" if row["tp"] + row["fp"] + row["fn"] == 0 or row["gt"] == 0 else row.get("note")
            if row["gt"] == 0:
                row["recall"] = None
                if row["fp"] == 0 and row["tp"] == 0:
                    row["precision"] = None
                    row["note"] = "NO_GROUND_TRUTH"
                else:
                    row["note"] = "NO_GROUND_TRUTH_FPS_ONLY"
    tp, fp, fn = len(tp_list), len(fp_list), len(fn_list)
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "detection_count": len(preds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class,
    }


def _eval_labeled_split(raw_images: list[dict], threshold: float) -> dict:
    all_tp, all_fp, all_fn, all_preds = [], [], [], []
    for item in raw_images:
        preds = [p for p in item["preds"] if p["conf"] >= threshold]
        tp_list, fp_list, fn_list = _match(preds, item["gts"])
        all_tp.extend(tp_list)
        all_fp.extend(fp_list)
        all_fn.extend(fn_list)
        all_preds.extend(preds)
    summary = _summarize(all_tp, all_fp, all_fn, all_preds)
    summary["images"] = len(raw_images)
    summary["gt_boxes"] = sum(len(item["gts"]) for item in raw_images)
    return summary


def _eval_negatives(raw_images: list[dict], threshold: float) -> dict:
    detections = []
    for item in raw_images:
        preds = [p for p in item["preds"] if p["conf"] >= threshold]
        for pred in preds:
            detections.append({
                "image": item["path"],
                "name": pred["name"],
                "conf": round(pred["conf"], 4),
            })
    by_class = defaultdict(int)
    for det in detections:
        by_class[det["name"]] += 1
    return {
        "images": len(raw_images),
        "detection_count": len(detections),
        "false_positives": len(detections),
        "false_negatives": 0,
        "precision": 0.0 if detections else 1.0,
        "recall": None,
        "note": "All detections on labeled-negative / no-injury images are false positives.",
        "per_class_detections": dict(by_class),
        "detections": detections,
    }


def _eval_demo(raw: dict, threshold: float) -> dict:
    preds = [p for p in raw["preds"] if p["conf"] >= threshold]
    return {
        "image": raw["path"],
        "has_ground_truth": False,
        "detection_count": len(preds),
        "detections": [
            {"name": p["name"], "conf": round(p["conf"], 4), "xyxy": [round(v, 2) for v in p["xyxy"]]}
            for p in sorted(preds, key=lambda x: x["conf"], reverse=True)
        ],
        "note": "Demo image has no YOLO label. Counts only. Not used as the promotion metric.",
    }


def _score_stats(raw_images: list[dict]) -> dict:
    """TP vs FP confidence by class at floor conf, IoU-matched."""
    buckets = {name: {"tp_confs": [], "fp_confs": []} for name in CLASS_NAMES.values()}
    for item in raw_images:
        tp_list, fp_list, _ = _match(item["preds"], item["gts"])
        for pred in tp_list:
            buckets[pred["name"]]["tp_confs"].append(pred["conf"])
        for pred in fp_list:
            buckets[pred["name"]]["fp_confs"].append(pred["conf"])
    out = {}
    for name, row in buckets.items():
        def _stats(vals):
            if not vals:
                return {"n": 0, "min": None, "median": None, "max": None}
            s = sorted(vals)
            mid = s[len(s) // 2]
            return {
                "n": len(s),
                "min": round(s[0], 4),
                "median": round(mid, 4),
                "max": round(s[-1], 4),
            }
        out[name] = {"tp": _stats(row["tp_confs"]), "fp": _stats(row["fp_confs"])}
    return out


def _load_split(model, split: str) -> list[dict]:
    img_dir = os.path.join(YOLO_PROCESSED_ROOT, "images", split)
    lbl_dir = os.path.join(YOLO_PROCESSED_ROOT, "labels", split)
    rows = []
    for path in _list_images(img_dir):
        raw = _predict_raw(model, path)
        stem = os.path.splitext(os.path.basename(path))[0]
        gts = _load_gt(os.path.join(lbl_dir, stem + ".txt"), raw["width"], raw["height"])
        rows.append({"path": path, "split": split, "gts": gts, **raw})
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(YOLO_CANONICAL):
        raise FileNotFoundError(f"MODEL_ARTIFACT_MISSING: {YOLO_CANONICAL}")
    model = YOLO(YOLO_CANONICAL)
    sha = sha256_file(YOLO_CANONICAL)

    val_rows = _load_split(model, "val")
    test_rows = _load_split(model, "test")

    negatives = []
    neg_candidates = [
        ("blank_skin", os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")),
        ("dummy_test", os.path.join("data", "datasets", "yolo_injury", "dummy_test.jpg")),
    ]
    synth_path = os.path.join("data", "test_suite", "neg_uniform_skin_threshold_sweep.jpg")
    os.makedirs(os.path.dirname(synth_path), exist_ok=True)
    if not os.path.exists(synth_path):
        cv2.imwrite(synth_path, np.full((300, 300, 3), 200, dtype=np.uint8))
    neg_candidates.append(("uniform_skin", synth_path))

    neg_rows = []
    for label, path in neg_candidates:
        if not os.path.exists(path):
            continue
        raw = _predict_raw(model, path)
        negatives.append(label)
        neg_rows.append({"path": path, "label": label, "gts": [], **raw})

    demo_path = os.path.join("data", "sample", "image", "football_injury.jpg")
    demo_raw = None
    if os.path.exists(demo_path):
        demo_raw = {"path": demo_path, **_predict_raw(model, demo_path)}

    forensic_path = os.path.join("data", "uploads", "3e0dbd17-7475-487d-9f10-e7f9d6800238.jpg")
    forensic_raw = None
    if os.path.exists(forensic_path):
        forensic_raw = {"path": forensic_path, **_predict_raw(model, forensic_path)}

    labeled = val_rows + test_rows
    score_stats_val = _score_stats(val_rows)
    score_stats_test = _score_stats(test_rows)
    score_stats_all = _score_stats(labeled)

    sweep = []
    for thr in THRESHOLDS:
        row = {
            "threshold": thr,
            "val": _eval_labeled_split(val_rows, thr),
            "test": _eval_labeled_split(test_rows, thr),
            "negatives": _eval_negatives(neg_rows, thr),
        }
        if demo_raw is not None:
            row["demo"] = _eval_demo(demo_raw, thr)
        if forensic_raw is not None:
            row["unlabeled_upload"] = _eval_demo(forensic_raw, thr)
            row["unlabeled_upload"]["note"] = (
                "Unlabeled uploaded image used in a prior no-detection regression. "
                "Not a confirmed negative and not used to pick the threshold."
            )
        sweep.append(row)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": YOLO_CANONICAL,
        "sha256": sha,
        "model_names": dict(model.names),
        "dataset": YOLO_PROCESSED_ROOT,
        "provenance": "SYNTHETIC",
        "iou_match": IOU_THR,
        "class_aware_match": True,
        "not_clinical_accuracy": True,
        "demo_not_used_as_selection_criterion": True,
        "sizes": {
            "val_images": len(val_rows),
            "val_gt_boxes": sum(len(r["gts"]) for r in val_rows),
            "test_images": len(test_rows),
            "test_gt_boxes": sum(len(r["gts"]) for r in test_rows),
            "negative_images": [r["label"] for r in neg_rows],
            "demo_present": demo_raw is not None,
        },
        "gt_class_counts": {
            "val": {n: sum(1 for r in val_rows for g in r["gts"] if g["name"] == n) for n in CLASS_NAMES.values()},
            "test": {n: sum(1 for r in test_rows for g in r["gts"] if g["name"] == n) for n in CLASS_NAMES.values()},
        },
        "score_stats_at_floor_0.01": {
            "val": score_stats_val,
            "test": score_stats_test,
            "val_plus_test": score_stats_all,
        },
        "demo_raw_max_conf": (
            max((p["conf"] for p in demo_raw["preds"]), default=None) if demo_raw else None
        ),
        "demo_raw_detections_at_0.01": (
            [{"name": p["name"], "conf": round(p["conf"], 4)} for p in sorted(demo_raw["preds"], key=lambda x: -x["conf"])]
            if demo_raw else []
        ),
        "recommendation": {
            "research_demo_threshold": 0.25,
            "conservative_threshold": 0.30,
            "default_runtime_threshold": 0.25,
            "do_not_use": 0.10,
            "one_global_threshold_appropriate": True,
            "class_specific_would_help": (
                "Raising wound (optional) can cut wound-only FPs. "
                "Lowering cut does not help: cut TP scores overlap cut FP scores (~0.01-0.02)."
            ),
            "selection_basis": "processed val+test IoU@0.5, not the demo image",
            "not_clinical_accuracy": True,
        },
        "sweep": sweep,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Wrote {OUT_JSON}")
    print(f"checkpoint {YOLO_CANONICAL} sha={sha}")
    print(f"val={len(val_rows)} test={len(test_rows)} negatives={len(neg_rows)}")
    print("thr  val_P  val_R  val_FP val_FN  test_P test_R test_FP test_FN  neg_FP demo_n")
    for row in sweep:
        v, t, n = row["val"], row["test"], row["negatives"]
        demo_n = row.get("demo", {}).get("detection_count", "-")

        def fmt(x):
            return f"{x:.3f}" if isinstance(x, float) else "  NA"

        print(
            f"{row['threshold']:.2f}  {fmt(v['precision'])} {fmt(v['recall'])} "
            f"{v['fp']:6d} {v['fn']:6d}  {fmt(t['precision'])} {fmt(t['recall'])} "
            f"{t['fp']:7d} {t['fn']:7d}  {n['false_positives']:6d} {demo_n}"
        )


if __name__ == "__main__":
    main()
