"""Retrain YOLO11n on yolo_retrain_v2. Never copies production weights as the candidate.

Backs up the canonical checkpoint first. Overwrites production only if the
honest held-out split shows a real cut/bruise improvement vs the live baseline.
"""
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

from ml.models.canonical_paths import (
    YOLO_CANONICAL,
    YOLO_PRETRAINED_INIT,
    YOLO_RETRAIN_V2_ROOT,
    YOLO_RETRAIN_V2_YAML,
    abs_path,
    posix,
    sha256_file,
)
from ml.training.prepare_yolo_retrain_v2 import prepare as prepare_dataset
from ml.training.train_yolo_processed import (
    NAMES,
    extract_metrics,
    read_results_csv,
)
from ml.vision.yolo_wrapper import DEFAULT_YOLO_INFER_CONF

PROJECT = os.path.join("ml", "models", "yolo_retrain_v2")
RUN_NAME = "run_v2"
BACKUP_SUFFIX = ".pre_retrain_v2_backup"
THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50]
REPORT_PATH = os.path.join(PROJECT, "RETRAIN_V2_REPORT.json")


def _rel(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def backup_canonical():
    src = abs_path(YOLO_CANONICAL)
    dest = src + BACKUP_SUFFIX
    shutil.copy2(src, dest)
    return dest, sha256_file(src)


def split_support(split: str):
    labels_dir = abs_path(os.path.join(YOLO_RETRAIN_V2_ROOT, "labels", split))
    boxes = Counter()
    images = Counter()
    n_images = 0
    n_empty = 0
    if not os.path.isdir(labels_dir):
        return {"images": 0, "empty_label_images": 0, "boxes": {}, "images_with_class": {}}
    for name in os.listdir(labels_dir):
        if not name.endswith(".txt"):
            continue
        n_images += 1
        lines = [ln.strip() for ln in open(os.path.join(labels_dir, name), encoding="utf-8") if ln.strip()]
        if not lines:
            n_empty += 1
            continue
        present = set()
        for line in lines:
            cid = int(float(line.split()[0]))
            cname = NAMES.get(cid, str(cid))
            boxes[cname] += 1
            present.add(cname)
        for cname in present:
            images[cname] += 1
    return {
        "images": n_images,
        "empty_label_images": n_empty,
        "boxes": dict(boxes),
        "images_with_class": dict(images),
    }


def evaluate(weights, data_yaml, split, save_dir):
    from ultralytics import YOLO
    os.makedirs(save_dir, exist_ok=True)
    model = YOLO(weights)
    result = model.val(
        data=data_yaml,
        split=split,
        verbose=False,
        plots=True,
        project=save_dir,
        name=split,
        exist_ok=True,
        workers=0,
    )
    metrics = extract_metrics(result, NAMES)
    metrics["split"] = split
    metrics["weights"] = posix(weights)
    metrics["sha256"] = sha256_file(weights)
    metrics["actual_sample_support"] = split_support(split)
    return metrics


def infer_at(model, path, conf):
    bgr = cv2.imread(path)
    if bgr is None:
        return {"image": posix(path), "error": "unreadable", "threshold": conf}
    h, w = bgr.shape[:2]
    res = model(path, conf=conf, verbose=False)[0]
    dets = []
    names = res.names if isinstance(res.names, dict) else {i: n for i, n in enumerate(res.names)}
    if res.boxes is not None:
        for box in res.boxes:
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            cid = int(box.cls[0])
            dets.append({
                "class": str(names.get(cid, cid)).lower(),
                "confidence": round(float(box.conf[0]), 4),
                "xyxy": [round(v, 2) for v in xyxy],
                "corner_like": (xyxy[0] <= 8 and xyxy[1] <= 8)
                or (xyxy[0] <= 8 and xyxy[3] >= h - 8)
                or (xyxy[2] >= w - 8 and xyxy[1] <= 8)
                or (xyxy[2] >= w - 8 and xyxy[3] >= h - 8),
            })
    return {
        "image": posix(os.path.relpath(path, abs_path("."))) if os.path.isabs(path) else posix(path),
        "width": w,
        "height": h,
        "threshold": conf,
        "n_detections": len(dets),
        "n_corner": sum(1 for d in dets if d["corner_like"]),
        "detections": dets,
    }


def sweep(weights, images):
    from ultralytics import YOLO
    model = YOLO(weights)
    rows = []
    for conf in THRESHOLDS:
        for path in images:
            if os.path.isfile(path):
                rows.append(infer_at(model, path, conf))
    return rows


def kept_detections(rows, image_substr, keep=DEFAULT_YOLO_INFER_CONF):
    """Count boxes kept on one image at the application threshold. None if missing."""
    needle = str(image_substr).replace("\\", "/")
    found = None
    for row in rows or []:
        image = str(row.get("image") or "").replace("\\", "/")
        if needle not in image:
            continue
        if abs(float(row.get("threshold", -1.0)) - float(keep)) > 1e-9:
            continue
        found = int(row.get("n_detections", 0))
    return found


def decide(baseline_test, candidate_test, train_log, prod_sha, cand_sha, sweeps=None):
    reasons = []
    if cand_sha == prod_sha:
        reasons.append("CANDIDATE_SHA_EQUALS_PRODUCTION")
    if not train_log or train_log.get("epochs_logged", 0) < 2:
        reasons.append("TRAINING_LOG_TOO_SHORT")
    elif not (
        train_log.get("last_epoch_box_loss") is not None
        and train_log.get("first_epoch_box_loss") is not None
        and train_log["last_epoch_box_loss"] < train_log["first_epoch_box_loss"]
    ):
        reasons.append("BOX_LOSS_DID_NOT_DECREASE")

    b_cut = (baseline_test.get("per_class") or {}).get("cut") or {}
    c_cut = (candidate_test.get("per_class") or {}).get("cut") or {}
    b_bru = (baseline_test.get("per_class") or {}).get("bruise") or {}
    c_bru = (candidate_test.get("per_class") or {}).get("bruise") or {}
    b_cut_r = b_cut.get("recall")
    c_cut_r = c_cut.get("recall")
    b_bru_r = b_bru.get("recall")
    c_bru_r = c_bru.get("recall")
    if c_cut_r is None:
        reasons.append("CANDIDATE_CUT_RECALL_UNDEFINED")
    elif b_cut_r is None or not (c_cut_r > b_cut_r):
        reasons.append(f"CUT_RECALL_NOT_GREATER candidate={c_cut_r} baseline={b_cut_r}")
    if c_bru_r is None:
        reasons.append("CANDIDATE_BRUISE_RECALL_UNDEFINED")
    elif b_bru_r is not None and c_bru_r + 1e-9 < b_bru_r - 0.05:
        reasons.append(f"BRUISE_RECALL_REGRESSED candidate={c_bru_r} baseline={b_bru_r}")

    sweeps = sweeps or {}
    baseline_neg = sweeps.get("baseline_demo_negatives") or []
    candidate_neg = sweeps.get("candidate_demo_negatives") or []
    keep = DEFAULT_YOLO_INFER_CONF
    negative_gate = {"keep_threshold": keep, "images": {}}
    for needle, key in (("blank_skin", "blank_skin"), ("dummy_test", "dummy_test")):
        baseline_n = kept_detections(baseline_neg, needle, keep)
        candidate_n = kept_detections(candidate_neg, needle, keep)
        negative_gate["images"][key] = {
            "baseline_n_detections": baseline_n,
            "candidate_n_detections": candidate_n,
        }
        if baseline_n is None or candidate_n is None:
            reasons.append(f"NEGATIVE_{key.upper()}_SWEEP_MISSING_AT_{keep}")
        elif candidate_n > baseline_n:
            reasons.append(
                f"NEGATIVE_{key.upper()}_FP_REGRESSED candidate={candidate_n} "
                f"baseline={baseline_n} keep={keep}"
            )

    promote = len(reasons) == 0
    return {
        "recommendation": "PROMOTE" if promote else "KEEP_BASELINE",
        "reasons": reasons or [
            "cut_recall_improved_and_bruise_did_not_collapse_and_true_negatives_did_not_regress"
        ],
        "promote": promote,
        "negative_gate": negative_gate,
        "note": (
            "PROMOTION_VALIDATED_WITH_LIMITATIONS. "
            "Wound has 0 honest labels (UNSUPPORTED). Overall mAP is not used as the promotion "
            "gate. Cut/bruise test support is still small; promotion is not a reliability claim. "
            f"True-negative images (blank_skin, dummy_test) must not gain boxes at keep={keep}. "
            "Those TN images are synthetic/blank-only and insufficient for real-world FP claims."
        ),
    }


def train_and_evaluate(epochs=20, imgsz=640, batch=4):
    from ultralytics import YOLO
    import torch

    os.makedirs(PROJECT, exist_ok=True)
    dataset_summary = prepare_dataset()
    backup_path, prod_sha_before = backup_canonical()
    data_yaml = abs_path(YOLO_RETRAIN_V2_YAML)
    init_weights = abs_path(YOLO_PRETRAINED_INIT)
    if not os.path.isfile(init_weights) or os.path.getsize(init_weights) < 1_000_000:
        raise FileNotFoundError("COCO yolo11n_pretrained.pt missing. Refusing to start from production.")
    if sha256_file(init_weights) == prod_sha_before:
        raise RuntimeError("Init weights SHA matches production. Refusing to copy production.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(PROJECT, RUN_NAME)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)

    model = YOLO(init_weights)
    print(f"TRAIN start init={init_weights} device={device} epochs={epochs}")
    train_results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=os.path.abspath(PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        seed=42,
        pretrained=True,
        plots=True,
        save=True,
        val=True,
        patience=8,
        workers=0,
        mosaic=0.1,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=5,
        verbose=True,
    )
    save_dir = str(getattr(train_results, "save_dir", run_dir))
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    last_pt = os.path.join(save_dir, "weights", "last.pt")
    if not os.path.isfile(best_pt):
        raise FileNotFoundError(f"Training did not write {best_pt}")
    if sha256_file(abs_path(YOLO_CANONICAL)) != prod_sha_before:
        raise RuntimeError("Production checkpoint changed during training.")

    train_log = read_results_csv(os.path.join(save_dir, "results.csv"))
    eval_root = os.path.join(save_dir, "held_out_eval")
    baseline_val = evaluate(abs_path(YOLO_CANONICAL), data_yaml, "val", os.path.join(eval_root, "baseline"))
    baseline_test = evaluate(abs_path(YOLO_CANONICAL), data_yaml, "test", os.path.join(eval_root, "baseline"))
    candidate_val = evaluate(best_pt, data_yaml, "val", os.path.join(eval_root, "candidate"))
    candidate_test = evaluate(best_pt, data_yaml, "test", os.path.join(eval_root, "candidate"))
    cand_sha = sha256_file(best_pt)

    demo = abs_path("data/sample/image/football_injury.jpg")
    blank = abs_path("data/datasets/yolo_injury/blank_skin.jpg")
    dummy = abs_path("data/datasets/yolo_injury/dummy_test.jpg")
    sweeps = {
        "baseline_demo_negatives": sweep(abs_path(YOLO_CANONICAL), [demo, blank, dummy]),
        "candidate_demo_negatives": sweep(best_pt, [demo, blank, dummy]),
    }
    decision = decide(
        baseline_test,
        candidate_test,
        train_log,
        prod_sha_before,
        cand_sha,
        sweeps=sweeps,
    )

    promoted = False
    if decision["promote"]:
        shutil.copy2(best_pt, abs_path(YOLO_CANONICAL))
        promoted = True
        print(f"PROMOTED candidate {cand_sha} over {prod_sha_before}")
    else:
        print("KEEP_BASELINE", decision["reasons"])

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "did_not_fabricate_metrics": True,
        "did_not_copy_production_as_candidate": True,
        "backup_path": posix(backup_path),
        "production_sha_before": prod_sha_before,
        "candidate_path": posix(best_pt),
        "candidate_sha256": cand_sha,
        "promoted": promoted,
        "production_sha_after": sha256_file(abs_path(YOLO_CANONICAL)),
        "dataset": dataset_summary,
        "training": {
            "init_weights": posix(init_weights),
            "init_sha256": sha256_file(init_weights),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "mosaic": 0.1,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "save_dir": posix(save_dir),
            "train_log": train_log,
        },
        "baseline_val": baseline_val,
        "baseline_test": baseline_test,
        "candidate_val": candidate_val,
        "candidate_test": candidate_test,
        "decision": decision,
        "threshold_sweeps": sweeps,
        "support": dataset_summary.get("boxes_per_class_by_split"),
    }
    os.makedirs(PROJECT, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote", REPORT_PATH)
    return report


if __name__ == "__main__":
    train_and_evaluate()
