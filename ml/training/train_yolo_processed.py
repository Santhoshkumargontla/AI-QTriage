"""Train YOLO11n on yolo_processed. Never overwrites production unless promotion rule passes.

Starts from COCO yolo11n.pt, not the production injury checkpoint.
Does not invent metrics. Does not copy existing injury weights and call it training.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import (
    YOLO_CANONICAL,
    YOLO_PROCESSED_ROOT,
    YOLO_PROCESSED_YAML,
    sha256_file,
)

NAMES = {0: "cut", 1: "bruise", 2: "wound"}
PROJECT = os.path.join("ml", "models", "yolo_processed_training")
RUN_NAME = "run_processed"
MIN_TRAIN_IMAGES_PER_CLASS = 10
MIN_TRAIN_BOXES_PER_CLASS = 10


def _rel(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def count_dataset():
    stats = {}
    for split in ("train", "val", "test"):
        img_dir = os.path.join(YOLO_PROCESSED_ROOT, "images", split)
        lbl_dir = os.path.join(YOLO_PROCESSED_ROOT, "labels", split)
        images = [
            p for p in glob.glob(os.path.join(img_dir, "*"))
            if os.path.splitext(p)[1].lower() in {".jpg", ".jpeg", ".png"}
        ]
        boxes = Counter()
        images_with = Counter()
        missing_label = 0
        empty_label = 0
        invalid = 0
        for img in images:
            stem = os.path.splitext(os.path.basename(img))[0]
            lbl = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lbl):
                missing_label += 1
                continue
            present = set()
            lines = [ln.strip() for ln in open(lbl, encoding="utf-8") if ln.strip()]
            if not lines:
                empty_label += 1
                continue
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    invalid += 1
                    continue
                try:
                    cid = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                except ValueError:
                    invalid += 1
                    continue
                if cid not in NAMES or w <= 0 or h <= 0:
                    invalid += 1
                    continue
                boxes[NAMES[cid]] += 1
                present.add(NAMES[cid])
            for name in present:
                images_with[name] += 1
        stats[split] = {
            "images": len(images),
            "boxes": {n: int(boxes.get(n, 0)) for n in NAMES.values()},
            "images_per_class": {n: int(images_with.get(n, 0)) for n in NAMES.values()},
            "missing_label": missing_label,
            "empty_label": empty_label,
            "invalid_lines": invalid,
        }
    return stats


def preflight(stats):
    issues = []
    train = stats["train"]
    if train["images"] < 1:
        issues.append({"severity": "blocker", "code": "NO_TRAIN_IMAGES", "detail": train})
    if train["missing_label"] or train["empty_label"] or train["invalid_lines"]:
        issues.append({
            "severity": "blocker",
            "code": "INVALID_TRAIN_LABELS",
            "detail": {
                "missing_label": train["missing_label"],
                "empty_label": train["empty_label"],
                "invalid_lines": train["invalid_lines"],
            },
        })
    for name in NAMES.values():
        n_img = train["images_per_class"].get(name, 0)
        n_box = train["boxes"].get(name, 0)
        if n_img == 0 or n_box == 0:
            issues.append({
                "severity": "blocker",
                "code": "CLASS_HAS_NO_TRAINING_EXAMPLES",
                "class": name,
                "train_images": n_img,
                "train_boxes": n_box,
                "val_images": stats["val"]["images_per_class"].get(name, 0),
                "test_images": stats["test"]["images_per_class"].get(name, 0),
            })
        elif n_img < MIN_TRAIN_IMAGES_PER_CLASS or n_box < MIN_TRAIN_BOXES_PER_CLASS:
            issues.append({
                "severity": "warning",
                "code": "CLASS_TOO_LITTLE_DATA",
                "class": name,
                "train_images": n_img,
                "train_boxes": n_box,
                "minimum_images": MIN_TRAIN_IMAGES_PER_CLASS,
                "minimum_boxes": MIN_TRAIN_BOXES_PER_CLASS,
            })
    return issues


def extract_metrics(result, expected_names):
    box = result.box
    names = dict(getattr(result, "names", expected_names) or expected_names)
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
    nc = len(expected_names)

    def vec(attr):
        value = getattr(box, attr, None)
        if value is None:
            return [None] * nc
        if hasattr(value, "tolist"):
            value = value.tolist()
        value = list(value)
        if len(value) < nc:
            value = value + [None] * (nc - len(value))
        return [_finite(v) for v in value[:nc]]

    precision = vec("p")
    recall = vec("r")
    map50 = vec("ap50") if hasattr(box, "ap50") else [None] * nc
    map5095 = vec("maps")
    per_class = {}
    for idx, name in expected_names.items():
        per_class[name] = {
            "precision": precision[idx] if idx < len(precision) else None,
            "recall": recall[idx] if idx < len(recall) else None,
            "mAP50": map50[idx] if idx < len(map50) else None,
            "mAP50-95": map5095[idx] if idx < len(map5095) else None,
        }
        if per_class[name]["precision"] is None and per_class[name]["recall"] is None:
            per_class[name]["note"] = "NO_GROUND_TRUTH_OR_UNDEFINED"
    matrix = None
    labels = None
    cm = getattr(result, "confusion_matrix", None)
    if cm is not None and getattr(cm, "matrix", None) is not None:
        matrix = np.asarray(cm.matrix).tolist()
        labels = [expected_names.get(i, str(i)) for i in range(len(expected_names))] + ["background"]
    return {
        "precision": _finite(getattr(box, "mp", None)),
        "recall": _finite(getattr(box, "mr", None)),
        "mAP50": _finite(getattr(box, "map50", None)),
        "mAP50-95": _finite(getattr(box, "map", None)),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": labels,
        "model_names": names,
        "metrics_source": "ultralytics.val",
        "not_clinical_accuracy": True,
    }


def evaluate(weights, data_yaml, split, save_dir):
    from ultralytics import YOLO
    os.makedirs(save_dir, exist_ok=True)
    model = YOLO(weights)
    result = model.val(
        data=data_yaml,
        split=split,
        plots=True,
        save_json=False,
        project=save_dir,
        name=f"val_{split}",
        exist_ok=True,
        verbose=False,
        workers=0,
    )
    metrics = extract_metrics(result, NAMES)
    run_dir = os.path.join(save_dir, f"val_{split}")
    metrics["artifact_dir"] = _rel(run_dir)
    metrics["plots"] = {
        os.path.basename(p): _rel(p)
        for p in glob.glob(os.path.join(run_dir, "*.png"))
    }
    return metrics


def read_results_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    def col(row, *keys):
        for key in keys:
            for actual in row:
                if actual.strip() == key:
                    return _finite(row[actual])
        return None
    first, last = rows[0], rows[-1]
    return {
        "epochs_logged": len(rows),
        "first_epoch_box_loss": col(first, "train/box_loss"),
        "last_epoch_box_loss": col(last, "train/box_loss"),
        "last_epoch_val_mAP50": col(last, "metrics/mAP50(B)", "metrics/mAP50(B) "),
        "last_epoch_val_mAP50-95": col(last, "metrics/mAP50-95(B)", "metrics/mAP50-95(B) "),
        "note": "results.csv epoch rows are training-run logs. They are not used as the promotion metric.",
    }


def decide_promotion(preflight_issues, train_log, baseline_val, candidate_val, baseline_test, candidate_test, prod_sha, cand_sha):
    reasons = []
    blockers = [i for i in preflight_issues if i["severity"] == "blocker"]
    if blockers:
        reasons.append("PREFLIGHT_BLOCKER: " + ", ".join(i["code"] + (":" + i.get("class", "") if i.get("class") else "") for i in blockers))
    if not train_log or train_log["epochs_logged"] < 2:
        reasons.append("TRAINING_LOG_TOO_SHORT")
    elif (
        train_log["first_epoch_box_loss"] is not None
        and train_log["last_epoch_box_loss"] is not None
        and not (train_log["last_epoch_box_loss"] < train_log["first_epoch_box_loss"])
    ):
        reasons.append("BOX_LOSS_DID_NOT_DECREASE")
    if cand_sha == prod_sha:
        reasons.append("CANDIDATE_SHA_EQUALS_PRODUCTION")
    val_c = candidate_val.get("mAP50-95")
    val_b = baseline_val.get("mAP50-95")
    test_c = candidate_test.get("mAP50-95")
    test_b = baseline_test.get("mAP50-95")
    if val_c is None or val_b is None:
        reasons.append("VAL_MAP_UNDEFINED")
    elif not (val_c > val_b):
        reasons.append(f"VAL_mAP50-95_NOT_GREATER candidate={val_c} baseline={val_b}")
    if test_c is None or test_b is None:
        reasons.append("TEST_MAP_UNDEFINED")
    elif test_c < test_b:
        reasons.append(f"TEST_mAP50-95_REGRESSED candidate={test_c} baseline={test_b}")
    promote = len(reasons) == 0
    return {
        "rule": {
            "use_training_metrics_for_promotion": False,
            "require_all_yaml_classes_have_train_examples": True,
            "min_train_images_per_class": MIN_TRAIN_IMAGES_PER_CLASS,
            "candidate_val_mAP50-95_must_be_strictly_greater_than_baseline": True,
            "candidate_test_mAP50-95_must_not_be_lower_than_baseline": True,
            "candidate_sha_must_differ_from_production": True,
            "box_loss_must_decrease": True,
            "production_overwritten_only_if_promote": True,
        },
        "recommendation": "PROMOTE" if promote else "KEEP_BASELINE",
        "reasons": reasons or ["all_rules_passed"],
        "not_clinical_accuracy": True,
    }


def find_init_weights():
    for path in ("yolo11n.pt", os.path.join("ml", "models", "yolo11n_pretrained.pt")):
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            return path
    raise FileNotFoundError("COCO yolo11n.pt not found. Refusing to start from production injury weights.")


def train_and_evaluate(epochs=25, imgsz=640, batch=4):
    from ultralytics import YOLO
    import torch

    os.makedirs(PROJECT, exist_ok=True)
    production_sha_before = sha256_file(YOLO_CANONICAL)
    stats = count_dataset()
    issues = preflight(stats)
    print("PREFLIGHT", json.dumps({"stats": stats, "issues": issues}, indent=2))

    init_weights = find_init_weights()
    if sha256_file(init_weights) == production_sha_before:
        raise RuntimeError("Init weights SHA-256 matches production. Refusing to copy production and call it training.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_yaml = os.path.abspath(YOLO_PROCESSED_YAML)
    run_dir = os.path.join(PROJECT, RUN_NAME)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    model = YOLO(init_weights)
    print(f"TRAIN start init={init_weights} device={device} epochs={epochs} data={data_yaml}")
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
        patience=10,
        workers=0,
        verbose=True,
    )
    save_dir = str(getattr(train_results, "save_dir", run_dir))
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    last_pt = os.path.join(save_dir, "weights", "last.pt")
    if not os.path.exists(best_pt) or not os.path.exists(last_pt):
        raise FileNotFoundError(f"Training did not write best/last: {best_pt} {last_pt}")

    train_plots = {
        os.path.basename(p): _rel(p)
        for p in glob.glob(os.path.join(save_dir, "*.png"))
    }
    required_plot_keys = []
    for pattern in ("confusion_matrix", "PR", "F1", "BoxPR", "BoxF1"):
        hits = [k for k in train_plots if pattern.lower() in k.lower().replace(" ", "").replace("_", "")]
        required_plot_keys.append({"want": pattern, "found": hits})

    train_log = read_results_csv(os.path.join(save_dir, "results.csv"))
    args_yaml = os.path.join(save_dir, "args.yaml")

    eval_root = os.path.join(save_dir, "held_out_eval")
    baseline_val = evaluate(YOLO_CANONICAL, data_yaml, "val", os.path.join(eval_root, "baseline"))
    baseline_test = evaluate(YOLO_CANONICAL, data_yaml, "test", os.path.join(eval_root, "baseline"))
    candidate_val = evaluate(best_pt, data_yaml, "val", os.path.join(eval_root, "candidate"))
    candidate_test = evaluate(best_pt, data_yaml, "test", os.path.join(eval_root, "candidate"))

    candidate_sha = sha256_file(best_pt)
    production_sha_after = sha256_file(YOLO_CANONICAL)
    if production_sha_after != production_sha_before:
        raise RuntimeError("Production checkpoint changed during training. Aborting.")

    decision = decide_promotion(
        issues, train_log, baseline_val, candidate_val, baseline_test, candidate_test,
        production_sha_before, candidate_sha,
    )

    promoted = False
    if decision["recommendation"] == "PROMOTE":
        backup = YOLO_CANONICAL + ".pre_processed_retrain_backup"
        if not os.path.exists(backup):
            shutil.copy2(YOLO_CANONICAL, backup)
        shutil.copy2(best_pt, YOLO_CANONICAL)
        promoted = True
    production_sha_final = sha256_file(YOLO_CANONICAL)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "did_not_fabricate_metrics": True,
        "did_not_use_train_metrics_as_generalization": True,
        "not_clinical_accuracy": True,
        "dataset": {
            "yaml": _rel(YOLO_PROCESSED_YAML),
            "root": _rel(YOLO_PROCESSED_ROOT),
            "provenance": "SYNTHETIC",
            "size": stats,
            "class_distribution": {
                "train_images_per_class": stats["train"]["images_per_class"],
                "train_boxes_per_class": stats["train"]["boxes"],
                "val_images_per_class": stats["val"]["images_per_class"],
                "test_images_per_class": stats["test"]["images_per_class"],
            },
        },
        "preflight_issues": issues,
        "training_configuration": {
            "framework": "ultralytics",
            "init_weights": _rel(init_weights),
            "init_sha256": sha256_file(init_weights),
            "not_started_from_production": sha256_file(init_weights) != production_sha_before,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "seed": 42,
            "patience": 10,
            "project": _rel(PROJECT),
            "run_name": RUN_NAME,
            "save_dir": _rel(save_dir),
        },
        "artifacts": {
            "args.yaml": _rel(args_yaml) if os.path.exists(args_yaml) else None,
            "results.csv": _rel(os.path.join(save_dir, "results.csv")) if os.path.exists(os.path.join(save_dir, "results.csv")) else None,
            "best.pt": _rel(best_pt),
            "last.pt": _rel(last_pt),
            "best_sha256": candidate_sha,
            "plots": train_plots,
            "plot_check": required_plot_keys,
        },
        "training_log_summary": train_log,
        "validation_results": {
            "baseline": baseline_val,
            "candidate": candidate_val,
        },
        "test_results": {
            "baseline": baseline_test,
            "candidate": candidate_test,
        },
        "baseline_comparison": {
            "production_path": _rel(YOLO_CANONICAL),
            "production_sha256_before": production_sha_before,
            "production_sha256_after_training_before_decision": production_sha_after,
            "production_sha256_final": production_sha_final,
            "production_unchanged": production_sha_final == production_sha_before,
            "candidate_path": _rel(best_pt),
            "same_dataset": _rel(YOLO_PROCESSED_YAML),
            "val_mAP50-95": {
                "baseline": baseline_val.get("mAP50-95"),
                "candidate": candidate_val.get("mAP50-95"),
            },
            "test_mAP50-95": {
                "baseline": baseline_test.get("mAP50-95"),
                "candidate": candidate_test.get("mAP50-95"),
            },
        },
        "promotion": decision,
        "promoted_to_production": promoted,
    }
    report_path = os.path.join(save_dir, "TRAINING_EVAL_REPORT.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    shutil.copy2(report_path, os.path.join(PROJECT, "TRAINING_EVAL_REPORT.json"))
    print("REPORT", report_path)
    print("RECOMMENDATION", decision["recommendation"])
    print("REASONS", decision["reasons"])
    return report


if __name__ == "__main__":
    train_and_evaluate()
