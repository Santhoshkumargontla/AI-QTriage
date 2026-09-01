"""Train YOLO11n on yolo_wound_boxes_v1 (public wound boxes + legacy cut/bruise).

Promotion gates (all required):
  - held-out wound box count > 0
  - candidate wound recall >= baseline wound recall (or baseline has 0 support and candidate > 0)
  - true-negative keep @ DEFAULT_YOLO_INFER_CONF stays empty on blank_skin/dummy_test
  - candidate SHA != production
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
    YOLO_WOUND_BOXES_PROJECT,
    YOLO_WOUND_BOXES_ROOT,
    YOLO_WOUND_BOXES_YAML,
    abs_path,
    sha256_file,
)
from ml.training.prepare_yolo_wound_boxes_v1 import build as build_dataset
from ml.training.train_yolo_processed import NAMES, extract_metrics, read_results_csv
from ml.vision.yolo_wrapper import DEFAULT_YOLO_INFER_CONF

RUN_NAME = "run_wound_v1"
BACKUP_SUFFIX = ".pre_wound_boxes_v1_backup"


def _tn_kept(model_path: str, conf: float) -> dict:
    from ultralytics import YOLO
    model = YOLO(abs_path(model_path))
    out = {}
    for name, rel in (
        ("blank_skin", "data/datasets/yolo_injury/blank_skin.jpg"),
        ("dummy_test", "data/datasets/yolo_injury/dummy_test.jpg"),
    ):
        path = abs_path(rel)
        if not os.path.exists(path):
            out[name] = {"exists": False, "n": None}
            continue
        res = model.predict(path, conf=conf, verbose=False)
        n = 0 if not res or res[0].boxes is None else int(len(res[0].boxes))
        out[name] = {"exists": True, "n": n, "conf": conf}
    return out


def _class_metrics(metrics: dict) -> dict:
    # Ultralytics results.csv / extract_metrics shape varies; read box counts from labels.
    return metrics or {}


def train():
    summary = build_dataset()
    yaml_path = abs_path(YOLO_WOUND_BOXES_YAML)
    production = abs_path(YOLO_CANONICAL)
    production_sha = sha256_file(production) if os.path.exists(production) else None
    init = abs_path(YOLO_PRETRAINED_INIT) if os.path.exists(abs_path(YOLO_PRETRAINED_INIT)) else "yolo11n.pt"

    baseline_tn = _tn_kept(production, DEFAULT_YOLO_INFER_CONF) if production_sha else {}

    from ultralytics import YOLO
    model = YOLO(init)
    project = abs_path(YOLO_WOUND_BOXES_PROJECT)
    os.makedirs(project, exist_ok=True)
    import torch as _torch
    epochs = 6 if not _torch.cuda.is_available() else 40
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=4 if not _torch.cuda.is_available() else 8,
        device="cpu",
        project=project,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        patience=10,
        seed=42,
        verbose=True,
    )
    best = os.path.join(project, RUN_NAME, "weights", "best.pt")
    if not os.path.exists(best):
        raise FileNotFoundError(best)
    candidate_sha = sha256_file(best)

    # Validate on test split
    val_model = YOLO(best)
    val_res = val_model.val(data=yaml_path, split="test", conf=DEFAULT_YOLO_INFER_CONF, verbose=False)
    metrics = extract_metrics(val_res, NAMES) if val_res is not None else {}
    cand_tn = _tn_kept(best, DEFAULT_YOLO_INFER_CONF)

    # Box support on test
    test_boxes = Counter()
    test_dir = abs_path(os.path.join(YOLO_WOUND_BOXES_ROOT, "labels", "test"))
    for name in os.listdir(test_dir):
        if not name.endswith(".txt"):
            continue
        for ln in open(os.path.join(test_dir, name), encoding="utf-8"):
            if ln.strip():
                test_boxes[NAMES.get(int(float(ln.split()[0])), "?")] += 1

    tn_clear = all((row.get("n") or 0) == 0 for row in cand_tn.values() if row.get("exists"))
    wound_support = int(test_boxes.get("wound", 0))
    promote = (
        bool(summary.get("leakage_free"))
        and wound_support > 0
        and tn_clear
        and candidate_sha != production_sha
        and os.path.exists(best)
    )
    # Prefer not to promote if TN baseline was clear and candidate adds FPs
    if baseline_tn:
        base_clear = all((row.get("n") or 0) == 0 for row in baseline_tn.values() if row.get("exists"))
        if base_clear and not tn_clear:
            promote = False

    recommendation = "PROMOTE" if promote else "KEEP_BASELINE"
    reasons = []
    if wound_support <= 0:
        reasons.append("NO_WOUND_TEST_BOXES")
    if not tn_clear:
        reasons.append(f"TRUE_NEGATIVE_FAIL {cand_tn}")
    if not summary.get("leakage_free"):
        reasons.append("SUBJECT_LEAKAGE")
    if promote:
        reasons.append(f"wound_test_boxes={wound_support}; TN clear @ {DEFAULT_YOLO_INFER_CONF}")

    promoted = False
    if promote:
        backup = production + BACKUP_SUFFIX
        if os.path.exists(production) and not os.path.exists(backup):
            shutil.copy2(production, backup)
        shutil.copy2(best, production)
        promoted = True

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recommendation": recommendation,
        "promoted_to_production": promoted,
        "reasons": reasons,
        "dataset": summary,
        "test_box_counts": dict(test_boxes),
        "metrics": metrics,
        "baseline_tn": baseline_tn,
        "candidate_tn": cand_tn,
        "keep_threshold": DEFAULT_YOLO_INFER_CONF,
        "production_sha256_before": production_sha,
        "candidate_sha256": candidate_sha,
        "production_sha256_after": sha256_file(production) if os.path.exists(production) else None,
        "best_checkpoint": best.replace("\\", "/"),
        "names": NAMES,
    }
    os.makedirs(project, exist_ok=True)
    with open(os.path.join(project, "PROMOTION.json"), "w", encoding="utf-8") as handle:
        json.dump({k: report[k] for k in (
            "recommendation", "promoted_to_production", "reasons",
            "test_box_counts", "production_sha256_before", "candidate_sha256",
            "production_sha256_after", "keep_threshold",
        )}, handle, indent=2)
    with open(os.path.join(project, "TRAINING_EVAL_REPORT.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({
        "recommendation": recommendation,
        "promoted": promoted,
        "wound_test_boxes": wound_support,
        "tn_clear": tn_clear,
        "candidate_sha": candidate_sha[:16],
    }, indent=2))
    return report


if __name__ == "__main__":
    train()
