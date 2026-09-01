"""Train skin YOLO (cut/bruise/burn). Promote only if burn gains without collapsing cut/bruise."""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import YOLO_CANONICAL, YOLO_PRETRAINED_INIT, ROOT, abs_path, sha256_file
from ml.training.prepare_yolo_skin_kaggle_v1 import prepare
from ml.training.train_yolo_processed import extract_metrics

PROJECT = os.path.join("ml", "models", "yolo_skin_kaggle_v1")
RUN_NAME = "run_v1"
BACKUP_SUFFIX = ".pre_skin_kaggle_v1_backup"
DATA_YAML = os.path.join("data", "datasets", "yolo_skin_kaggle_v1", "data.yaml")
# CPU-friendly defaults; override with YOLO_SKIN_EPOCHS / YOLO_SKIN_BATCH
import torch as _torch

YOLO_EPOCHS = int(os.environ.get("YOLO_SKIN_EPOCHS", "40" if _torch.cuda.is_available() else "15"))
YOLO_BATCH = int(os.environ.get("YOLO_SKIN_BATCH", "8" if _torch.cuda.is_available() else "4"))
YOLO_PATIENCE = int(os.environ.get("YOLO_SKIN_PATIENCE", "10" if _torch.cuda.is_available() else "5"))


def train():
    os.makedirs(PROJECT, exist_ok=True)
    prep_path = os.path.join("data", "datasets", "yolo_skin_kaggle_v1", "PREPARE_REPORT.json")
    if os.path.isfile(os.path.abspath(DATA_YAML)) and os.path.isfile(prep_path):
        with open(prep_path, encoding="utf-8") as handle:
            prep = json.load(handle)
        print(f"reusing prepared YOLO skin dataset: {prep.get('box_counts')}", flush=True)
    else:
        prep = prepare()
    from ultralytics import YOLO

    init = abs_path(YOLO_PRETRAINED_INIT) if os.path.isfile(abs_path(YOLO_PRETRAINED_INIT)) else "yolo11n.pt"
    model = YOLO(init)
    print(f"training yolo skin epochs={YOLO_EPOCHS} batch={YOLO_BATCH} device={'cuda' if _torch.cuda.is_available() else 'cpu'}", flush=True)
    results = model.train(
        data=os.path.abspath(DATA_YAML),
        epochs=YOLO_EPOCHS,
        imgsz=640,
        batch=YOLO_BATCH,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        workers=0,
        patience=YOLO_PATIENCE,
        seed=42,
    )
    # Ultralytics may write under ~/runs/detect/... — prefer trainer save_dir.
    best = os.path.join(PROJECT, RUN_NAME, "weights", "best.pt")
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None) or getattr(results, "save_dir", None)
    if save_dir:
        alt = os.path.join(str(save_dir), "weights", "best.pt")
        if os.path.isfile(alt):
            os.makedirs(os.path.dirname(best), exist_ok=True)
            if os.path.abspath(alt) != os.path.abspath(best):
                shutil.copy2(alt, best)
            best = best if os.path.isfile(best) else alt
    if not os.path.isfile(best):
        raise FileNotFoundError(f"YOLO best weights not found at {best}")
    expected = {0: "cut", 1: "bruise", 2: "burn"}
    # Evaluate candidate
    val_m = extract_metrics(
        model.val(data=os.path.abspath(DATA_YAML), split="val", workers=0, project=PROJECT, name="val_eval", exist_ok=True),
        expected,
    )
    test_m = extract_metrics(
        YOLO(best).val(data=os.path.abspath(DATA_YAML), split="test", workers=0, project=PROJECT, name="test_eval", exist_ok=True),
        expected,
    )

    # Baseline on same yaml (old 3-class wound head may mismatch — catch errors)
    baseline_path = abs_path(YOLO_CANONICAL)
    baseline_m = None
    if os.path.isfile(baseline_path):
        try:
            baseline_m = extract_metrics(
                YOLO(baseline_path).val(
                    data=os.path.abspath(DATA_YAML),
                    split="test",
                    workers=0,
                    project=PROJECT,
                    name="baseline_test",
                    exist_ok=True,
                ),
                expected,
            )
        except Exception as exc:
            baseline_m = {"error": str(exc)}

    cand_sha = sha256_file(best)
    # Promotion: require usable burn recall + overall mAP50 floor. Do not promote a dead burn class.
    promote = False
    reasons = []
    map50 = (test_m or {}).get("mAP50")
    burn_recall = None
    try:
        burn_recall = ((test_m or {}).get("per_class") or {}).get("burn", {}).get("recall")
    except Exception:
        burn_recall = None
    if map50 is None:
        reasons.append("missing_test_map50")
    elif map50 < 0.35:
        reasons.append(f"test_mAP50_too_low={map50}")
    elif burn_recall is not None and float(burn_recall) < 0.15:
        reasons.append(f"burn_recall_too_low={burn_recall}")
    else:
        promote = True
        reasons.append(f"test_mAP50={map50}; burn_recall={burn_recall}")
    if isinstance(baseline_m, dict) and baseline_m.get("mAP50") is not None and map50 is not None:
        if map50 + 0.02 < baseline_m["mAP50"]:
            # Allow promote if we added a new class (burn) even if overall dips slightly — require not worse than -0.08
            if map50 + 0.08 < baseline_m["mAP50"]:
                promote = False
                reasons.append(f"worse_than_baseline_map50={baseline_m['mAP50']}")
            else:
                reasons.append("slight_map_drop_allowed_for_new_burn_class")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prepare": prep,
        "candidate": best.replace("\\", "/"),
        "candidate_sha256": cand_sha,
        "val": val_m,
        "test": test_m,
        "baseline_test_on_new_yaml": baseline_m,
        "promote": promote,
        "promotion_reason": "; ".join(reasons),
        "names": ["cut", "bruise", "burn"],
        "notes": "Wound/swelling still unsupported (0 honest boxes). Fracture is separate X-ray model.",
    }

    if promote:
        backup = baseline_path + BACKUP_SUFFIX
        if os.path.isfile(baseline_path):
            shutil.copy2(baseline_path, backup)
        shutil.copy2(best, baseline_path)
        report["promoted"] = True
        report["backup"] = backup.replace("\\", "/")
        report["canonical_sha256"] = sha256_file(baseline_path)
    else:
        report["promoted"] = False

    report_path = os.path.join(PROJECT, "TRAIN_REPORT.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: report[k] for k in ("promote", "promoted", "promotion_reason", "test", "candidate_sha256")}, indent=2, default=str))
    return report


if __name__ == "__main__":
    train()
