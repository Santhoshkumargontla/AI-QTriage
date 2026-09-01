"""Train fracture X-ray YOLO detector. Saves under ml/models/vision/ — NOT skin canonical."""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import YOLO_PRETRAINED_INIT, ROOT, abs_path, sha256_file
from ml.training.prepare_yolo_fracture_xray_v1 import prepare
from ml.training.train_yolo_processed import extract_metrics

PROJECT = os.path.join("ml", "models", "yolo_fracture_xray_v1")
RUN_NAME = "run_v1"
DATA_YAML = os.path.join("data", "datasets", "yolo_fracture_xray_v1", "data.yaml")
ARTIFACT = os.path.join("ml", "models", "vision", "yolo11_fracture_xray_best.pt")

import torch as _torch

FRAC_EPOCHS = int(os.environ.get("YOLO_FRAC_EPOCHS", "30" if _torch.cuda.is_available() else "10"))
FRAC_BATCH = int(os.environ.get("YOLO_FRAC_BATCH", "8" if _torch.cuda.is_available() else "4"))
FRAC_PATIENCE = int(os.environ.get("YOLO_FRAC_PATIENCE", "8" if _torch.cuda.is_available() else "4"))


def train():
    os.makedirs(PROJECT, exist_ok=True)
    prep_path = os.path.join("data", "datasets", "yolo_fracture_xray_v1", "PREPARE_REPORT.json")
    if os.path.isfile(os.path.abspath(DATA_YAML)) and os.path.isfile(prep_path):
        with open(prep_path, encoding="utf-8") as handle:
            prep = json.load(handle)
        print(f"reusing fracture dataset: {prep.get('split_image_counts')}", flush=True)
    else:
        prep = prepare()
    from ultralytics import YOLO

    init = abs_path(YOLO_PRETRAINED_INIT) if os.path.isfile(abs_path(YOLO_PRETRAINED_INIT)) else "yolo11n.pt"
    model = YOLO(init)
    print(f"training fracture xray epochs={FRAC_EPOCHS} batch={FRAC_BATCH}", flush=True)
    frac = float(os.environ.get("YOLO_FRAC_FRACTION", "1.0" if _torch.cuda.is_available() else "0.25"))
    model.train(
        data=os.path.abspath(DATA_YAML),
        epochs=FRAC_EPOCHS,
        imgsz=640,
        batch=FRAC_BATCH,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        workers=0,
        patience=FRAC_PATIENCE,
        seed=42,
        fraction=frac,
    )
    best = os.path.join(PROJECT, RUN_NAME, "weights", "best.pt")
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir:
        alt = os.path.join(str(save_dir), "weights", "best.pt")
        if os.path.isfile(alt):
            os.makedirs(os.path.dirname(best), exist_ok=True)
            if os.path.abspath(alt) != os.path.abspath(best):
                shutil.copy2(alt, best)
            best = best if os.path.isfile(best) else alt
    if not os.path.isfile(best):
        raise FileNotFoundError(f"fracture best weights missing: {best}")
    names = list((prep.get("names") or []))
    expected = {i: str(n) for i, n in enumerate(names)}
    test_m = extract_metrics(
        YOLO(best).val(
            data=os.path.abspath(DATA_YAML),
            split="test",
            workers=0,
            project=PROJECT,
            name="test_eval",
            exist_ok=True,
        ),
        expected,
    )
    os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
    shutil.copy2(best, ARTIFACT)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prepare": prep,
        "artifact": ARTIFACT.replace("\\", "/"),
        "artifact_sha256": sha256_file(ARTIFACT),
        "test": test_m,
        "train_fraction": frac,
        "epochs": FRAC_EPOCHS,
        "modality": "XRAY",
        "promoted_to_skin_canonical": False,
        "notes": "Research X-ray fracture detector only. Not wired into skin photo analyze path. CPU runs may use fraction<1.",
    }
    with open(os.path.join(PROJECT, "TRAIN_REPORT.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    train()
