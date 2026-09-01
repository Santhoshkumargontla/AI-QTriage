"""
AI-QTriage — Real Dataset YOLO11 Training Script (Phase 4)
Trains a dedicated YOLO11 model on data/datasets/yolo_real_wound/yolo_real.yaml.
Preserves existing synthetic model yolo11n_best.pt as baseline.
Saves new trained weights to the canonical runtime path
ml/models/vision/yolo11_injury_best.pt (overwrites the live detector).
The previous filename yolo11_real_wound_best.pt is archived and unused.
"""

import os
import sys
import time
import yaml
import shutil
import torch
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.models.canonical_paths import YOLO_CANONICAL

DATASET_YAML = os.path.join(ROOT_DIR, "data", "datasets", "yolo_real_wound", "yolo_real.yaml")
OUTPUT_MODEL_PATH = os.path.join(ROOT_DIR, YOLO_CANONICAL)
BASE_WEIGHTS = os.path.join(ROOT_DIR, "ml", "models", "yolo11n_pretrained.pt")

def train_real_model():
    print("=================================================================")
    print("AI-QTriage — YOLO11 REAL WOUND MODEL TRAINING (PHASE 4)")
    print("=================================================================")

    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(f"Dataset config yolo_real.yaml not found at {DATASET_YAML}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Acceleration Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    base_model_path = BASE_WEIGHTS if os.path.exists(BASE_WEIGHTS) else "yolo11n.pt"
    print(f"Loading pretrained base checkpoint: {base_model_path}")
    model = YOLO(base_model_path)

    epochs = 25
    imgsz = 640
    batch_size = 8 if device == "cpu" else 16

    print(f"\n[TRAINING HYPERPARAMETERS]")
    print(f"  - Model Architecture : YOLO11 Nano (yolo11n.pt)")
    print(f"  - Dataset Config     : {DATASET_YAML}")
    print(f"  - Total Epochs       : {epochs}")
    print(f"  - Image Resolution   : {imgsz}x{imgsz}")
    print(f"  - Batch Size         : {batch_size}")
    print(f"  - Optimizer          : Auto (AdamW/SGD)")
    print(f"  - Hardware           : {device.upper()}")

    start_time = time.time()

    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=os.path.join(ROOT_DIR, "ml", "models", "yolo_real_training"),
        name="run_real_wound",
        exist_ok=True,
        verbose=True,
        seed=42
    )

    duration = round(time.time() - start_time, 2)
    print(f"\n[TRAINING COMPLETED]")
    print(f"  - Total Duration     : {duration} seconds")

    # Locate best trained weights
    best_weights_src = os.path.join(ROOT_DIR, "ml", "models", "yolo_real_training", "run_real_wound", "weights", "best.pt")
    if not os.path.exists(best_weights_src):
        best_weights_src = os.path.join(ROOT_DIR, "ml", "models", "yolo_real_training", "run_real_wound", "weights", "last.pt")

    if os.path.exists(best_weights_src):
        shutil.copy2(best_weights_src, OUTPUT_MODEL_PATH)
        print(f"\n[OK] Model successfully saved to: {OUTPUT_MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Trained weights file not found at {best_weights_src}")

    # Extract validation metrics from training results
    try:
        val_metrics = results.results_dict
        mp = val_metrics.get("metrics/precision(B)", 0.0)
        mr = val_metrics.get("metrics/recall(B)", 0.0)
        map50 = val_metrics.get("metrics/mAP50(B)", 0.0)
        map5095 = val_metrics.get("metrics/mAP50-95(B)", 0.0)

        print(f"\n[FINAL VALIDATION METRICS]")
        print(f"  - Precision (B)      : {mp:.4f}")
        print(f"  - Recall (B)         : {mr:.4f}")
        print(f"  - mAP@50 (B)         : {map50:.4f}")
        print(f"  - mAP@50-95 (B)      : {map5095:.4f}")
    except Exception as e:
        print(f"Notice: Could not parse results_dict directly ({e}). Weight file saved cleanly.")

    print("=================================================================")
    print("YOLO11 REAL MODEL TRAINING COMPLETE [OK]")
    print("=================================================================")

if __name__ == "__main__":
    train_real_model()
