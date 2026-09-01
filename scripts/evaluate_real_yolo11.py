"""
AI-QTriage — Real Dataset YOLO11 Held-Out Test Evaluation Script (Phase 5)
Evaluates the canonical runtime YOLO checkpoint
(ml/models/vision/yolo11_injury_best.pt) on the held-out test split
across confidence thresholds: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40.

The old filename yolo11_real_wound_best.pt lives in ml/models/_archive/
and is not loaded at runtime.
"""

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.models.canonical_paths import YOLO_CANONICAL

MODEL_PATH = os.path.join(ROOT_DIR, YOLO_CANONICAL)
TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "data", "datasets", "yolo_real_wound", "images", "test")
TEST_LABELS_DIR = os.path.join(ROOT_DIR, "data", "datasets", "yolo_real_wound", "labels", "test")
YAML_PATH = os.path.join(ROOT_DIR, "data", "datasets", "yolo_real_wound", "yolo_real.yaml")

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

def evaluate_held_out_test():
    print("=================================================================")
    print("AI-QTriage — HELD-OUT TEST SPLIT EVALUATION (PHASE 5)")
    print("=================================================================")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model checkpoint not found at {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    print(f"Loaded trained YOLO11 checkpoint: {MODEL_PATH}")
    print(f"Model Class Names: {model.names}")

    # Standard Ultralytics Validation on Held-Out Test Split
    print("\n[ULTRALYTICS BENCHMARK EVALUATION ON HELD-OUT TEST SET]")
    try:
        val_results = model.val(
            data=YAML_PATH,
            split="test",
            imgsz=640,
            batch=8,
            verbose=True
        )
        mp = val_results.results_dict.get("metrics/precision(B)", 0.0)
        mr = val_results.results_dict.get("metrics/recall(B)", 0.0)
        map50 = val_results.results_dict.get("metrics/mAP50(B)", 0.0)
        map5095 = val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)

        print(f"  - Test Precision (B) : {mp:.4f}")
        print(f"  - Test Recall (B)    : {mr:.4f}")
        print(f"  - Test mAP@50 (B)    : {map50:.4f}")
        print(f"  - Test mAP@50-95 (B) : {map5095:.4f}")
    except Exception as e:
        print(f"Notice: Ultralytics val failed: {e}")

    # Custom Threshold-by-Threshold Evaluation
    test_files = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")) + glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))
    print(f"\n[THRESHOLD SWEEP EVALUATION ({len(test_files)} HELD-OUT TEST IMAGES)]")

    results_table = []

    for conf_th in THRESHOLDS:
        tp, fp, fn = 0, 0, 0
        total_detections = 0

        for img_path in test_files:
            bname = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(TEST_LABELS_DIR, f"{bname}.txt")

            gt_boxes = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            gt_boxes.append((int(parts[0]), [float(x) for x in parts[1:]]))

            preds = model(img_path, conf=conf_th, verbose=False)[0].boxes
            pred_count = len(preds)
            total_detections += pred_count

            if len(gt_boxes) > 0 and pred_count > 0:
                tp += 1
            elif len(gt_boxes) > 0 and pred_count == 0:
                fn += 1
            elif len(gt_boxes) == 0 and pred_count > 0:
                fp += 1

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0

        results_table.append({
            "Confidence Threshold": conf_th,
            "Total Detections": total_detections,
            "True Positives (TP)": tp,
            "False Positives (FP)": fp,
            "False Negatives (FN)": fn,
            "Precision": precision,
            "Recall": recall
        })

    df_res = pd.DataFrame(results_table)
    print(df_res.to_string(index=False))

    print("=================================================================")
    print("HELD-OUT EVALUATION COMPLETE [OK]")
    print("=================================================================")

if __name__ == "__main__":
    evaluate_held_out_test()
