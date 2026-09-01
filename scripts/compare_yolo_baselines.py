"""
AI-QTriage — Baseline Comparison Script (Phase 6)

Compares the synthetic baseline (yolo11n_best.pt, tests only) against the
canonical runtime checkpoint (ml/models/vision/yolo11_injury_best.pt).

The old filename yolo11_real_wound_best.pt was archived to
ml/models/_archive/ and is not loaded at runtime.
"""

import os
import glob
import pandas as pd
from ultralytics import YOLO
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.models.canonical_paths import YOLO_CANONICAL

SYNTHETIC_MODEL_PATH = os.path.join(ROOT_DIR, "ml", "models", "yolo11n_best.pt")
REAL_MODEL_PATH = os.path.join(ROOT_DIR, YOLO_CANONICAL)
BLANK_IMAGE_PATH = os.path.join(ROOT_DIR, "data", "datasets", "yolo_injury", "blank_skin.jpg")

def compare_baselines():
    print("=================================================================")
    print("AI-QTriage — BASELINE MODEL COMPARISON (PHASE 6)")
    print("=================================================================")

    if not os.path.exists(SYNTHETIC_MODEL_PATH):
        print(f"Warning: Synthetic model not found at {SYNTHETIC_MODEL_PATH}")
        synth_model = None
    else:
        synth_model = YOLO(SYNTHETIC_MODEL_PATH)
        print(f"[OK] Synthetic Baseline Loaded: {SYNTHETIC_MODEL_PATH}")

    if not os.path.exists(REAL_MODEL_PATH):
        print(f"Warning: Real model not found at {REAL_MODEL_PATH}")
        real_model = None
    else:
        real_model = YOLO(REAL_MODEL_PATH)
        print(f"[OK] Real Experimental Model Loaded: {REAL_MODEL_PATH}")

    # Blank image false positive check
    synth_blank_dets = 0
    real_blank_dets = 0

    if os.path.exists(BLANK_IMAGE_PATH):
        if synth_model:
            synth_blank_dets = len(synth_model(BLANK_IMAGE_PATH, conf=0.20, verbose=False)[0].boxes)
        if real_model:
            real_blank_dets = len(real_model(BLANK_IMAGE_PATH, conf=0.20, verbose=False)[0].boxes)

    comparison_data = [
        {
            "Attribute / Metric": "Model Path",
            "Synthetic Baseline": "ml/models/yolo11n_best.pt",
            "Real-Data Experimental": YOLO_CANONICAL.replace("\\", "/")
        },
        {
            "Attribute / Metric": "Training Data Provenance",
            "Synthetic Baseline": "Synthetic Wound Generation",
            "Real-Data Experimental": "Real Patient Photographic Wounds (WOUNDSEG/Roboflow)"
        },
        {
            "Attribute / Metric": "Unique Clean Training Images",
            "Synthetic Baseline": "59 images",
            "Real-Data Experimental": "38 unique images (33 subjects)"
        },
        {
            "Attribute / Metric": "Supported Classes",
            "Synthetic Baseline": "cut, bruise, abrasion, laceration",
            "Real-Data Experimental": "cut, bruise, wound"
        },
        {
            "Attribute / Metric": "Blank Control False Positives",
            "Synthetic Baseline": f"{synth_blank_dets} detections",
            "Real-Data Experimental": f"{real_blank_dets} detections"
        },
        {
            "Attribute / Metric": "Clinically Validated",
            "Synthetic Baseline": "FALSE (Research Prototype)",
            "Real-Data Experimental": "FALSE (Research Prototype)"
        }
    ]

    df_comp = pd.DataFrame(comparison_data)
    print("\n[SIDE-BY-SIDE COMPARISON TABLE]")
    print(df_comp.to_string(index=False))

    print("=================================================================")
    print("BASELINE COMPARISON COMPLETE [OK]")
    print("=================================================================")

if __name__ == "__main__":
    compare_baselines()
