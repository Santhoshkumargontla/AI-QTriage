"""Single source of truth for runtime model artifacts."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

YOLO_CANONICAL = os.path.join("ml", "models", "vision", "yolo11_injury_best.pt")
YOLO_BASELINE_COPY = os.path.join("ml", "models", "yolo_real_training", "run_real_wound", "weights", "best.pt")
YOLO_CANDIDATE = os.path.join("ml", "models", "vision", "yolo11_merged_candidate.pt")
EFFNET_CANONICAL = os.path.join("ml", "models", "vision", "efficientnetv2_injury_best.pt")
UNET_CANONICAL = os.path.join("ml", "models", "vision", "unet_injury_best.pt")
XGB_CANONICAL = os.path.join("ml", "models", "xgboost_best.json")
VQC_DIR = os.path.join("ml", "models", "vqc")
SENSOR_MODEL = os.path.join("ml", "models", "sensor_motion_best.json")
SENSOR_SCALER = os.path.join("ml", "models", "sensor_scaler.pkl")

REGISTRY_PATH = os.path.join("ml", "models", "model_registry.json")


def abs_path(rel: str) -> str:
    return os.path.abspath(rel)
