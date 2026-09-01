"""One-shot baseline hasher. Does not modify weights."""
import os
import sys
import json
import hashlib
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from ml.models.canonical_paths import (
    YOLO_CANONICAL,
    EFFNET_CANONICAL,
    UNET_CANONICAL,
    XGB_CANONICAL,
    VQC_DIR,
    SENSOR_MODEL,
    SENSOR_SCALER,
)
from ml.vision.yolo_wrapper import YOLO11Detector


def sha256(path):
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row(name, path, runtime=True):
    exists = os.path.exists(path)
    return {
        "name": name,
        "path": path.replace("\\", "/"),
        "exists": exists,
        "size_bytes": os.path.getsize(path) if exists else None,
        "sha256": sha256(path) if exists else None,
        "used_at_runtime_by_default": runtime,
    }


def main():
    detector = YOLO11Detector()
    info = detector.get_info()
    runtime = [
        row("YOLO11", info.get("model_path") or YOLO_CANONICAL),
        row("EfficientNetV2", EFFNET_CANONICAL),
        row("U-Net", UNET_CANONICAL),
        row("XGBoost", XGB_CANONICAL),
        row("VQC weights", os.path.join(VQC_DIR, "vqc_weights.npz")),
        row("VQC scaler", os.path.join(VQC_DIR, "scaler.pkl")),
        row("VQC pca", os.path.join(VQC_DIR, "pca.pkl")),
        row("Sensor model", SENSOR_MODEL),
        row("Sensor scaler", SENSOR_SCALER),
    ]
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "yolo_canonical_constant": YOLO_CANONICAL.replace("\\", "/"),
        "yolo_wrapper_loaded_path": (info.get("model_path") or "").replace("\\", "/"),
        "yolo_wrapper_info": {
            key: info.get(key)
            for key in (
                "model_path",
                "ckpt_path",
                "task",
                "model_names",
                "status",
                "model_version",
                "supported_classes",
                "untrained_classes",
            )
        },
        "yolo_env": os.environ.get("YOLO_MODEL_VERSION"),
        "runtime_hashes": runtime,
    }
    out_path = os.path.join(os.path.dirname(__file__), "runtime_model_hashes.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    for item in runtime:
        print(item["name"])
        print("  path=", item["path"])
        print("  exists=", item["exists"], "size=", item["size_bytes"])
        print("  sha256=", item["sha256"])
    print("YOLO_CANONICAL", YOLO_CANONICAL)
    print("wrapper_path", info.get("model_path"))
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
