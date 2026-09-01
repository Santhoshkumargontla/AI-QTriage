"""Generate model_registry.json and canonical_manifest.json from disk hashes + metadata.

Never invent metrics. SHA-256 is always computed from the live canonical file.
"""
import os
import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import (
    YOLO_CANONICAL,
    YOLO_METADATA,
    EFFNET_CANONICAL,
    UNET_CANONICAL,
    XGB_CANONICAL,
    VQC_DIR,
    VQC_WEIGHTS,
    SENSOR_MODEL,
    SENSOR_SCALER,
    REGISTRY_PATH,
    MANIFEST_PATH,
    RUNTIME_MODELS,
    exists,
    posix,
    read_json,
    resolve_existing,
    sha256_file,
)


def _sha256(path: str) -> str:
    return sha256_file(path)


def _load_json(path: str) -> dict:
    return read_json(path)


def _eval_for(name: str) -> str:
    for spec in RUNTIME_MODELS:
        if spec["model_name"] == name:
            return posix(spec["evaluation_artifact"])
    return ""


def _entry(name, version, rel_path, dataset, classes, command, notes, metadata_path=None, extra=None):
    if not exists(rel_path):
        print(f"Warning: missing artifact {rel_path}")
        return None
    located = resolve_existing(rel_path)
    meta = _load_json(metadata_path) if metadata_path else {}
    metrics = meta.get("metrics") or extra or {}
    status = meta.get("status") or ("TRAINED_AND_EVALUATED" if metrics else "MODEL_LOADS")
    return {
        "model_name": name,
        "version": meta.get("version") or version,
        "status": status,
        "canonical_path": posix(rel_path),
        "artifact_path": posix(rel_path),
        "artifact_sha256": _sha256(located),
        "artifact_size_mb": round(os.path.getsize(located) / (1024 * 1024), 2),
        "file_size": os.path.getsize(located),
        "schema_version": version,
        "classes": meta.get("classes") or classes,
        "training_status": meta.get("training_status") or status,
        "readiness_status": status,
        "evaluation_artifact": _eval_for(name),
        "dataset_type": metrics.get("dataset_type") or dataset,
        "training_dataset": metrics.get("dataset_name") or dataset,
        "dataset_provenance": meta.get("dataset_provenance") or dataset,
        "data_provenance_class": meta.get("data_provenance_class"),
        "evaluation_status": "evaluated" if metrics else "not_evaluated_in_metadata",
        "last_evaluated": metrics.get("trained_at") or datetime.now(timezone.utc).isoformat(),
        "sample_count": metrics.get("train_samples") or metrics.get("test_samples"),
        "training_command": command,
        "metrics": metrics,
        "known_limitations": meta.get("known_limitations") or notes,
        "notes": notes,
        "training_was_real": bool(meta.get("training_was_real", False)),
    }


def _yolo_runtime_entry():
    """Registry YOLO row must match the exact file YOLO11Detector loads."""
    from ml.vision.yolo_wrapper import YOLO11Detector, UNTRAINED_CLASS

    rel_path = posix(YOLO_CANONICAL)
    meta = _load_json(YOLO_METADATA)
    if not exists(YOLO_CANONICAL):
        print(f"MODEL_ARTIFACT_MISSING: {rel_path}")
        return {
            "model_name": "YOLO11 Detection",
            "version": meta.get("version") or "v1.3.0",
            "status": "MODEL_ARTIFACT_MISSING",
            "canonical_path": rel_path,
            "artifact_path": rel_path,
            "artifact_sha256": None,
            "task": None,
            "classes": [],
            "untrained_class_status": UNTRAINED_CLASS,
            "evaluation_artifact": _eval_for("YOLO11 Detection"),
            "readiness_status": "MODEL_ARTIFACT_MISSING",
            "notes": "Runtime YOLO canonical file is missing. No fallback checkpoint is used.",
        }

    detector = YOLO11Detector()
    info = detector.get_info()
    live_classes = list(info.get("classes") or info.get("supported_classes") or [])
    located = resolve_existing(YOLO_CANONICAL)
    return {
        "model_name": "YOLO11 Detection",
        "version": info.get("version") or meta.get("version") or "v1.3.0",
        "status": info.get("status") or meta.get("status") or "INFERENCE_EXECUTES",
        "canonical_path": rel_path,
        "artifact_path": rel_path,
        "artifact_sha256": info.get("artifact_sha256") or sha256_file(located),
        "artifact_size_mb": round(os.path.getsize(located) / (1024 * 1024), 2),
        "file_size": os.path.getsize(located),
        "schema_version": info.get("version") or "v1.3.0",
        "task": info.get("task"),
        "classes": live_classes,
        "untrained_class_status": UNTRAINED_CLASS,
        "training_status": meta.get("training_status") or meta.get("status") or info.get("status"),
        "readiness_status": info.get("status") or meta.get("status") or "INFERENCE_EXECUTES",
        "evaluation_artifact": _eval_for("YOLO11 Detection"),
        "dataset_type": (meta.get("metrics") or {}).get("dataset_type") or "yolo_merged_or_real_wound",
        "training_dataset": (meta.get("metrics") or {}).get("dataset_name") or "yolo_merged_or_real_wound",
        "evaluation_status": "evaluated" if meta.get("metrics") else "not_evaluated_in_metadata",
        "last_evaluated": (meta.get("metrics") or {}).get("trained_at") or datetime.now(timezone.utc).isoformat(),
        "sample_count": (meta.get("metrics") or {}).get("train_samples"),
        "training_command": r"backend\venv\Scripts\python.exe ml\training\train_yolo_retrain_v2.py",
        "metrics": meta.get("metrics") or {},
        "notes": (
            "Single runtime checkpoint YOLO_CANONICAL. Classes are model.names. "
            "Any class not in model.names is UNTRAINED_CLASS. Backups are not loaded."
        ),
        "known_limitations": meta.get("known_limitations")
            or "Small val set. Not clinical. Corner boxes on demo graphics are detection quality, not CSS scale errors.",
        "data_provenance_class": meta.get("data_provenance_class") or "MIXED",
        "dataset_provenance": meta.get("dataset_provenance"),
        "training_was_real": bool(meta.get("training_was_real", False)),
    }


def write_canonical_manifest(registry: dict) -> dict:
    """Lean catalog: name, path, SHA-256, version, classes, training, eval, readiness."""
    models = []
    for spec in RUNTIME_MODELS:
        name = spec["model_name"]
        entry = registry.get(name) or {}
        path = spec["canonical_path"]
        sidecar = spec.get("sidecar_path")
        row = {
            "model_name": name,
            "canonical_path": posix(path),
            "sha256": entry.get("artifact_sha256") or (_sha256(path) if exists(path) else None),
            "version": entry.get("version"),
            "classes": entry.get("classes"),
            "training_status": entry.get("training_status") or entry.get("status"),
            "evaluation_artifact": posix(spec["evaluation_artifact"]),
            "readiness_status": entry.get("readiness_status") or entry.get("status"),
            "wrapper": spec["wrapper"],
            "used_by_runtime": True,
        }
        if sidecar:
            row["sidecar_path"] = posix(sidecar)
            row["sidecar_sha256"] = _sha256(sidecar) if exists(sidecar) else None
        extra_sidecars = spec.get("sidecar_paths") or ()
        if extra_sidecars:
            row["sidecars"] = [
                {
                    "path": posix(path),
                    "sha256": _sha256(path) if exists(path) else None,
                }
                for path in extra_sidecars
            ]
        models.append(row)
    manifest = {
        "source": "ml.models.canonical_paths.RUNTIME_MODELS + live SHA-256",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
    }
    dest = resolve_existing(MANIFEST_PATH)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[OK] Manifest written to {posix(MANIFEST_PATH)}")
    return manifest


def generate_model_registry():
    registry = {}
    specs = [
        _entry(
            "XGBoost Multimodal",
            "v1.3.0",
            posix(XGB_CANONICAL),
            "synthetic_multimodal_fusion",
            ["LOW", "MODERATE", "HIGH"],
            r"backend\venv\Scripts\python.exe ml\training\train_xgboost.py",
            "DATA_PROVENANCE=SYNTHETIC. 23-feature synthetic fusion model. READY_FOR_RESEARCH_DEMO, not clinical. 0 genuinely paired clinical samples.",
            os.path.join("ml", "models", "xgboost_metadata.json"),
        ),
        _entry(
            "Experimental 4-Qubit VQC",
            "v1.3.0",
            posix(VQC_WEIGHTS),
            "synthetic_multimodal_fusion",
            ["LOW", "MODERATE", "HIGH"],
            r"backend\venv\Scripts\python.exe ml\training\train_vqc.py",
            "EXPERIMENTAL_ONLY. Isolated from main decision. Training circuit equals inference circuit. DATA_PROVENANCE=SYNTHETIC.",
            os.path.join(VQC_DIR, "vqc_metadata.json"),
        ),
        _entry(
            "EfficientNetV2 Classification",
            "kaggle-v1",
            posix(EFFNET_CANONICAL),
            "kaggle_multi_class_wound_photos",
            [
                "abrasion",
                "bruise",
                "burn",
                "cut",
                "laceration",
                "wound",
                "normal",
                "ood_reject",
            ],
            r"backend\venv\Scripts\python.exe ml\training\train_efficientnet_kaggle_v1.py",
            "Fallback classes match kaggle-v1. Live classes/metrics come from efficientnetv2_metadata.json. "
            "Do not use ml/training/train_efficientnet.py (drawings) to overwrite this checkpoint.",
            os.path.join("ml", "models", "vision", "efficientnetv2_metadata.json"),
        ),
        _entry(
            "ResNet34-UNet Segmentation",
            "deduped-subject-v1",
            posix(UNET_CANONICAL),
            "unet_deduped_subject",
            ["wound_mask"],
            r"backend\venv\Scripts\python.exe ml\training\train_unet.py",
            "Status and metrics come from unet_metadata.json after real training.",
            os.path.join("ml", "models", "vision", "unet_metadata.json"),
        ),
        _yolo_runtime_entry(),
        _entry(
            "Sensor Motion Event Classifier",
            "v2.0.0-real",
            posix(SENSOR_MODEL),
            "sisfall_uci_har_real",
            ["normal_activity", "fall", "impact"],
            r"backend\venv\Scripts\python.exe ml\training\train_sensor_model.py",
            "REAL SisFall+UCI HAR. Missing features are reported, never invented. Demo CSV remains simulated.",
            os.path.join("ml", "models", "sensor_metadata.json"),
        ),
    ]
    for item in specs:
        if item:
            registry[item["model_name"]] = item
    sensor = registry.get("Sensor Motion Event Classifier")
    if sensor and exists(SENSOR_SCALER):
        sensor["scaler_path"] = posix(SENSOR_SCALER)
        sensor["scaler_sha256"] = _sha256(SENSOR_SCALER)
    dest = resolve_existing(REGISTRY_PATH)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)
    print(f"[OK] Registry written to {posix(REGISTRY_PATH)}")
    write_canonical_manifest(registry)
    return registry


if __name__ == "__main__":
    generate_model_registry()
