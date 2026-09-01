"""Single source of truth for runtime model artifacts.

Path constants are project-relative so registry JSON stays portable.
All disk I/O must go through abs_path / exists / resolve_existing / sha256_file
so loading does not depend on the process current working directory.
"""
import hashlib
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# The only production/runtime YOLO checkpoint. YOLO11Detector never searches
# candidate/backup paths and never honors YOLO_MODEL_VERSION for file selection.
YOLO_CANONICAL = os.path.join("ml", "models", "vision", "yolo11_injury_best.pt")
YOLO_RUNTIME = YOLO_CANONICAL
YOLO_SYNTHETIC_BASELINE = os.path.join("ml", "models", "yolo11n_best.pt")
YOLO_PRETRAINED_INIT = os.path.join("ml", "models", "yolo11n_pretrained.pt")

# Historical copies. Not loaded at runtime. Byte-duplicates live under ml/models/_archive/.
YOLO_BASELINE_COPY = os.path.join("ml", "models", "yolo_real_training", "run_real_wound", "weights", "best.pt")
YOLO_BACKUP_PATHS = (
    YOLO_CANONICAL + ".pre_retrain_v2_backup",
    YOLO_CANONICAL + ".pre_retrain_backup",
    YOLO_BASELINE_COPY,
    YOLO_SYNTHETIC_BASELINE,
    YOLO_PRETRAINED_INIT,
    os.path.join("ml", "models", "vision", "yolo11_injury_real_retrained.pt"),
    os.path.join("ml", "models", "yolo_processed_training", "run_processed", "weights", "best.pt"),
)
EFFNET_CANONICAL = os.path.join("ml", "models", "vision", "efficientnetv2_injury_best.pt")
UNET_CANONICAL = os.path.join("ml", "models", "vision", "unet_injury_best.pt")
XGB_CANONICAL = os.path.join("ml", "models", "xgboost_best.json")
VQC_DIR = os.path.join("ml", "models", "vqc")
VQC_WEIGHTS = os.path.join(VQC_DIR, "vqc_weights.npz")
VQC_SCALER = os.path.join(VQC_DIR, "scaler.pkl")
VQC_PCA = os.path.join(VQC_DIR, "pca.pkl")
SENSOR_MODEL = os.path.join("ml", "models", "sensor_motion_best.json")
SENSOR_SCALER = os.path.join("ml", "models", "sensor_scaler.pkl")

# Ultralytics output of the promoted YOLO retrain. Byte-identical to YOLO_CANONICAL.
# Wrappers never load this path; keep it as the training-run record.
YOLO_RETRAIN_V2_BEST = os.path.join("ml", "models", "yolo_retrain_v2", "run_v2", "weights", "best.pt")

YOLO_METADATA = os.path.join("ml", "models", "vision", "yolo11_metadata.json")
EFFNET_METADATA = os.path.join("ml", "models", "vision", "efficientnetv2_metadata.json")
UNET_METADATA = os.path.join("ml", "models", "vision", "unet_metadata.json")
XGB_METADATA = os.path.join("ml", "models", "xgboost_metadata.json")
VQC_METADATA = os.path.join(VQC_DIR, "vqc_metadata.json")
SENSOR_METADATA = os.path.join("ml", "models", "sensor_metadata.json")

EVAL_HELD_OUT = os.path.join("data", "results", "canonical_held_out_evaluation.json")
EVAL_RESULTS = os.path.join("data", "results", "evaluation_results.json")
EVAL_COMPARE = os.path.join("data", "results", "vqc_xgb_comparison.json")
EVAL_YOLO_THRESHOLD = os.path.join("ml", "models", "yolo_threshold_eval", "THRESHOLD_SWEEP_REPORT.json")
EVAL_EFFNET_RELIABILITY = os.path.join("ml", "models", "efficientnet_reliability", "EFFICIENTNET_RELIABILITY_REPORT.json")
EVAL_UNET_RELIABILITY = os.path.join("ml", "models", "unet_reliability", "UNET_RELIABILITY_REPORT.json")

# Clean processed YOLO detect set. Raw sources are never modified.
# abrasion / laceration / swelling are dropped here, not remapped to wound.
YOLO_PROCESSED_ROOT = os.path.join("data", "datasets", "yolo_processed")
YOLO_PROCESSED_YAML = os.path.join(YOLO_PROCESSED_ROOT, "data.yaml")
YOLO_RETRAIN_V2_ROOT = os.path.join("data", "datasets", "yolo_retrain_v2")
YOLO_RETRAIN_V2_YAML = os.path.join(YOLO_RETRAIN_V2_ROOT, "data.yaml")

# Deduped classification set for EfficientNet. Raw sources are not modified.
EFFNET_PROCESSED_ROOT = os.path.join("data", "datasets", "efficientnet_processed")
EFFNET_PROCESSED_MANIFEST = os.path.join(EFFNET_PROCESSED_ROOT, "manifest.csv")
EFFNET_CANDIDATE_DIR = os.path.join("ml", "models", "efficientnet_processed_training")
EFFNET_CANDIDATE = os.path.join(EFFNET_CANDIDATE_DIR, "efficientnetv2_candidate.pt")

# Deduped image/mask set for U-Net. Raw sources are not modified.
UNET_PROCESSED_ROOT = os.path.join("data", "datasets", "unet_processed")
UNET_PROCESSED_MANIFEST = os.path.join(UNET_PROCESSED_ROOT, "manifest.csv")
UNET_CANDIDATE_DIR = os.path.join("ml", "models", "unet_processed_training")
UNET_CANDIDATE = os.path.join(UNET_CANDIDATE_DIR, "unet_candidate.pt")

# Public real-photo U-Net set (wseg CC-BY-NC + Medetec) plus documented empty synthetics.
UNET_PUBLIC_ROOT = os.path.join("data", "datasets", "unet_public_real")
UNET_PUBLIC_MANIFEST = os.path.join(UNET_PUBLIC_ROOT, "manifest.csv")
UNET_PUBLIC_CANDIDATE_DIR = os.path.join("ml", "models", "unet_public_training")
UNET_PUBLIC_CANDIDATE = os.path.join(UNET_PUBLIC_CANDIDATE_DIR, "unet_candidate.pt")

# EfficientNet cut/bruise drawings + documented synthetic normal/reject class.
EFFNET_NORMAL_ROOT = os.path.join("data", "datasets", "efficientnet_with_normal")
EFFNET_NORMAL_MANIFEST = os.path.join(EFFNET_NORMAL_ROOT, "manifest.csv")
EFFNET_NORMAL_CANDIDATE_DIR = os.path.join("ml", "models", "efficientnet_normal_training")
EFFNET_NORMAL_CANDIDATE = os.path.join(EFFNET_NORMAL_CANDIDATE_DIR, "efficientnetv2_candidate.pt")

# Subject-aware EfficientNet set: synthetic cut/bruise + REAL normal patches.
EFFNET_SUBJECT_ROOT = os.path.join("data", "datasets", "efficientnet_subject_normal")
EFFNET_SUBJECT_MANIFEST = os.path.join(EFFNET_SUBJECT_ROOT, "manifest.csv")
EFFNET_SUBJECT_CANDIDATE_DIR = os.path.join("ml", "models", "efficientnet_subject_training")
EFFNET_SUBJECT_CANDIDATE = os.path.join(EFFNET_SUBJECT_CANDIDATE_DIR, "efficientnetv2_candidate.pt")

# Deduped subject-aware U-Net set (AZH + wseg + Medetec + synthetic empty).
UNET_DEDUPED_ROOT = os.path.join("data", "datasets", "unet_deduped_subject")
UNET_DEDUPED_MANIFEST = os.path.join(UNET_DEDUPED_ROOT, "manifest.csv")
UNET_DEDUPED_CANDIDATE_DIR = os.path.join("ml", "models", "unet_deduped_training")
UNET_DEDUPED_CANDIDATE = os.path.join(UNET_DEDUPED_CANDIDATE_DIR, "unet_candidate.pt")

# YOLO with public wound boxes + legacy cut/bruise drawings.
YOLO_WOUND_BOXES_ROOT = os.path.join("data", "datasets", "yolo_wound_boxes_v1")
YOLO_WOUND_BOXES_YAML = os.path.join(YOLO_WOUND_BOXES_ROOT, "data.yaml")
YOLO_WOUND_BOXES_PROJECT = os.path.join("ml", "models", "yolo_wound_boxes_v1")

REGISTRY_PATH = os.path.join("ml", "models", "model_registry.json")
MANIFEST_PATH = os.path.join("ml", "models", "canonical_manifest.json")


def posix(path: str) -> str:
    return str(path).replace("\\", "/")


def abs_path(rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.abspath(os.path.join(ROOT, rel))


def exists(path: str) -> bool:
    """True if the artifact exists under ROOT, or as an absolute/cwd path."""
    if not path:
        return False
    if os.path.isabs(path):
        return os.path.exists(path)
    if os.path.exists(abs_path(path)):
        return True
    return os.path.exists(path)


def resolve_existing(path: str) -> str:
    """Resolve an artifact against the project root first, then cwd.

    Prefers ROOT so a leftover file under a random cwd cannot shadow the
    canonical checkpoint. Returns the ROOT-absolute path when the relative
    file is missing, so error messages point at the expected location.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    rooted = abs_path(path)
    if os.path.exists(rooted):
        return rooted
    if os.path.exists(path):
        return os.path.abspath(path)
    return rooted


def sha256_file(path: str) -> str:
    located = resolve_existing(path)
    digest = hashlib.sha256()
    with open(located, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str) -> dict:
    located = resolve_existing(path)
    if not os.path.exists(located):
        return {}
    try:
        with open(located, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


# Live catalog used by generate_registry / canonical_manifest.json.
# evaluation_artifact is the metrics file that must describe THIS checkpoint.
RUNTIME_MODELS = (
    {
        "model_name": "YOLO11 Detection",
        "canonical_path": YOLO_CANONICAL,
        "metadata_path": YOLO_METADATA,
        "evaluation_artifact": EVAL_YOLO_THRESHOLD,
        "wrapper": "ml.vision.yolo_wrapper.YOLO11Detector",
    },
    {
        "model_name": "EfficientNetV2 Classification",
        "canonical_path": EFFNET_CANONICAL,
        "metadata_path": EFFNET_METADATA,
        "evaluation_artifact": EFFNET_METADATA,
        "wrapper": "ml.vision.efficientnet_wrapper.EfficientNetV2Classifier",
    },
    {
        "model_name": "ResNet34-UNet Segmentation",
        "canonical_path": UNET_CANONICAL,
        "metadata_path": UNET_METADATA,
        "evaluation_artifact": UNET_METADATA,
        "wrapper": "ml.vision.unet_wrapper.UNetSegmenter",
    },
    {
        "model_name": "XGBoost Multimodal",
        "canonical_path": XGB_CANONICAL,
        "metadata_path": XGB_METADATA,
        "evaluation_artifact": EVAL_HELD_OUT,
        "wrapper": "ml.classifiers.xgboost_classifier.XGBoostClassifier",
    },
    {
        "model_name": "Experimental 4-Qubit VQC",
        "canonical_path": VQC_WEIGHTS,
        "metadata_path": VQC_METADATA,
        "evaluation_artifact": EVAL_HELD_OUT,
        "wrapper": "ml.classifiers.vqc_classifier.VQCClassifier",
        "sidecar_paths": (VQC_SCALER, VQC_PCA),
    },
    {
        "model_name": "Sensor Motion Event Classifier",
        "canonical_path": SENSOR_MODEL,
        "metadata_path": SENSOR_METADATA,
        "evaluation_artifact": SENSOR_METADATA,
        "wrapper": "ml.classifiers.sensor_classifier.SensorClassifier",
        "sidecar_path": SENSOR_SCALER,
    },
)
