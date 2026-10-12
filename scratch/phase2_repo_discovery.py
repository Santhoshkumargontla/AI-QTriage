import os
import sys
import json
import hashlib
from datetime import datetime, timezone

ROOT = os.path.abspath(".")

def get_file_info(rel_path):
    p = os.path.join(ROOT, rel_path)
    if not os.path.exists(p):
        return {"exists": False, "path": rel_path}
    size = os.path.getsize(p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return {
        "exists": True,
        "path": rel_path.replace("\\", "/"),
        "size_bytes": size,
        "sha256": h.hexdigest(),
        "modified_iso": datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).isoformat()
    }

# 1. Models
active_models = [
    {
        "name": "YOLO11 Detection",
        "canonical_path": "ml/models/vision/yolo11_injury_best.pt",
        "wrapper": "ml.vision.yolo_wrapper.YOLO11Detector",
        "training_script": "ml/training/train_yolo_retrain_v2.py",
        "evaluation_artifact": "ml/models/yolo_threshold_eval/THRESHOLD_SWEEP_REPORT.json",
        "metadata_path": "ml/models/vision/yolo11_metadata.json",
        "task": "Bounding Box Object Detection (6 injury classes: abrasion, bruise, burn, cut, laceration, wound)",
        "status": "READY_FOR_RESEARCH_DEMO"
    },
    {
        "name": "EfficientNetV2 Classification",
        "canonical_path": "ml/models/vision/efficientnetv2_injury_best.pt",
        "wrapper": "ml.vision.efficientnet_wrapper.EfficientNetV2Classifier",
        "training_script": "ml/training/train_efficientnet_kaggle_v1.py",
        "evaluation_artifact": "ml/models/vision/efficientnetv2_metadata.json",
        "metadata_path": "ml/models/vision/efficientnetv2_metadata.json",
        "task": "Multi-class Image Classification (8 classes: abrasion, bruise, burn, cut, laceration, wound, normal, ood_reject)",
        "status": "READY_FOR_RESEARCH_DEMO"
    },
    {
        "name": "ResNet34-UNet Segmentation",
        "canonical_path": "ml/models/vision/unet_injury_best.pt",
        "wrapper": "ml.vision.unet_wrapper.UNetSegmenter",
        "training_script": "ml/training/train_unet.py",
        "evaluation_artifact": "ml/models/vision/unet_metadata.json",
        "metadata_path": "ml/models/vision/unet_metadata.json",
        "task": "Binary Wound Pixel Segmentation (wound mask vs background)",
        "status": "READY_FOR_RESEARCH_DEMO"
    },
    {
        "name": "Sensor Motion Event Classifier",
        "canonical_path": "ml/models/sensor_motion_best.json",
        "wrapper": "ml.classifiers.sensor_classifier.SensorClassifier",
        "training_script": "ml/training/train_sensor_model.py",
        "evaluation_artifact": "ml/models/sensor_metadata.json",
        "metadata_path": "ml/models/sensor_metadata.json",
        "task": "Kinematic Motion Event Classification (normal_activity, fall, impact)",
        "sidecar_artifacts": ["ml/models/sensor_scaler.pkl"],
        "status": "TRAINED"
    },
    {
        "name": "XGBoost Multimodal Classifier",
        "canonical_path": "ml/models/xgboost_best.json",
        "wrapper": "ml.classifiers.xgboost_classifier.XGBoostClassifier",
        "training_script": "ml/training/train_xgboost.py",
        "evaluation_artifact": "data/results/canonical_held_out_evaluation_corrected.json",
        "metadata_path": "ml/models/xgboost_metadata.json",
        "task": "Multimodal Severity Triage (LOW, MODERATE, HIGH) across 23 features",
        "status": "TRAINED"
    },
    {
        "name": "Experimental 4-Qubit VQC",
        "canonical_path": "ml/models/vqc/vqc_weights.npz",
        "wrapper": "ml.classifiers.vqc_classifier.VQCClassifier",
        "training_script": "ml/training/train_vqc.py",
        "evaluation_artifact": "data/results/canonical_held_out_evaluation_corrected.json",
        "metadata_path": "ml/models/vqc/vqc_metadata.json",
        "task": "QML Variational Quantum Classifier (AngleEmbedding + StronglyEntanglingLayers) on classical simulator",
        "sidecar_artifacts": ["ml/models/vqc/scaler.pkl", "ml/models/vqc/pca.pkl"],
        "status": "EXPERIMENTAL_ONLY"
    }
]

models_inventory = {}
for m in active_models:
    info = get_file_info(m["canonical_path"])
    m_data = {**m, "file_info": info}
    if "sidecar_artifacts" in m:
        m_data["sidecar_info"] = [get_file_info(p) for p in m["sidecar_artifacts"]]
    models_inventory[m["name"]] = m_data

# Datasets
datasets_inventory = [
    {
        "name": "yolo_expanded_v1",
        "path": "data/datasets/yolo_expanded_v1",
        "manifest": "data/datasets/yolo_expanded_v1/PREPARE_REPORT.json",
        "yaml": "data/datasets/yolo_expanded_v1/data.yaml",
        "task": "YOLO11 Detection"
    },
    {
        "name": "efficientnet_kaggle_v1",
        "path": "data/datasets/efficientnet_kaggle_v1",
        "manifest": "data/datasets/efficientnet_kaggle_v1/manifest.csv",
        "task": "EfficientNetV2 Classification"
    },
    {
        "name": "unet_deduped_subject",
        "path": "data/datasets/unet_deduped_subject",
        "manifest": "data/datasets/unet_deduped_subject/manifest.csv",
        "task": "ResNet34-UNet Segmentation"
    },
    {
        "name": "sisfall_uci_har_real",
        "path": "data/datasets/manifests/sensor_real_manifest.csv",
        "manifest": "data/datasets/manifests/sensor_real_manifest.csv",
        "task": "Sensor Motion Event Classifier"
    },
    {
        "name": "synthetic_multimodal_fusion",
        "path": "data/datasets/multimodal_synthetic_manifest.csv",
        "manifest": "data/datasets/multimodal_synthetic_manifest.csv",
        "task": "XGBoost & VQC Multimodal Classification"
    }
]

dataset_results = []
for d in datasets_inventory:
    p_info = get_file_info(d["manifest"])
    dataset_results.append({**d, "manifest_info": p_info})

# Evaluation Artifacts
eval_artifacts = [
    "data/results/canonical_held_out_evaluation_corrected.json",
    "data/results/independent_ablation_results.json",
    "data/results/phase1_duplicate_split_audit.json",
    "data/results/phase1_git_artifact_inventory.json",
    "data/results/phase1_software_verification.json",
    "data/results/canonical_held_out_evaluation.json", # Stale
    "data/results/evaluation_results.json", # Stale
    "scratch/audit_evidence/vision_full_reproduced_eval.json",
    "scratch/audit_evidence/reproduced_metrics.json",
    "scratch/audit_evidence/yolo_reproduced_metrics.json"
]

eval_results = {p: get_file_info(p) for p in eval_artifacts}

artifact_inventory = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "git_head": "e8ecd2d59a6a1876fc5f1f241495034b1a29ba38",
    "active_models": models_inventory,
    "datasets": dataset_results,
    "evaluation_artifacts": eval_results
}

os.makedirs("reports/phase2", exist_ok=True)
with open("reports/phase2/artifact_inventory.json", "w", encoding="utf-8") as f:
    json.dump(artifact_inventory, f, indent=2)

print("Saved reports/phase2/artifact_inventory.json successfully.")
