"""
AI-QTriage Model Registry Manager
Tracks trained model artifacts, random seeds, dataset hashes, framework versions, and metrics.
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

REGISTRY_FILE = os.path.join(os.path.dirname(__file__), "model_registry.json")

def get_file_hash(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "N/A"
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def register_model_artifact(
    model_name: str,
    version: str,
    artifact_path: str,
    training_dataset: str,
    sample_count: int,
    classes: list,
    metrics: Dict[str, Any],
    training_command: str,
    random_seed: int = 42,
    manifest_hash: str = "N/A",
    git_commit: str = "main-v1.1.0",
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Registers or updates a trained model artifact entry in model_registry.json.
    """
    registry = {}
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except (OSError, json.JSONDecodeError):
            registry = {}

    import sklearn
    import torch

    entry = {
        "model_name": model_name,
        "version": version,
        "artifact_path": artifact_path,
        "artifact_sha256": get_file_hash(artifact_path),
        "artifact_size_mb": round(os.path.getsize(artifact_path) / (1024 * 1024), 2) if os.path.exists(artifact_path) else 0.0,
        "training_dataset": training_dataset,
        "dataset_manifest_hash": manifest_hash,
        "sample_count": sample_count,
        "classes": classes,
        "training_random_seed": random_seed,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_command": training_command,
        "git_commit": git_commit,
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "pytorch_version": torch.__version__,
        "metrics": metrics,
        "notes": notes or ""
    }

    registry[model_name] = entry

    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"[OK] Registered model '{model_name}' (version {version}) in {REGISTRY_FILE}")
    return entry

if __name__ == "__main__":
    print("Model Registry Manager Ready.")
