"""Verify-first audit: model paths, hashes, duplicates, cwd resolution.

Run from any cwd. Does not delete or move files.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.chdir(ROOT)

from ml.models.canonical_paths import (  # noqa: E402
    EFFNET_CANONICAL,
    REGISTRY_PATH,
    ROOT as CANON_ROOT,
    SENSOR_MODEL,
    SENSOR_SCALER,
    UNET_CANONICAL,
    VQC_DIR,
    XGB_CANONICAL,
    YOLO_BACKUP_PATHS,
    YOLO_CANONICAL,
    abs_path,
    resolve_existing,
    sha256_file,
)

WEIGHT_EXTS = {".pt", ".pth", ".json", ".npz", ".pkl", ".onnx", ".bin"}
SKIP_DIR_NAMES = {
    "node_modules",
    "venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".tmp_pytest",
}


def sha256_path(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk_model_files(base: str) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in WEIGHT_EXTS or name.endswith(".pre_retrain_backup") or name.endswith(".pt.pre_processed_retrain_backup"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def rel(path: str) -> str:
    try:
        return os.path.relpath(path, ROOT).replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def main() -> dict:
    models_root = os.path.join(ROOT, "ml", "models")
    files = walk_model_files(models_root)
    rows = []
    by_hash = defaultdict(list)
    for path in files:
        digest = sha256_path(path)
        size = os.path.getsize(path)
        row = {
            "rel": rel(path),
            "abs": path,
            "sha256": digest,
            "size": size,
            "in_archive": "/_archive/" in rel(path).replace("\\", "/"),
        }
        rows.append(row)
        if digest:
            by_hash[digest].append(row["rel"])

    canonical = {
        "YOLO": YOLO_CANONICAL,
        "EfficientNet": EFFNET_CANONICAL,
        "U-Net": UNET_CANONICAL,
        "XGBoost": XGB_CANONICAL,
        "VQC_dir": VQC_DIR,
        "VQC_weights": os.path.join(VQC_DIR, "vqc_weights.npz"),
        "VQC_pca": os.path.join(VQC_DIR, "pca.pkl"),
        "VQC_scaler": os.path.join(VQC_DIR, "scaler.pkl"),
        "Sensor": SENSOR_MODEL,
        "Sensor_scaler": SENSOR_SCALER,
        "Registry": REGISTRY_PATH,
        "XGB_metadata": os.path.join("ml", "models", "xgboost_metadata.json"),
        "YOLO_metadata": os.path.join("ml", "models", "vision", "yolo11_metadata.json"),
        "EffNet_metadata": os.path.join("ml", "models", "vision", "efficientnetv2_metadata.json"),
        "UNet_metadata": os.path.join("ml", "models", "vision", "unet_metadata.json"),
        "VQC_metadata": os.path.join(VQC_DIR, "vqc_metadata.json"),
        "Sensor_metadata": os.path.join("ml", "models", "sensor_metadata.json"),
        "eval_results": os.path.join("data", "results", "evaluation_results.json"),
        "eval_compare": os.path.join("data", "results", "vqc_xgb_comparison.json"),
        "eval_canonical": os.path.join("data", "results", "canonical_held_out_evaluation.json"),
    }

    canon_status = {}
    for name, path in canonical.items():
        rooted = abs_path(path)
        cwd_exists = os.path.exists(path)
        root_exists = os.path.exists(rooted)
        resolved = resolve_existing(path)
        digest = sha256_path(rooted) if root_exists else None
        canon_status[name] = {
            "declared": path.replace("\\", "/"),
            "abs": rooted,
            "cwd_exists_from_root": cwd_exists,
            "root_exists": root_exists,
            "resolve_existing": resolved.replace("\\", "/"),
            "sha256": digest,
        }

    registry = {}
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, encoding="utf-8") as handle:
            registry = json.load(handle)

    registry_vs_disk = []
    for model_name, entry in registry.items():
        art = entry.get("artifact_path") or entry.get("canonical_path")
        disk_sha = sha256_path(abs_path(art)) if art else None
        registry_vs_disk.append({
            "model_name": model_name,
            "canonical_path": entry.get("canonical_path"),
            "artifact_path": entry.get("artifact_path"),
            "registry_sha256": entry.get("artifact_sha256"),
            "disk_sha256": disk_sha,
            "sha_match": bool(entry.get("artifact_sha256") and disk_sha and entry["artifact_sha256"] == disk_sha),
            "status": entry.get("status"),
            "training_status": entry.get("training_status"),
            "classes": entry.get("classes"),
            "version": entry.get("version"),
        })

    backup_status = []
    for path in YOLO_BACKUP_PATHS:
        backup_status.append({
            "declared": path.replace("\\", "/"),
            "exists_cwd": os.path.exists(path),
            "exists_root": os.path.exists(abs_path(path)),
        })

    duplicates = {h: paths for h, paths in by_hash.items() if len(paths) > 1}

    out = {
        "ROOT": ROOT,
        "CANON_ROOT": CANON_ROOT,
        "roots_match": os.path.normpath(ROOT) == os.path.normpath(CANON_ROOT),
        "canonical": canon_status,
        "registry_vs_disk": registry_vs_disk,
        "yolo_backup_paths": backup_status,
        "file_count": len(rows),
        "files": rows,
        "duplicate_sha_groups": duplicates,
    }
    dest = os.path.join(ROOT, "scratch", "audit_model_paths_inventory.json")
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"wrote {dest}")
    print(f"ROOT={ROOT}")
    print(f"files={len(rows)} duplicate_groups={len(duplicates)}")
    for item in registry_vs_disk:
        print(f"REG {item['model_name']}: sha_match={item['sha_match']} status={item['status']}")
    return out


if __name__ == "__main__":
    main()
