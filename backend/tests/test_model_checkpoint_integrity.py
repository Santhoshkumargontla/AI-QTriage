"""Automated tests for Phase 1 model checkpoint existence, canonical paths, and SHA-256 integrity."""
import json
import os
import hashlib
import pytest

from ml.models.canonical_paths import (
    ROOT,
    RUNTIME_MODELS,
    REGISTRY_PATH,
    MANIFEST_PATH,
    YOLO_CANONICAL,
    EFFNET_CANONICAL,
    UNET_CANONICAL,
    SENSOR_MODEL,
    XGB_CANONICAL,
    VQC_WEIGHTS,
    exists,
    resolve_existing,
    sha256_file,
    read_json,
    posix,
)


def compute_sha256(path: str) -> str:
    located = resolve_existing(path)
    assert os.path.exists(located), f"Checkpoint missing: {located}"
    h = hashlib.sha256()
    with open(located, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_all_canonical_checkpoints_exist_on_disk():
    for spec in RUNTIME_MODELS:
        name = spec["model_name"]
        path = spec["canonical_path"]
        assert exists(path), f"Canonical checkpoint for '{name}' does not exist at {path}"
        assert os.path.getsize(resolve_existing(path)) > 0, f"Checkpoint for '{name}' is empty (0 bytes)"


def test_obsolete_legacy_paths_are_not_canonical():
    # Obsolete unet_best.pt was replaced by unet_injury_best.pt
    legacy_unet = os.path.join(ROOT, "ml", "models", "vision", "unet_best.pt")
    assert not os.path.exists(legacy_unet), "Legacy unet_best.pt must not exist; canonical is unet_injury_best.pt"
    
    # Obsolete sensor_model.pt was replaced by sensor_motion_best.json
    legacy_sensor = os.path.join(ROOT, "ml", "models", "sensor_model.pt")
    assert not os.path.exists(legacy_sensor), "Legacy sensor_model.pt must not exist; canonical is sensor_motion_best.json"


def test_live_sha256_matches_metadata_and_registry():
    registry = read_json(REGISTRY_PATH)
    manifest = read_json(MANIFEST_PATH)
    manifest_by_name = {m["model_name"]: m for m in manifest.get("models", [])}

    for spec in RUNTIME_MODELS:
        name = spec["model_name"]
        canonical_path = spec["canonical_path"]
        meta_path = spec["metadata_path"]

        # 1. Live computed SHA-256
        live_sha = compute_sha256(canonical_path)

        # 2. Compare with model metadata JSON
        meta = read_json(meta_path)
        expected_meta_sha = meta.get("artifact_sha256")
        assert expected_meta_sha is not None, f"artifact_sha256 missing in metadata for {name}"
        assert live_sha == expected_meta_sha, (
            f"SHA-256 mismatch in metadata for {name}: computed {live_sha} != expected {expected_meta_sha}"
        )

        # 3. Compare with model_registry.json
        reg_entry = registry.get(name)
        assert reg_entry is not None, f"Model {name} missing from model_registry.json"
        assert reg_entry.get("artifact_sha256") == live_sha, (
            f"SHA-256 mismatch in model_registry.json for {name}"
        )

        # 4. Compare with canonical_manifest.json
        man_entry = manifest_by_name.get(name)
        assert man_entry is not None, f"Model {name} missing from canonical_manifest.json"
        assert man_entry.get("sha256") == live_sha, (
            f"SHA-256 mismatch in canonical_manifest.json for {name}"
        )


def test_phase1_artifact_inventory_integrity():
    inventory_path = os.path.join(ROOT, "data", "results", "phase1_git_artifact_inventory.json")
    assert os.path.exists(inventory_path), "Missing phase1_git_artifact_inventory.json"
    
    inventory = read_json(inventory_path)
    checkpoints = inventory.get("checkpoint_hash_verification", {})

    key_map = {
        "yolo11": YOLO_CANONICAL,
        "efficientnetv2": EFFNET_CANONICAL,
        "unet": UNET_CANONICAL,
        "sensor": SENSOR_MODEL,
        "xgboost": XGB_CANONICAL,
        "vqc": VQC_WEIGHTS,
    }

    for key, path in key_map.items():
        entry = checkpoints.get(key)
        assert entry is not None, f"Missing inventory verification entry for {key}"
        assert entry.get("match") is True, f"Inventory records mismatch for {key}"
        live_sha = compute_sha256(path)
        assert entry.get("actual_sha256") == live_sha, (
            f"Inventory actual_sha256 {entry.get('actual_sha256')} != live {live_sha} for {key}"
        )
