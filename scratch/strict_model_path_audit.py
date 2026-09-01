"""Verify-first: hash every ml/models weight, CWD probes, code references. No deletes."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ml.models.canonical_paths import (  # noqa: E402
    EFFNET_CANDIDATE,
    EFFNET_CANONICAL,
    EVAL_COMPARE,
    EVAL_HELD_OUT,
    EVAL_RESULTS,
    EVAL_YOLO_THRESHOLD,
    MANIFEST_PATH,
    REGISTRY_PATH,
    RUNTIME_MODELS,
    SENSOR_MODEL,
    SENSOR_SCALER,
    UNET_CANDIDATE,
    UNET_CANONICAL,
    VQC_DIR,
    VQC_WEIGHTS,
    XGB_CANONICAL,
    YOLO_BACKUP_PATHS,
    YOLO_CANONICAL,
    YOLO_PRETRAINED_INIT,
    YOLO_SYNTHETIC_BASELINE,
    abs_path,
    posix,
    read_json,
    resolve_existing,
    sha256_file,
)

WEIGHT_EXTS = {".pt", ".pth", ".json", ".npz", ".pkl", ".onnx"}
SKIP_DIR = {"node_modules", "venv", ".git", "__pycache__", ".pytest_cache"}
CODE_EXTS = {".py", ".json", ".ts", ".tsx", ".md"}


def sha256_path(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk_weights(base: str) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in WEIGHT_EXTS or ".pre_retrain" in name or name.endswith(".backup"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def rel(path: str) -> str:
    try:
        return posix(os.path.relpath(path, ROOT))
    except ValueError:
        return posix(path)


def code_mentions(basename: str) -> list[str]:
    hits = []
    for walk_root in (
        os.path.join(ROOT, "ml"),
        os.path.join(ROOT, "backend"),
        os.path.join(ROOT, "scripts"),
        os.path.join(ROOT, "frontend"),
    ):
        for dirpath, dirnames, filenames in os.walk(walk_root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR and d != "venv"]
            for name in filenames:
                if os.path.splitext(name)[1].lower() not in CODE_EXTS:
                    continue
                path = os.path.join(dirpath, name)
                try:
                    text = open(path, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                if basename in text:
                    hits.append(rel(path))
                    if len(hits) >= 12:
                        return hits
    return hits


def probe_loads(cwd: str) -> dict:
    previous = os.getcwd()
    os.chdir(cwd)
    out = {"cwd": os.getcwd(), "loads": {}}
    try:
        from ml.vision.yolo_wrapper import YOLO11Detector
        from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
        from ml.vision.unet_wrapper import UNetSegmenter
        from ml.classifiers.xgboost_classifier import XGBoostClassifier
        from ml.classifiers.vqc_classifier import VQCClassifier
        from ml.classifiers.sensor_classifier import SensorClassifier

        yolo = YOLO11Detector()
        out["loads"]["yolo"] = {
            "loaded": yolo.model is not None,
            "sha": yolo.artifact_sha256,
            "path": posix(str(yolo.model_path)),
            "status": yolo.status,
        }
        xgb = XGBoostClassifier(XGB_CANONICAL)
        out["loads"]["xgb"] = {
            "loaded": xgb.is_trained,
            "sha": xgb.artifact_sha256,
            "path": posix(str(xgb.model_path)),
        }
        eff = EfficientNetV2Classifier()
        out["loads"]["effnet"] = {"loaded": eff.is_loaded, "path": posix(str(eff.model_path or ""))}
        unet = UNetSegmenter()
        out["loads"]["unet"] = {"loaded": unet.is_loaded, "path": posix(str(unet.model_path or ""))}
        vqc = VQCClassifier(VQC_DIR)
        out["loads"]["vqc"] = {"loaded": vqc.is_trained, "status": vqc.status, "dir": posix(str(vqc.model_dir or ""))}
        sen = SensorClassifier()
        out["loads"]["sensor"] = {"loaded": sen.is_trained, "path": posix(str(sen.model_path))}
        out["exists_rel_yolo"] = os.path.exists(YOLO_CANONICAL)
        out["resolve_yolo"] = posix(resolve_existing(YOLO_CANONICAL))
    finally:
        os.chdir(previous)
    return out


def main() -> dict:
    models_root = os.path.join(ROOT, "ml", "models")
    files = walk_weights(models_root)
    rows = []
    by_hash = defaultdict(list)
    for path in files:
        digest = sha256_path(path)
        r = rel(path)
        row = {
            "rel": r,
            "sha256": digest,
            "size": os.path.getsize(path),
            "in_archive": "/_archive/" in r,
            "basename": os.path.basename(path),
            "code_mentions": code_mentions(os.path.basename(path)),
        }
        rows.append(row)
        if digest:
            by_hash[digest].append(r)

    canonical_rels = {posix(spec["canonical_path"]) for spec in RUNTIME_MODELS}
    canonical_rels.add(posix(SENSOR_SCALER))
    canonical_rels.add(posix(os.path.join(VQC_DIR, "scaler.pkl")))
    canonical_rels.add(posix(os.path.join(VQC_DIR, "pca.pkl")))
    backup_rels = {posix(p) for p in YOLO_BACKUP_PATHS}
    backup_rels.add(posix(YOLO_SYNTHETIC_BASELINE))
    backup_rels.add(posix(YOLO_PRETRAINED_INIT))
    candidate_rels = {posix(EFFNET_CANDIDATE), posix(UNET_CANDIDATE)}

    canon_sha = {posix(spec["canonical_path"]): sha256_file(spec["canonical_path"]) for spec in RUNTIME_MODELS}

    classified = []
    for row in rows:
        r = row["rel"]
        digest = row["sha256"]
        role = "other"
        if row["in_archive"]:
            role = "already_archived"
        elif r in canonical_rels:
            role = "runtime_canonical"
        elif r in candidate_rels:
            role = "unpromoted_candidate"
        elif r in backup_rels or r.endswith(".pre_retrain_backup") or r.endswith(".pre_retrain_v2_backup") or "pre_processed_retrain_backup" in r:
            role = "valid_backup"
        elif digest in canon_sha.values() and r not in canonical_rels:
            role = "byte_duplicate_of_canonical"
        elif r.endswith("_metadata.json") or r.endswith("model_registry.json") or r.endswith("canonical_manifest.json"):
            role = "metadata_or_registry"
        elif "reliability" in r or "threshold_eval" in r or "TRAINING_EVAL" in r or "PROMOTION" in r or "REPORT" in r:
            role = "evaluation_or_training_report"
        classified.append({**row, "role": role, "duplicate_of": [k for k, v in canon_sha.items() if v == digest]})

    registry = read_json(REGISTRY_PATH)
    manifest = read_json(MANIFEST_PATH)
    alignment = []
    for spec in RUNTIME_MODELS:
        name = spec["model_name"]
        path = spec["canonical_path"]
        disk = sha256_file(path) if os.path.isfile(abs_path(path)) else None
        reg = registry.get(name) or {}
        man = next((m for m in manifest.get("models") or [] if m["model_name"] == name), {})
        alignment.append({
            "model_name": name,
            "canonical_path": posix(path),
            "disk_sha": disk,
            "registry_sha": reg.get("artifact_sha256"),
            "manifest_sha": man.get("sha256"),
            "wrapper": spec["wrapper"],
            "evaluation_artifact": posix(spec["evaluation_artifact"]),
            "eval_exists": os.path.isfile(abs_path(spec["evaluation_artifact"])),
            "registry_path_match": posix(reg.get("canonical_path") or "") == posix(path),
            "sha_aligned": disk and disk == reg.get("artifact_sha256") == man.get("sha256"),
        })

    foreign = tempfile.mkdtemp(prefix="aiq_cwd_")
    cwd_probes = {
        "project_root": probe_loads(ROOT),
        "backend_dir": probe_loads(os.path.join(ROOT, "backend")),
        "foreign_cwd": probe_loads(foreign),
    }

    unused_duplicate_candidates = [
        c for c in classified
        if c["role"] == "byte_duplicate_of_canonical"
        and not c["in_archive"]
        and not c["code_mentions"]
    ]

    report = {
        "started": datetime.now(timezone.utc).isoformat(),
        "root": ROOT,
        "file_count": len(rows),
        "alignment": alignment,
        "cwd_probes": cwd_probes,
        "classified": classified,
        "hash_groups": {k: v for k, v in by_hash.items() if len(v) > 1},
        "unused_duplicate_candidates": unused_duplicate_candidates,
        "eval_artifacts": {
            "held_out": posix(EVAL_HELD_OUT),
            "results": posix(EVAL_RESULTS),
            "compare": posix(EVAL_COMPARE),
            "yolo_threshold": posix(EVAL_YOLO_THRESHOLD),
        },
    }
    dest = os.path.join(ROOT, "scratch", "strict_model_path_audit.json")
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("wrote", dest)
    print("alignment")
    for row in alignment:
        print(" ", row["model_name"], "sha_aligned", row["sha_aligned"], row["disk_sha"][:12] if row["disk_sha"] else None)
    print("unused byte-duplicates with no code mentions:", len(unused_duplicate_candidates))
    for row in unused_duplicate_candidates:
        print(" ", row["rel"], row["sha256"][:12])
    print("cwd loads")
    for label, block in cwd_probes.items():
        loads = {k: v.get("loaded") for k, v in block["loads"].items()}
        print(" ", label, "rel_yolo_exists", block.get("exists_rel_yolo"), loads)
    return report


if __name__ == "__main__":
    main()
