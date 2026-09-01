"""Phase 1 pre-training forensic audit — read-only inventory of models, code paths, and risks."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.models.canonical_paths import (  # noqa: E402
    RUNTIME_MODELS,
    ROOT as CANON_ROOT,
    sha256_file,
    read_json,
    abs_path,
)

SCRATCH = ROOT / "scratch"
ARTIFACT_EXTS = {".pt", ".json", ".npz", ".pkl", ".onnx"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    try:
        return sha256_file(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
    except OSError:
        return None


def _scan_artifacts() -> list[dict]:
    models_dir = ROOT / "ml" / "models"
    rows: list[dict] = []
    canonical_paths = {abs_path(m["canonical_path"]) for m in RUNTIME_MODELS}
    # Hash canonical + candidate weights only (full tree listing without hashing every sidecar).
    priority_globs = [
        "vision/*.pt",
        "yolo_*/*/weights/*.pt",
        "efficientnet_*/*.pt",
        "unet_*/*.pt",
        "_archive/*.pt",
    ]
    seen: set[str] = set()
    candidates: list[Path] = []
    for pattern in priority_globs:
        candidates.extend(models_dir.glob(pattern))
    for spec in RUNTIME_MODELS:
        candidates.append(Path(abs_path(spec["canonical_path"])))
    for path in sorted(set(candidates)):
        if not path.is_file() or path.suffix.lower() not in ARTIFACT_EXTS:
            continue
        rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        ap = str(path.resolve())
        rows.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "is_runtime_canonical": ap in canonical_paths,
                "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return rows


def _scan_hardcoded_classes() -> list[dict]:
    """Targeted scan of runtime-critical files only."""
    targets = [
        ROOT / "backend" / "main.py",
        ROOT / "ml" / "vision" / "yolo_wrapper.py",
        ROOT / "ml" / "vision" / "efficientnet_wrapper.py",
        ROOT / "ml" / "fusion" / "feature_fusion.py",
        ROOT / "frontend" / "app" / "cases" / "[id]" / "page.tsx",
    ]
    patterns = [
        (r"swelling", "swelling_reference"),
        (r"UNTRAINED_CLASS", "untrained_marker"),
        (r"auto.?train", "auto_train_reference"),
        (r"fallback", "fallback_reference"),
    ]
    hits: list[dict] = []
    for path in targets:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat, kind in patterns:
            if re.search(pat, text, re.I):
                hits.append({"file": path.relative_to(ROOT).as_posix(), "pattern": kind})
    return hits


def _runtime_inventory() -> list[dict]:
    rows = []
    for spec in RUNTIME_MODELS:
        cp = abs_path(spec["canonical_path"])
        meta = read_json(spec.get("metadata_path", ""))
        rows.append(
            {
                "model_name": spec["model_name"],
                "canonical_path": spec["canonical_path"],
                "exists": os.path.exists(cp),
                "sha256": _sha256(Path(cp)) if os.path.exists(cp) else None,
                "metadata_version": meta.get("version"),
                "metadata_status": meta.get("status") or meta.get("training_status"),
                "classes": meta.get("classes"),
                "dataset_provenance": meta.get("dataset_provenance") or meta.get("data_provenance_class"),
                "wrapper": spec["wrapper"],
            }
        )
    return rows


def _duplicate_sha_groups(artifacts: list[dict]) -> dict:
    by_sha: dict[str, list[str]] = {}
    for a in artifacts:
        sha = a.get("sha256")
        if not sha:
            continue
        by_sha.setdefault(sha, []).append(a["path"])
    return {sha: paths for sha, paths in by_sha.items() if len(paths) > 1}


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    artifacts = _scan_artifacts()
    runtime = _runtime_inventory()
    dupes = _duplicate_sha_groups(artifacts)

    audit = {
        "created_utc": _utc(),
        "phase": "pre_training_forensic_audit",
        "project_root": str(ROOT),
        "runtime_models": runtime,
        "findings": {
            "yolo_production_wound_zero_boxes": True,
            "yolo_data_provenance_synthetic": True,
            "effnet_kaggle_v1_active": any(
                r["model_name"] == "EfficientNetV2 Classification" and r.get("metadata_version") == "kaggle-v1"
                for r in runtime
            ),
            "unet_deduped_subject_v1_active": True,
            "xgboost_synthetic_multimodal": True,
            "vqc_experimental_only": True,
            "sensor_not_in_analyze_pipeline": True,
            "roboflow_api_key_required_for_real_yolo_boxes": not bool(os.environ.get("ROBOFLOW_API_KEY")),
            "kaggle_credentials_present": os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")),
        },
        "duplicate_sha256_groups": dupes,
        "hardcoded_class_scan_sample": _scan_hardcoded_classes(),
        "risk_flags": [
            "YOLO canonical trained mostly on synthetic cut/bruise drawings; wound class has 0 training boxes in metadata.",
            "Real cut localization on phone photos failed forensic hand-case gate (gate4_hand_localization).",
            "Roboflow injury-detection datasets blocked without ROBOFLOW_API_KEY.",
            "Kaggle wound-segmentation-2760 blocked without kaggle.json credentials on this host.",
            "Fusion feature schema still uses legacy Cut/Bruise/Swelling keys while EffNet is 8-class kaggle-v1.",
            "API correctly refuses auto-training; vision failures return null bbox/confidence.",
        ],
    }

    inventory = {
        "created_utc": _utc(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "runtime_canonical": [a for a in artifacts if a["is_runtime_canonical"]],
    }

    arch_lines = [
        "# AI-QTriage Pre-Training Architecture Report",
        "",
        f"Generated: {_utc()}",
        "",
        "## Runtime stack",
        "",
        "1. **Frontend** (Next.js): create-case wizard → `/api/cases/{id}/analyze` → case detail tabs.",
        "2. **Backend** (FastAPI `backend/main.py`): lazy singleton YOLO → U-Net ROI/full → EfficientNet → 23-d fusion → XGBoost; VQC experimental.",
        "3. **MongoDB** collection `cases`: stores `visible_injury`, fusion outputs, sensor summary.",
        "",
        "## Canonical model paths",
        "",
    ]
    for r in runtime:
        arch_lines.append(
            f"- **{r['model_name']}**: `{r['canonical_path']}` — exists={r['exists']}, sha={ (r['sha256'] or '')[:16]}…"
        )
    arch_lines.extend(
        [
            "",
            "## Data layout",
            "",
            "- Raw Kaggle: `data/raw/kaggle/` (yasinpratomo, ibrahimfateen, shubhambaid burn YOLO, fracture X-ray — not skin canonical).",
            "- External segmentation: `data/datasets/external/` (wseg, Medetec, AZH).",
            "- Processed YOLO: `data/datasets/yolo_retrain_v2` (synthetic), `yolo_real_skin_v2` (mask-derived wound + negatives).",
            "- Processed EffNet: `data/datasets/efficientnet_kaggle_v1`.",
            "- Processed U-Net: `data/datasets/unet_deduped_subject`.",
            "",
            "## Promotion policy",
            "",
            "Candidates under `ml/models/*_candidate*` or training run folders are **not** loaded until gates pass.",
            "YOLO real_skin_v2 candidate was **KEEP_BASELINE** (failed gate1 + gate4).",
            "",
            "## Blockers for real-data YOLO retrain",
            "",
            "- Requires Roboflow Universe injury-detection exports (CC BY 4.0) with honest bounding boxes.",
            "- `ROBOFLOW_API_KEY` not set in this environment.",
            "- Classification-only Kaggle folders must **not** be converted to YOLO boxes.",
        ]
    )

    (SCRATCH / "pre_training_forensic_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (SCRATCH / "pre_training_model_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    (SCRATCH / "pre_training_architecture_report.md").write_text("\n".join(arch_lines), encoding="utf-8")
    print("Wrote scratch/pre_training_forensic_audit.json")
    print("Wrote scratch/pre_training_model_inventory.json")
    print("Wrote scratch/pre_training_architecture_report.md")


if __name__ == "__main__":
    main()
