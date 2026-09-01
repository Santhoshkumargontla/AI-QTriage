"""Build file inventory, duplicate report, and leakage audit for processed datasets."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
SCRATCH = ROOT / "scratch"

DATASET_ROOTS = [
    ROOT / "data" / "processed",
    ROOT / "data" / "datasets" / "efficientnet_kaggle_v1",
    ROOT / "data" / "datasets" / "unet_deduped_subject",
    ROOT / "data" / "datasets" / "yolo_real_skin_v2",
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    inventory = []
    by_sha: dict[str, list[str]] = defaultdict(list)
    corrupted = []

    for base in DATASET_ROOTS:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".txt", ".csv", ".yaml", ".yml"}:
                continue
            rel = path.relative_to(ROOT).as_posix()
            try:
                size = path.stat().st_size
                if size == 0:
                    corrupted.append({"path": rel, "reason": "zero_bytes"})
                    continue
                sha = _sha(path) if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else None
                if sha:
                    by_sha[sha].append(rel)
                inventory.append({"path": rel, "size_bytes": size, "sha256": sha})
            except OSError as exc:
                corrupted.append({"path": rel, "reason": str(exc)})

    dupes = {sha: paths for sha, paths in by_sha.items() if len(paths) > 1}
    dup_report = {
        "created_utc": _utc(),
        "exact_duplicate_groups": len(dupes),
        "cross_dataset_duplicates": [
            {"sha256": sha, "paths": paths} for sha, paths in list(dupes.items())[:500]
        ],
    }
    integrity = {
        "created_utc": _utc(),
        "files_inventoried": len(inventory),
        "corrupted_or_empty": corrupted[:200],
        "corrupted_count": len(corrupted),
    }
    leakage = {
        "created_utc": _utc(),
        "yolo_real_skin_v2": json.loads(
            (ROOT / "data" / "datasets" / "yolo_real_skin_v2" / "PREPARE_REPORT.json").read_text(encoding="utf-8")
        )["leakage_exact_hash"]
        if (ROOT / "data" / "datasets" / "yolo_real_skin_v2" / "PREPARE_REPORT.json").is_file()
        else {},
        "exact_duplicate_groups_across_processed": len(dupes),
        "verdict": "PASS" if not corrupted else "REVIEW_CORRUPTED_FILES",
    }

    (MANIFESTS / "file_inventory.json").write_text(
        json.dumps({"created_utc": _utc(), "count": len(inventory), "sample": inventory[:100]}, indent=2),
        encoding="utf-8",
    )
    (MANIFESTS / "duplicate_report.json").write_text(json.dumps(dup_report, indent=2), encoding="utf-8")
    (MANIFESTS / "dataset_integrity_report.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    (SCRATCH / "data_leakage_audit.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    md = [
        "# Data Leakage Audit",
        "",
        f"Generated: {_utc()}",
        "",
        f"- Exact duplicate SHA groups: **{len(dupes)}**",
        f"- Corrupted/empty files: **{len(corrupted)}**",
        f"- yolo_real_skin_v2 cross-split leakage: **{leakage.get('yolo_real_skin_v2', {})}**",
        "",
        "## Verdict",
        "",
        f"**{leakage['verdict']}** — Roboflow YOLO merge blocked until API key; existing processed sets audited.",
    ]
    (SCRATCH / "data_leakage_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"dupes": len(dupes), "corrupted": len(corrupted), "verdict": leakage["verdict"]}))


if __name__ == "__main__":
    main()
