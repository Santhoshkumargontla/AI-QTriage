"""Merge Roboflow injury YOLO exports into data/processed/yolo with label normalization.

Requires ROBOFLOW_API_KEY and downloaded trees under data/raw/roboflow/.
Does NOT invent boxes from classification folders.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "processed" / "yolo"
RAW_ROBO = ROOT / "data" / "raw" / "roboflow"
MANIFESTS = ROOT / "data" / "manifests"

# Target taxonomy — only classes with mapped boxes are kept.
CLASS_MAP = {
    "cut": 0,
    "cuts": 0,
    "abrasion": 1,
    "abrasions": 1,
    "abration": 1,
    "abrations": 1,
    "bruise": 2,
    "bruises": 2,
    "bruising": 2,
    "laceration": 3,
    "lacerations": 3,
    "laseration": 3,
    "wound": 4,
    "stab": 4,
    "stab_wound": 4,
}
NAMES = ["cut", "abrasion", "bruise", "laceration", "wound"]
SKIP_CLASSES = {"burn", "burns", "scarring", "injury-detection", "no_abnormality", "blister"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_yaml_names(yaml_path: Path) -> dict[int, str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def _collect_sources() -> list[Path]:
    if not RAW_ROBO.is_dir():
        return []
    roots = []
    for child in RAW_ROBO.iterdir():
        if not child.is_dir():
            continue
        yaml_candidates = list(child.rglob("data.yaml"))
        if yaml_candidates:
            roots.append(yaml_candidates[0].parent)
    return roots


def _parse_label_line(line: str, src_names: dict[int, str]) -> tuple[int, str, str] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cid = int(float(parts[0]))
    raw = src_names.get(cid, str(cid)).strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")
    if any(skip in key for skip in SKIP_CLASSES):
        return None
    if key not in CLASS_MAP:
        return None
    tid = CLASS_MAP[key]
    coords = " ".join(parts[1:])
    return tid, NAMES[tid], f"{tid} {coords}"


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    sources = _collect_sources()
    if not sources:
        report = {
            "created_utc": _utc(),
            "status": "BLOCKED_NO_ROBOFLOW_DATA",
            "reason": "No data/raw/roboflow/* exports found. Set ROBOFLOW_API_KEY and run acquire_real_public_datasets.py first.",
        }
        (MANIFESTS / "yolo_roboflow_prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    if OUT.exists():
        # preserve prior tree — write to sibling v2
        out = ROOT / "data" / "processed" / "yolo_roboflow_v1"
    else:
        out = OUT

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    seen_sha: set[str] = set()
    box_counts: Counter = Counter()
    img_counts: Counter = Counter()
    provenance: list[dict] = []
    rng = random.Random(42)

    for src_root in sources:
        yaml_path = src_root / "data.yaml"
        if not yaml_path.is_file():
            yaml_path = next(src_root.rglob("data.yaml"), None)
        if not yaml_path:
            continue
        src_names = _read_yaml_names(yaml_path)
        src_id = src_root.name
        for split in ("train", "valid", "val", "test"):
            norm_split = "val" if split == "valid" else split
            img_dir = src_root / split / "images"
            if not img_dir.is_dir():
                img_dir = src_root / "images" / split
            lab_dir = src_root / split / "labels"
            if not lab_dir.is_dir():
                lab_dir = src_root / "labels" / split
            if not img_dir.is_dir():
                continue
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    continue
                sha = _file_sha(img_path)
                if sha in seen_sha:
                    continue
                seen_sha.add(sha)
                stem = f"{src_id}_{img_path.stem}"
                lab_path = lab_dir / (img_path.stem + ".txt")
                lines_out = []
                if lab_path.is_file():
                    for ln in lab_path.read_text(encoding="utf-8").splitlines():
                        parsed = _parse_label_line(ln, src_names)
                        if parsed:
                            lines_out.append(parsed[2])
                            box_counts[parsed[1]] += 1
                # hash-based split reassignment to avoid source split leakage
                bucket = int(sha[:8], 16) % 100
                dest_split = "train" if bucket < 70 else ("val" if bucket < 85 else "test")
                dest_img = out / "images" / dest_split / (stem + img_path.suffix.lower())
                dest_lab = out / "labels" / dest_split / (stem + ".txt")
                shutil.copy2(img_path, dest_img)
                dest_lab.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
                img_counts[dest_split] += 1
                provenance.append({"sha256": sha, "source": src_id, "split": dest_split, "boxes": len(lines_out)})

    # drop unsupported classes from yaml
    supported = [c for c in NAMES if box_counts[c] >= 1]
    name_to_id = {c: i for i, c in enumerate(supported)}
    # remap labels if any class dropped
    if supported != NAMES:
        for lab_path in out.rglob("labels/*/*.txt"):
            new_lines = []
            for ln in lab_path.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) != 5:
                    continue
                old_id = int(parts[0])
                if old_id >= len(NAMES):
                    continue
                cls = NAMES[old_id]
                if cls not in name_to_id:
                    continue
                new_lines.append(f"{name_to_id[cls]} {' '.join(parts[1:])}")
            lab_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

    yaml_out = {
        "path": str(out.relative_to(ROOT)).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(supported)},
        "nc": len(supported),
    }
    (out / "dataset.yaml").write_text(yaml.dump(yaml_out, sort_keys=False), encoding="utf-8")

    report = {
        "created_utc": _utc(),
        "status": "OK",
        "output": str(out.relative_to(ROOT)).replace("\\", "/"),
        "sources": [str(p.relative_to(ROOT)) for p in sources],
        "image_counts": dict(img_counts),
        "box_counts": dict(box_counts),
        "supported_classes": supported,
        "unique_images": len(seen_sha),
        "leakage_exact_sha_cross_split": 0,
    }
    (MANIFESTS / "yolo_split_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (MANIFESTS / "yolo_class_distribution.json").write_text(
        json.dumps({"box_counts": dict(box_counts), "supported": supported}, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
