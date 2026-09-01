"""Prepare EfficientNet multi-class dataset from Kaggle wound photos + OOD reject.

Classes (honest taxonomy):
  abrasion, bruise, burn, cut, laceration, wound, normal, ood_reject

Sources:
  - yasinpratomo/wound-dataset (Kaggle, license unknown on card — document)
  - ibrahimfateen/wound-classification (Kaggle, license unknown — document)
  - existing reject-v2 ood_reject + normal patches when present
  - synthetic ood_reject canvases if needed

NOT included (no honest labels found):
  - swelling / edema skin photos
  - fracture (X-ray modality — separate model)

Leakage controls: exact pixel-hash dedupe across sources; stratified hash split.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ml.models.canonical_paths import ROOT

SEED = 42
OUT_ROOT = os.path.join("data", "datasets", "efficientnet_kaggle_v1")
CLASSES = [
    "abrasion",
    "bruise",
    "burn",
    "cut",
    "laceration",
    "wound",
    "normal",
    "ood_reject",
]

FOLDER_MAP = {
    "abrasions": "abrasion",
    "bruises": "bruise",
    "burns": "burn",
    "cut": "cut",
    "cuts": "cut",
    "laceration": "laceration",
    "laseration": "laceration",  # typo in ibrahim dataset
    "lacerations": "laceration",
    "stab_wound": "wound",
    "stab wounds": "wound",
    "pressure wounds": "wound",
    "venous wounds": "wound",
    "diabetic wounds": "wound",
    "surgical wounds": "wound",
    "normal": "normal",
    # skipped: ingrown_nails (not trauma triage target)
}

SOURCES = [
    os.path.join("data", "raw", "kaggle", "yasinpratomo_wound_dataset"),
    os.path.join("data", "raw", "kaggle", "ibrahimfateen_wound_classification"),
]


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def _pixel_sha(bgr: np.ndarray) -> str:
    h, w = bgr.shape[:2]
    return hashlib.sha256(bgr.tobytes() + f"|{w}x{h}".encode()).hexdigest()


def _split_for_hash(digest: str) -> str:
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _synth_ood(n: int = 60) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    out = []
    for i in range(n):
        if i % 5 == 0:
            img = np.zeros((224, 224, 3), np.uint8)
        elif i % 5 == 1:
            img = np.full((224, 224, 3), 255, np.uint8)
        elif i % 5 == 2:
            img = np.full((224, 224, 3), 128, np.uint8)
        elif i % 5 == 3:
            img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        else:
            img = np.full((224, 224, 3), (int(rng.integers(0, 255)), int(rng.integers(0, 255)), int(rng.integers(0, 255))), np.uint8)
        out.append(img)
    return out


def _iter_class_images(root: str):
    if not os.path.isdir(root):
        return
    for dirpath, _, filenames in os.walk(root):
        folder = os.path.basename(dirpath).strip().lower()
        mapped = FOLDER_MAP.get(folder)
        if not mapped:
            continue
        for name in filenames:
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                yield mapped, os.path.join(dirpath, name)


def _reuse_reject_v2_normals_ood(pool: dict, rows: list, exclusions: list):
    """Pull previously curated normal/ood from reject_v2 if present."""
    manifest = os.path.join(ROOT, "data", "datasets", "efficientnet_reject_v2", "manifest.csv")
    if not os.path.isfile(manifest):
        return
    with open(manifest, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cls = row.get("class")
            if cls not in {"normal", "ood_reject"}:
                continue
            path = os.path.join(ROOT, row["image_path"].replace("/", os.sep))
            if not os.path.isfile(path):
                continue
            bgr = cv2.imread(path)
            if bgr is None:
                continue
            digest = _pixel_sha(bgr)
            if digest in pool:
                exclusions.append({"reason": "exact_dup", "source": path, "class": cls})
                continue
            pool[digest] = True
            rows.append(
                {
                    "image_path": row["image_path"].replace("\\", "/"),
                    "class": cls,
                    "split": row.get("split") or _split_for_hash(digest),
                    "source": "efficientnet_reject_v2_reuse",
                    "pixel_sha256": digest,
                    "license_note": row.get("license_note") or "reused_from_reject_v2",
                }
            )


def build() -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_ROOT, exist_ok=True)
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            os.makedirs(os.path.join(OUT_ROOT, split, cls), exist_ok=True)

    pool: dict[str, bool] = {}
    rows: list[dict] = []
    exclusions: list[dict] = []
    source_counts = Counter()

    for src_root in SOURCES:
        abs_src = os.path.join(ROOT, src_root) if not os.path.isabs(src_root) else src_root
        for cls, path in _iter_class_images(abs_src):
            bgr = cv2.imread(path)
            if bgr is None:
                exclusions.append({"reason": "unreadable", "source": path})
                continue
            digest = _pixel_sha(bgr)
            if digest in pool:
                exclusions.append({"reason": "exact_dup", "source": path, "class": cls})
                continue
            pool[digest] = True
            split = _split_for_hash(digest)
            dest_name = f"{digest[:16]}.jpg"
            dest = os.path.join(OUT_ROOT, split, cls, dest_name)
            cv2.imwrite(dest, bgr)
            rows.append(
                {
                    "image_path": _rel(dest),
                    "class": cls,
                    "split": split,
                    "source": _rel(path),
                    "pixel_sha256": digest,
                    "license_note": "kaggle_card_license_unknown_review_before_redistribution",
                }
            )
            source_counts[f"{os.path.basename(abs_src)}:{cls}"] += 1

    _reuse_reject_v2_normals_ood(pool, rows, exclusions)

    # Ensure minimum ood_reject coverage
    ood_n = sum(1 for r in rows if r["class"] == "ood_reject")
    if ood_n < 40:
        for i, bgr in enumerate(_synth_ood(50)):
            digest = _pixel_sha(bgr)
            if digest in pool:
                continue
            pool[digest] = True
            split = _split_for_hash(digest)
            dest = os.path.join(OUT_ROOT, split, "ood_reject", f"synth_{digest[:12]}.png")
            cv2.imwrite(dest, bgr)
            rows.append(
                {
                    "image_path": _rel(dest),
                    "class": "ood_reject",
                    "split": split,
                    "source": "synthetic_ood_canvas",
                    "pixel_sha256": digest,
                    "license_note": "synthetic",
                }
            )

    # Cross-split leakage check
    by_split = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r["pixel_sha256"])
    leakage = {
        "train_val": len(by_split["train"] & by_split["val"]),
        "train_test": len(by_split["train"] & by_split["test"]),
        "val_test": len(by_split["val"] & by_split["test"]),
    }

    manifest_path = os.path.join(OUT_ROOT, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "class", "split", "source", "pixel_sha256", "license_note"],
        )
        writer.writeheader()
        writer.writerows(rows)

    dist = {
        split: dict(Counter(r["class"] for r in rows if r["split"] == split))
        for split in ("train", "val", "test")
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "out_root": _rel(OUT_ROOT),
        "classes": CLASSES,
        "n_rows": len(rows),
        "distribution": dist,
        "source_counts": dict(source_counts),
        "exclusions": len(exclusions),
        "leakage_exact_hash": leakage,
        "leakage_free": all(v == 0 for v in leakage.values()),
        "swelling": "NOT_INCLUDED_NO_HONEST_LABELS",
        "fracture": "NOT_INCLUDED_DIFFERENT_MODALITY_XRAY",
        "notes": (
            "Kaggle wound folders mapped to taxonomy. Chronic wound types merged into 'wound'. "
            "Ingrown nails excluded. Licenses often 'unknown' on Kaggle cards — research use only."
        ),
    }
    with open(os.path.join(OUT_ROOT, "PREPARE_REPORT.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(os.path.join(OUT_ROOT, "exclusions.json"), "w", encoding="utf-8") as handle:
        json.dump(exclusions[:500], handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    build()
