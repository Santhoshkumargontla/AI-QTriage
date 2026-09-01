"""Prepare skin YOLO dataset: existing cut/bruise + Kaggle burn boxes (CC0).

Classes:
  0 cut, 1 bruise, 2 burn

Does NOT add fracture (X-ray). Does NOT invent swelling boxes.
Wound remains absent unless honest boxes exist (still 0 here).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone

import cv2
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ml.models.canonical_paths import ROOT, YOLO_RETRAIN_V2_ROOT

SEED = 42
OUT_ROOT = os.path.join("data", "datasets", "yolo_skin_kaggle_v1")
NAMES = {0: "cut", 1: "bruise", 2: "burn"}
BURN_ROOT = os.path.join("data", "raw", "kaggle", "shubhambaid_skin_burn")


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_for_hash(digest: str) -> str:
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _copy_existing_retrain_v2(stats: Counter):
    """Copy cut/bruise YOLO samples from yolo_retrain_v2 with same class ids 0/1."""
    src = os.path.join(ROOT, YOLO_RETRAIN_V2_ROOT) if not os.path.isabs(YOLO_RETRAIN_V2_ROOT) else YOLO_RETRAIN_V2_ROOT
    copied = 0
    for split in ("train", "val", "test"):
        img_dir = os.path.join(src, "images", split)
        lab_dir = os.path.join(src, "labels", split)
        if not os.path.isdir(img_dir):
            continue
        for name in os.listdir(img_dir):
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            stem = os.path.splitext(name)[0]
            img_src = os.path.join(img_dir, name)
            lab_src = os.path.join(lab_dir, stem + ".txt")
            lines = []
            if os.path.isfile(lab_src):
                for ln in open(lab_src, encoding="utf-8"):
                    ln = ln.strip()
                    if not ln:
                        continue
                    parts = ln.split()
                    cid = int(float(parts[0]))
                    if cid not in (0, 1):
                        continue  # drop unsupported wound if present
                    if len(parts) != 5:
                        continue
                    lines.append(f"{cid} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                    stats[NAMES[cid]] += 1
            dest_img = os.path.join(OUT_ROOT, "images", split, f"v2_{name}")
            dest_lab = os.path.join(OUT_ROOT, "labels", split, f"v2_{stem}.txt")
            shutil.copy2(img_src, dest_img)
            with open(dest_lab, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + ("\n" if lines else ""))
            copied += 1
    return copied


def _add_burns(stats: Counter, max_images: int = 180):
    """Subsample burns so they do not drown cut/bruise (~60 boxes each)."""
    import random

    root = os.path.join(ROOT, BURN_ROOT)
    candidates = []
    skipped = 0
    for name in os.listdir(root):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(root, name)
        lab_path = os.path.join(root, stem + ".txt")
        if not os.path.isfile(lab_path):
            skipped += 1
            continue
        lines_out = []
        for ln in open(lab_path, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 5:
                continue
            # Remap burn degree ids 0/1/2 → single class burn=2
            lines_out.append(f"2 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
        if not lines_out:
            skipped += 1
            continue
        candidates.append((img_path, lines_out))

    rng = random.Random(SEED)
    rng.shuffle(candidates)
    selected = candidates[:max_images]
    added = 0
    for img_path, lines_out in selected:
        digest = _sha_file(img_path)
        split = _split_for_hash(digest)
        ext = os.path.splitext(img_path)[1]
        dest_img = os.path.join(OUT_ROOT, "images", split, f"burn_{digest[:16]}{ext}")
        dest_lab = os.path.join(OUT_ROOT, "labels", split, f"burn_{digest[:16]}.txt")
        shutil.copy2(img_path, dest_img)
        with open(dest_lab, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines_out) + "\n")
        for _ in lines_out:
            stats["burn"] += 1
        added += 1
    return added, skipped + max(0, len(candidates) - max_images)


def prepare() -> dict:
    if os.path.isdir(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, "labels", split), exist_ok=True)

    stats = Counter()
    n_v2 = _copy_existing_retrain_v2(stats)
    n_burn, n_skip = _add_burns(stats)

    yaml_path = os.path.join(OUT_ROOT, "data.yaml")
    data = {
        "path": os.path.abspath(OUT_ROOT).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 3,
        "names": [NAMES[i] for i in range(3)],
    }
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    split_counts = {}
    for split in ("train", "val", "test"):
        imgs = len(os.listdir(os.path.join(OUT_ROOT, "images", split)))
        split_counts[split] = imgs

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "out_root": _rel(OUT_ROOT),
        "yaml": _rel(yaml_path),
        "names": data["names"],
        "copied_from_retrain_v2_images": n_v2,
        "burn_images_added": n_burn,
        "burn_skipped": n_skip,
        "box_counts": dict(stats),
        "split_image_counts": split_counts,
        "wound_boxes": 0,
        "swelling_boxes": 0,
        "notes": "Burn labels from shubhambaid/skin-burn-dataset (CC0-1.0). Degrees collapsed to class burn.",
    }
    with open(os.path.join(OUT_ROOT, "PREPARE_REPORT.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    prepare()
