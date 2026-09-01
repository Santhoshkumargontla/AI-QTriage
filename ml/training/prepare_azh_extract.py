"""Extract AZH wound patches zip to data/datasets/external/azh_patches.

Source: uwm-bigdata/wound-segmentation data/wound_dataset/azh_wound_care_center_dataset_patches.zip
Labels are PNG masks (0/255), not YOLO txt. Subject id = hash stem before _patchIndex.
"""
from __future__ import annotations

import json
import os
import zipfile
from collections import Counter
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ZIP_PATH = os.path.join(
    ROOT,
    "data",
    "datasets",
    "external",
    "wound-segmentation",
    "data",
    "wound_dataset",
    "azh_wound_care_center_dataset_patches.zip",
)
OUT = os.path.join(ROOT, "data", "datasets", "external", "azh_patches")


def extract() -> dict:
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(ZIP_PATH)
    os.makedirs(OUT, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zh:
        zh.extractall(OUT)
    counts = Counter()
    subjects = set()
    for split in ("train", "test"):
        img_dir = os.path.join(OUT, split, "images")
        lab_dir = os.path.join(OUT, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for name in os.listdir(img_dir):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = os.path.splitext(name)[0]
            subject = stem.rsplit("_", 1)[0]
            subjects.add(subject)
            mask = os.path.join(lab_dir, name)
            counts[f"{split}_images"] += 1
            if os.path.exists(mask):
                counts[f"{split}_masks"] += 1
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_zip": os.path.relpath(ZIP_PATH, ROOT).replace("\\", "/"),
        "out": os.path.relpath(OUT, ROOT).replace("\\", "/"),
        "counts": dict(counts),
        "unique_subjects": len(subjects),
        "license_note": "AZH Wound and Vascular Center patches redistributed via UWM wound-segmentation repo. Research use; not cut/bruise/swelling taxonomy.",
        "subject_rule": "subject_id = filename hash before trailing _N patch index",
    }
    with open(os.path.join(OUT, "EXTRACT_SUMMARY.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    extract()
