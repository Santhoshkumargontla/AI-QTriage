"""Build yolo_retrain_v2 from existing honest labels + existing negatives.

Does not fabricate images or boxes. Does not remap swelling/abrasion/laceration
to wound. Does not modify yolo_processed or raw sources.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import (
    YOLO_PROCESSED_ROOT,
    YOLO_RETRAIN_V2_ROOT,
    YOLO_RETRAIN_V2_YAML,
    abs_path,
)

ACTIVE = {0: "cut", 1: "bruise", 2: "wound"}
NEGATIVES = (
    ("data/datasets/yolo_injury/blank_skin.jpg", "val", "blank_skin"),
    ("data/datasets/yolo_injury/dummy_test.jpg", "test", "dummy_test"),
)


def _rel(path: str) -> str:
    return str(path).replace("\\", "/")


def prepare():
    src_root = abs_path(YOLO_PROCESSED_ROOT)
    dest_root = abs_path(YOLO_RETRAIN_V2_ROOT)
    src_manifest = os.path.join(src_root, "manifest.csv")
    if not os.path.isfile(src_manifest):
        raise FileNotFoundError("yolo_processed/manifest.csv missing. Run prepare_yolo_processed_dataset.py first.")

    if os.path.isdir(dest_root):
        shutil.rmtree(dest_root)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(dest_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dest_root, "labels", split), exist_ok=True)

    rows = []
    with open(src_manifest, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            src_img = abs_path(row["dest_image"])
            src_lbl = abs_path(row["dest_label"])
            name = os.path.basename(src_img)
            dest_img = os.path.join(dest_root, "images", split, name)
            dest_lbl = os.path.join(dest_root, "labels", split, os.path.splitext(name)[0] + ".txt")
            shutil.copy2(src_img, dest_img)
            shutil.copy2(src_lbl, dest_lbl)
            row = dict(row)
            row["dest_image"] = _rel(os.path.relpath(dest_img, abs_path(".")))
            row["dest_label"] = _rel(os.path.relpath(dest_lbl, abs_path(".")))
            row["retrain_v2"] = "copied_from_yolo_processed"
            rows.append(row)

    negatives_added = []
    for rel, split, stem in NEGATIVES:
        src = abs_path(rel)
        if not os.path.isfile(src):
            continue
        ext = os.path.splitext(src)[1]
        dest_img = os.path.join(dest_root, "images", split, f"negative__{stem}{ext}")
        dest_lbl = os.path.join(dest_root, "labels", split, f"negative__{stem}.txt")
        shutil.copy2(src, dest_img)
        with open(dest_lbl, "w", encoding="utf-8") as handle:
            handle.write("")
        rec = {
            "sample_id": f"negative__{stem}",
            "split": split,
            "dest_image": _rel(os.path.relpath(dest_img, abs_path("."))),
            "dest_label": _rel(os.path.relpath(dest_lbl, abs_path("."))),
            "source_dataset": "yolo_injury_negative",
            "source_path": _rel(rel),
            "kept_classes": "",
            "boxes_out": "0",
            "retrain_v2": "existing_negative_empty_label",
            "provenance": "SYNTHETIC",
        }
        rows.append(rec)
        negatives_added.append(rec)

    boxes = {s: Counter() for s in ("train", "val", "test")}
    images = {s: Counter() for s in ("train", "val", "test")}
    empty = Counter()
    for split in ("train", "val", "test"):
        img_dir = os.path.join(dest_root, "images", split)
        lbl_dir = os.path.join(dest_root, "labels", split)
        for name in os.listdir(img_dir):
            stem = os.path.splitext(name)[0]
            lbl = os.path.join(lbl_dir, stem + ".txt")
            lines = [ln.strip() for ln in open(lbl, encoding="utf-8") if ln.strip()] if os.path.isfile(lbl) else []
            if not lines:
                empty[split] += 1
                continue
            present = set()
            for line in lines:
                cid = int(float(line.split()[0]))
                cname = ACTIVE.get(cid, str(cid))
                boxes[split][cname] += 1
                present.add(cname)
            for cname in present:
                images[split][cname] += 1

    hashes = defaultdict(list)
    stems = defaultdict(list)
    for split in ("train", "val", "test"):
        img_dir = os.path.join(dest_root, "images", split)
        for name in os.listdir(img_dir):
            path = os.path.join(img_dir, name)
            digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
            hashes[digest].append(f"{split}/{name}")
            stems[os.path.splitext(name)[0]].append(split)
    hash_leaks = {k: v for k, v in hashes.items() if len({p.split("/")[0] for p in v}) > 1}
    stem_leaks = {k: v for k, v in stems.items() if len(set(v)) > 1}

    yaml_path = abs_path(YOLO_RETRAIN_V2_YAML)
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(
            f"path: {dest_root}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            "  0: cut\n"
            "  1: bruise\n"
            "  2: wound\n"
            "provenance: SYNTHETIC\n"
            "notes: |\n"
            "  Honest cut/bruise from yolo_processed. wound has 0 labels.\n"
            "  swelling/abrasion/laceration were not remapped.\n"
            "  Existing blank_skin (val) and dummy_test (test) added as empty-label negatives.\n"
            "  Raw sources were not modified. No fabricated boxes.\n"
        )

    manifest_path = os.path.join(dest_root, "manifest.csv")
    fields = sorted({k for row in rows for k in row})
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "did_not_fabricate_images": True,
        "did_not_remap_dropped_classes": True,
        "silent_remap": False,
        "source": "yolo_processed + existing negative files",
        "negatives_added": negatives_added,
        "empty_label_images_by_split": dict(empty),
        "boxes_per_class_by_split": {k: dict(v) for k, v in boxes.items()},
        "images_per_class_by_split": {k: dict(v) for k, v in images.items()},
        "image_counts": {
            split: len(os.listdir(os.path.join(dest_root, "images", split)))
            for split in ("train", "val", "test")
        },
        "known_limitation": (
            "No legitimate original wound boxes exist in-repo after dropping "
            "swelling/abrasion/laceration remaps. cut/bruise remain SYNTHETIC drawings plus two public samples."
        ),
        "yaml": _rel(YOLO_RETRAIN_V2_YAML),
        "leakage": {
            "exact_file_hash_across_splits": hash_leaks,
            "stem_across_splits": stem_leaks,
            "note": "Subject IDs are NOT_AVAILABLE on synthetic filenames. Split membership copied from yolo_processed.",
        },
    }
    with open(os.path.join(dest_root, "processing_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    prepare()
