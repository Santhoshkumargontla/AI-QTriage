"""Build a leakage-free EfficientNet set. Does not fabricate negative images.

Does not modify raw sources. Does not remap abrasion/laceration/burn to swelling.
Exact pixel duplicates are kept once. Splits are hash-disjoint.

Existing blank_skin.jpg and dummy_test.jpg are recorded as OOD evaluation
files only. Two images are not enough to train a normal/reject class.
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

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.canonical_paths import EFFNET_PROCESSED_MANIFEST, EFFNET_PROCESSED_ROOT

SEED = 42
MIN_UNIQUE_FOR_SPLIT = 3
PUBLIC_VISUAL = ["cut", "bruise", "swelling", "abrasion", "laceration", "burn"]
EXISTING_NEGATIVES = (
    ("data/datasets/yolo_injury/blank_skin.jpg", "blank_skin", "yolo_injury"),
    ("data/datasets/yolo_injury/dummy_test.jpg", "dummy_test", "yolo_injury"),
)

# Set after build() from classes that have enough unique images.
CLASSES: list[str] = ["cut", "bruise", "swelling"]


def _rel(path: str) -> str:
    return path.replace("\\", "/")


def _pixel_sha256(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    h, w = bgr.shape[:2]
    digest = hashlib.sha256(bgr.tobytes() + f"|{w}x{h}".encode()).hexdigest()
    return digest, bgr


def _add_unique(pool: dict, record: dict, exclusions: list, why_dup: str):
    digest = record["pixel_sha256"]
    if digest in pool:
        exclusions.append({
            "reason": why_dup,
            "class": record["class"],
            "source": record["source_path"],
            "kept": pool[digest]["source_path"],
        })
        return False
    pool[digest] = record
    return True


def _split_by_hash(records: list[dict]) -> None:
    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["class"]].append(rec)
    for _cls, group in by_class.items():
        group.sort(key=lambda r: r["pixel_sha256"])
        n = len(group)
        if n == 1:
            group[0]["split"] = "train"
            continue
        if n == 2:
            group[0]["split"] = "train"
            group[1]["split"] = "val"
            continue
        n_test = max(1, int(round(n * 0.15)))
        n_val = max(1, int(round(n * 0.15)))
        if n_test + n_val >= n:
            n_test, n_val = 1, 1
        for i, rec in enumerate(group):
            if i < n - n_test - n_val:
                rec["split"] = "train"
            elif i < n - n_test:
                rec["split"] = "val"
            else:
                rec["split"] = "test"


def build() -> dict:
    global CLASSES
    if os.path.exists(EFFNET_PROCESSED_ROOT):
        shutil.rmtree(EFFNET_PROCESSED_ROOT)
    img_root = os.path.join(EFFNET_PROCESSED_ROOT, "images")
    os.makedirs(img_root, exist_ok=True)

    pool: dict[str, dict] = {}
    exclusions: list[dict] = []

    yolo_man = os.path.join("data", "datasets", "yolo_processed", "manifest.csv")
    if os.path.exists(yolo_man):
        with open(yolo_man, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                kept = [c.strip() for c in (row.get("kept_classes") or "").split(",") if c.strip()]
                if len(kept) != 1 or kept[0] not in ("cut", "bruise"):
                    continue
                src = row["dest_image"]
                if not os.path.exists(src):
                    continue
                digest, _ = _pixel_sha256(src)
                if digest is None:
                    continue
                rec = {
                    "sample_id": "yolo_" + row["sample_id"],
                    "class": kept[0],
                    "split": row["split"],
                    "source_dataset": "yolo_processed",
                    "source_path": _rel(src),
                    "provenance": row.get("provenance") or "SYNTHETIC",
                    "pixel_sha256": digest,
                    "negative": False,
                }
                _add_unique(pool, rec, exclusions, "exact_pixel_duplicate")

    pub_man = os.path.join("data", "datasets", "public_wound_dataset", "manifest.csv")
    if os.path.exists(pub_man):
        with open(pub_man, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    idx = int(str(row["sample_id"]).split("_")[-1])
                except ValueError:
                    continue
                visual = PUBLIC_VISUAL[(idx - 1) % 6]
                if visual not in ("cut", "bruise", "swelling"):
                    exclusions.append({
                        "reason": "dropped_not_remapped",
                        "class": visual,
                        "source": row.get("image_path"),
                    })
                    continue
                src = os.path.join("data", "datasets", "public_wound_dataset", row["image_path"].replace("/", os.sep))
                if not os.path.exists(src):
                    continue
                digest, _ = _pixel_sha256(src)
                if digest is None:
                    continue
                rec = {
                    "sample_id": "public_" + row["sample_id"],
                    "class": visual,
                    "split": None,
                    "source_dataset": "public_wound_dataset",
                    "source_path": _rel(src),
                    "provenance": "SYNTHETIC",
                    "pixel_sha256": digest,
                    "negative": False,
                }
                _add_unique(pool, rec, exclusions, "exact_pixel_duplicate")

    inj_man = os.path.join("data", "datasets", "injury_dataset", "manifest.csv")
    if os.path.exists(inj_man):
        with open(inj_man, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cls = (row.get("class") or "").strip().lower()
                if cls not in ("cut", "bruise", "swelling"):
                    continue
                src = os.path.join("data", "datasets", "injury_dataset", row["image_path"].replace("/", os.sep))
                if not os.path.exists(src):
                    continue
                digest, _ = _pixel_sha256(src)
                if digest is None:
                    continue
                rec = {
                    "sample_id": "injury_" + row["sample_id"],
                    "class": cls,
                    "split": None,
                    "source_dataset": "injury_dataset",
                    "source_path": _rel(src),
                    "provenance": "SYNTHETIC",
                    "pixel_sha256": digest,
                    "negative": False,
                }
                _add_unique(pool, rec, exclusions, "exact_pixel_duplicate")

    unique_by_class = Counter(r["class"] for r in pool.values())
    dropped_insufficient = {
        cls: n for cls, n in unique_by_class.items() if n < MIN_UNIQUE_FOR_SPLIT
    }
    kept_classes = [cls for cls in ("cut", "bruise", "swelling") if unique_by_class.get(cls, 0) >= MIN_UNIQUE_FOR_SPLIT]
    if not kept_classes:
        raise RuntimeError("No class has enough unique images to train.")
    CLASSES = kept_classes

    removed = []
    for digest, rec in list(pool.items()):
        if rec["class"] not in kept_classes:
            removed.append({
                "reason": "class_insufficient_unique_for_honest_split",
                "class": rec["class"],
                "source": rec["source_path"],
                "unique_count": unique_by_class[rec["class"]],
            })
            del pool[digest]

    # Detection splits from YOLO are not a classification split. Re-split every
    # unique image by pixel hash so train/val/test are hash-disjoint per class.
    for rec in pool.values():
        rec["split"] = None
    _split_by_hash(list(pool.values()))
    records = list(pool.values())

    final_rows = []
    for rec in records:
        split = rec["split"]
        dest_dir = os.path.join(img_root, split)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, rec["sample_id"] + ".jpg")
        src = rec["source_path"]
        if os.path.normpath(os.path.abspath(src)) != os.path.normpath(os.path.abspath(dest)):
            shutil.copy2(src, dest)
        rec = dict(rec)
        rec["image_path"] = _rel(dest)
        rec["negative"] = False
        final_rows.append(rec)

    final_rows.sort(key=lambda r: (r["split"], r["class"], r["sample_id"]))

    ood_eval = []
    for rel, stem, src_ds in EXISTING_NEGATIVES:
        if not os.path.isfile(rel):
            continue
        digest, _ = _pixel_sha256(rel)
        ood_eval.append({
            "sample_id": stem,
            "path": _rel(rel),
            "source_dataset": src_ds,
            "provenance": "EXISTING_FILE_NOT_A_TRAINING_CLASS",
            "pixel_sha256": digest,
            "used_as_training_label": False,
            "note": "Too few existing no-injury files to train a normal/reject class. Eval-only.",
        })

    os.makedirs(EFFNET_PROCESSED_ROOT, exist_ok=True)
    fields = [
        "sample_id", "split", "class", "image_path", "source_dataset",
        "source_path", "provenance", "pixel_sha256", "negative",
    ]
    with open(EFFNET_PROCESSED_MANIFEST, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(final_rows)

    with open(os.path.join(EFFNET_PROCESSED_ROOT, "ood_eval.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ood_eval[0].keys()) if ood_eval else ["sample_id"])
        writer.writeheader()
        writer.writerows(ood_eval)

    split_hashes = defaultdict(set)
    for rec in final_rows:
        split_hashes[rec["split"]].add(rec["pixel_sha256"])
    overlap = {
        "train_val": sorted(split_hashes["train"] & split_hashes["val"]),
        "train_test": sorted(split_hashes["train"] & split_hashes["test"]),
        "val_test": sorted(split_hashes["val"] & split_hashes["test"]),
    }
    counts = {
        split: dict(Counter(r["class"] for r in final_rows if r["split"] == split))
        for split in ("train", "val", "test")
    }
    missing_split = []
    for cls in CLASSES:
        for split in ("train", "val", "test"):
            if counts.get(split, {}).get(cls, 0) < 1:
                missing_split.append(f"{cls}:{split}")
    if missing_split:
        raise RuntimeError(
            "Honest split failed; class missing a split: " + ", ".join(missing_split)
        )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": _rel(EFFNET_PROCESSED_ROOT),
        "manifest": _rel(EFFNET_PROCESSED_MANIFEST),
        "did_not_fabricate_negatives": True,
        "classes": CLASSES,
        "unique_before_class_filter": dict(unique_by_class),
        "dropped_insufficient_unique": dropped_insufficient,
        "n": len(final_rows),
        "split_sizes": {s: sum(1 for r in final_rows if r["split"] == s) for s in ("train", "val", "test")},
        "class_counts_by_split": counts,
        "class_counts": dict(Counter(r["class"] for r in final_rows)),
        "existing_negative_files_eval_only": ood_eval,
        "trained_normal_class": False,
        "hash_overlap": overlap,
        "leakage_free": all(len(v) == 0 for v in overlap.values()),
        "exclusions_n": len(exclusions) + len(removed),
        "dropped_not_remapped": sum(1 for e in exclusions if e.get("reason") == "dropped_not_remapped"),
        "provenance": "SYNTHETIC",
        "known_limitations": [
            "Images are synthetic drawings, not clinical photography.",
            "No legitimate labeled no-injury dataset was available. blank_skin.jpg and dummy_test.jpg are eval-only (n=2).",
            "A normal/reject class was not trained because negatives were not fabricated.",
            "Closed-set softmax can still collapse OOD inputs onto an injury class. Input-quality gates remain required.",
        ],
    }
    with open(os.path.join(EFFNET_PROCESSED_ROOT, "processing_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "summary": summary,
            "exclusions_sample": (exclusions + removed)[:80],
        }, handle, indent=2)
    with open(os.path.join(EFFNET_PROCESSED_ROOT, "taxonomy.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "trained_classes": CLASSES,
            "dropped_insufficient_unique": dropped_insufficient,
            "not_remapped": ["abrasion", "laceration", "burn"],
            "reject_or_normal_class": None,
            "existing_negatives_eval_only": [e["path"] for e in ood_eval],
            "reason_no_reject_class": (
                "Only two existing no-injury files were found (blank_skin.jpg, dummy_test.jpg). "
                "They are eval-only. Negatives were not fabricated. Closed-set softmax plus "
                "input-quality/confidence gates is the rejection mechanism."
            ),
        }, handle, indent=2)
    with open(os.path.join(EFFNET_PROCESSED_ROOT, "split.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "class", "pixel_sha256", "image_path"])
        writer.writeheader()
        for rec in final_rows:
            writer.writerow({k: rec[k] for k in ["sample_id", "split", "class", "pixel_sha256", "image_path"]})

    print(json.dumps(summary, indent=2))
    if not summary["leakage_free"]:
        raise RuntimeError("Hash overlap across splits; refusing to write a leaking set.")
    return summary


if __name__ == "__main__":
    build()
