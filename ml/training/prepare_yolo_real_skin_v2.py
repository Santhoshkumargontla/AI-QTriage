"""Build yolo_real_skin_v2: wound_boxes_v1 + confirmed skin negatives (empty labels).

Classes (production taxonomy): cut, bruise, wound
- cut/bruise: honest boxes from yolo_retrain_v2 (mostly SYNTHETIC drawings)
- wound: mask-derived boxes from public ulcer datasets (PUBLIC_REAL_PHOTOS)
- negatives: ibrahimfateen Normal + efficientnet normal patches (CONFIRMED_NEGATIVE, empty labels)

Does NOT fabricate cut/bruise boxes from classification folders.
Does NOT include external forensic benchmark images in train/val/test.
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
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ml.models.canonical_paths import ROOT
from ml.training.prepare_yolo_wound_boxes_v1 import build as build_wound_boxes

SEED = 42
OUT_ROOT = os.path.join("data", "datasets", "yolo_real_skin_v2")
NAMES = {0: "cut", 1: "bruise", 2: "wound"}
BENCHMARK_ROOT = os.path.join("data", "benchmarks", "yolo_real_skin_regression_v1")

# External forensic cases — never used as training samples.
FORENSIC_EXCLUDE_SHA256 = set()


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_bucket(digest: str) -> str:
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def _register_forensic_excludes() -> None:
    """Hash-lock known forensic uploads so they cannot enter YOLO splits."""
    candidates = [
        os.path.join(ROOT, "data", "uploads", "3f629ca8-dd98-427d-a708-f976e2042555.jpeg"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            FORENSIC_EXCLUDE_SHA256.add(_file_sha(path))


def _add_confirmed_negatives(records: list, seen_hash: set, stats: Counter) -> None:
    """Empty-label images with provenance CONFIRMED_NEGATIVE."""
    sources = []

    normal_dir = os.path.join(
        ROOT,
        "data/raw/kaggle/ibrahimfateen_wound_classification/Wound_dataset copy/Normal",
    )
    if os.path.isdir(normal_dir):
        for name in os.listdir(normal_dir):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                sources.append((os.path.join(normal_dir, name), "ibrahimfateen_normal", "CONFIRMED_NEGATIVE"))

    eff_normal = os.path.join(ROOT, "data/datasets/efficientnet_kaggle_v1")
    for split in ("train", "val", "test"):
        nd = os.path.join(eff_normal, split, "normal")
        if not os.path.isdir(nd):
            continue
        for name in os.listdir(nd):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                sources.append((os.path.join(nd, name), "efficientnet_kaggle_normal", "CONFIRMED_NEGATIVE"))

    for blank in (
        "data/datasets/yolo_injury/blank_skin.jpg",
        "data/datasets/yolo_injury/dummy_test.jpg",
    ):
        bp = os.path.join(ROOT, blank)
        if os.path.isfile(bp):
            sources.append((bp, "synthetic_blank", "SYNTHETIC_BLANK_NEGATIVE"))

    for ip, source_dataset, provenance in sources:
        digest = _file_sha(ip)
        if digest in seen_hash or digest in FORENSIC_EXCLUDE_SHA256:
            continue
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        seen_hash.add(digest)
        stem = hashlib.sha256((digest + source_dataset).encode()).hexdigest()[:16]
        records.append({
            "sample_id": f"neg_{source_dataset}_{stem}",
            "subject_id": f"neg:{source_dataset}:{stem}",
            "class_ids": [],
            "lines": [],
            "image_bgr": bgr,
            "provenance": provenance,
            "source_dataset": source_dataset,
            "source_path": _rel(ip),
            "pixel_sha256": digest,
            "empty": True,
            "split": _split_bucket(digest),
        })
        stats["negative_images"] += 1


def build(force_rebuild_wound: bool = False) -> dict:
    _register_forensic_excludes()
    wound_root = os.path.join(ROOT, "data/datasets/yolo_wound_boxes_v1", "manifest.csv")
    if force_rebuild_wound or not os.path.isfile(wound_root):
        build_wound_boxes()

    # Load wound_boxes manifest into memory records
    records = []
    seen_hash = set()
    stats = Counter()
    with open(wound_root, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            digest = row.get("pixel_sha256") or ""
            if digest in FORENSIC_EXCLUDE_SHA256:
                stats["excluded_forensic"] += 1
                continue
            img_path = os.path.join(ROOT, (row["dest_image"] or "").replace("/", os.sep))
            lbl_path = os.path.join(ROOT, (row["dest_label"] or "").replace("/", os.sep))
            if not os.path.isfile(img_path):
                continue
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            if digest:
                seen_hash.add(digest)
            lines = []
            if os.path.isfile(lbl_path):
                lines = [ln.strip() for ln in open(lbl_path, encoding="utf-8") if ln.strip()]
            class_ids = sorted({int(float(ln.split()[0])) for ln in lines}) if lines else []
            records.append({
                "sample_id": row["sample_id"],
                "subject_id": row["subject_id"],
                "class_ids": class_ids,
                "lines": lines,
                "image_bgr": bgr,
                "provenance": row.get("provenance") or "UNKNOWN",
                "source_dataset": row.get("source_dataset") or "yolo_wound_boxes_v1",
                "source_path": row.get("source_path") or row["dest_image"],
                "pixel_sha256": digest or _file_sha(img_path),
                "empty": len(lines) == 0,
                "split": row["split"],
            })

    _add_confirmed_negatives(records, seen_hash, stats)

    # Write YOLO layout
    if os.path.isdir(os.path.join(ROOT, OUT_ROOT)):
        shutil.rmtree(os.path.join(ROOT, OUT_ROOT))
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(ROOT, OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(ROOT, OUT_ROOT, "labels", split), exist_ok=True)

    man_rows = []
    box_counts = Counter()
    split_box = defaultdict(lambda: Counter())
    split_img = Counter()
    for rec in records:
        split = rec["split"]
        stem = rec["sample_id"]
        img_out = os.path.join(ROOT, OUT_ROOT, "images", split, f"{stem}.jpg")
        lbl_out = os.path.join(ROOT, OUT_ROOT, "labels", split, f"{stem}.txt")
        cv2.imwrite(img_out, rec["image_bgr"])
        with open(lbl_out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(rec["lines"]) + ("\n" if rec["lines"] else ""))
        split_img[split] += 1
        for ln in rec["lines"]:
            cid = int(float(ln.split()[0]))
            box_counts[NAMES[cid]] += 1
            split_box[split][NAMES[cid]] += 1
        man_rows.append({
            "sample_id": stem,
            "split": split,
            "subject_id": rec["subject_id"],
            "dest_image": _rel(img_out),
            "dest_label": _rel(lbl_out),
            "source_dataset": rec["source_dataset"],
            "source_path": rec["source_path"],
            "provenance": rec["provenance"],
            "pixel_sha256": rec["pixel_sha256"],
            "n_boxes": len(rec["lines"]),
            "class_ids": ",".join(str(c) for c in rec["class_ids"]),
            "is_negative": int(rec["empty"]),
        })

    yaml_path = os.path.join(ROOT, OUT_ROOT, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump({
            "path": os.path.abspath(os.path.join(ROOT, OUT_ROOT)).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": NAMES,
            "nc": 3,
        }, handle, sort_keys=False)

    man_path = os.path.join(ROOT, OUT_ROOT, "manifest.csv")
    with open(man_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(man_rows[0].keys()))
        writer.writeheader()
        writer.writerows(man_rows)

    # Leakage audit
    by_split_hash = defaultdict(set)
    for r in man_rows:
        by_split_hash[r["split"]].add(r["pixel_sha256"])
    leakage = {
        "train_val": len(by_split_hash["train"] & by_split_hash["val"]),
        "train_test": len(by_split_hash["train"] & by_split_hash["test"]),
        "val_test": len(by_split_hash["val"] & by_split_hash["test"]),
    }

    os.makedirs(os.path.join(ROOT, BENCHMARK_ROOT), exist_ok=True)
    benchmark = {
        "version": "yolo_real_skin_regression_v1",
        "forensic_cases": [
            {
                "case_id": "3f629ca8-dd98-427d-a708-f976e2042555",
                "image_path": "data/uploads/3f629ca8-dd98-427d-a708-f976e2042555.jpeg",
                "role": "external_forensic_positive_cut",
                "ground_truth_class": "cut",
                "ground_truth_type": "visual_cut_on_hand",
                "in_training_set": False,
                "note": "Not used for training or threshold tuning. Approximate IoU uses color-heuristic injury region.",
            }
        ],
        "confirmed_negatives_glob": "data/datasets/yolo_real_skin_v2/manifest.csv (is_negative=1)",
    }
    with open(os.path.join(ROOT, BENCHMARK_ROOT, "benchmark_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(benchmark, handle, indent=2)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": OUT_ROOT,
        "yaml": _rel(yaml_path),
        "manifest_sha256": _file_sha(man_path),
        "n": len(man_rows),
        "split_image_counts": dict(split_img),
        "box_counts_total": dict(box_counts),
        "box_counts_by_split": {k: dict(v) for k, v in split_box.items()},
        "negative_images_added": stats["negative_images"],
        "excluded_forensic": stats["excluded_forensic"],
        "provenance_counts": dict(Counter(r["provenance"] for r in man_rows)),
        "names": NAMES,
        "leakage_exact_hash": leakage,
        "leakage_free": all(v == 0 for v in leakage.values()),
        "class_support_honest": {
            "cut": {"supported": box_counts["cut"] > 0, "total_boxes": box_counts["cut"]},
            "bruise": {"supported": box_counts["bruise"] > 0, "total_boxes": box_counts["bruise"]},
            "wound": {"supported": box_counts["wound"] > 0, "total_boxes": box_counts["wound"]},
        },
        "known_limitations": [
            "cut/bruise boxes remain mostly SYNTHETIC drawings — no honest Kaggle cut/bruise YOLO labels available.",
            "wound boxes are mask-derived chronic ulcer regions — not sports/linear cuts.",
            "Real cut localization on phone photos requires REQUIRES_MORE_DATA with honest cut bounding boxes.",
            "Confirmed negatives reduce false positives but do not create cut localization labels.",
        ],
    }
    report_path = os.path.join(ROOT, OUT_ROOT, "PREPARE_REPORT.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
