"""Build a leakage-free U-Net image/mask set. Does not fabricate negatives.

Does not modify raw sources. Exact pixel duplicates are kept once.
Splits are hash-disjoint.

Existing blank_skin.jpg and dummy_test.jpg are eval-only empty-mask pairs.
Two files are not enough to train a background class. Empty medical
scenes and random empty textures are not generated.
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

from ml.models.canonical_paths import UNET_PROCESSED_MANIFEST, UNET_PROCESSED_ROOT

SEED = 42
MAX_POS_AREA = 0.85
MIN_POS_AREA = 0.005
EXISTING_NEGATIVES = (
    ("data/datasets/yolo_injury/blank_skin.jpg", "blank_skin", "yolo_injury"),
    ("data/datasets/yolo_injury/dummy_test.jpg", "dummy_test", "yolo_injury"),
)


def _rel(path: str) -> str:
    return path.replace("\\", "/")


def _pixel_sha256(bgr: np.ndarray) -> str:
    h, w = bgr.shape[:2]
    return hashlib.sha256(bgr.tobytes() + f"|{w}x{h}".encode()).hexdigest()


def _read_pair(img_path: str, mask_path: str):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None, None, "unreadable"
    if img.shape[:2] != mask.shape[:2]:
        return None, None, "dimension_mismatch"
    return img, mask, None


def _verify_positive(mask: np.ndarray) -> str | None:
    area = float((mask > 127).mean())
    if area < MIN_POS_AREA:
        return "empty_mask_on_positive"
    if area > MAX_POS_AREA:
        return "unreasonable_full_mask"
    return None


def _add_unique(pool: dict, digest: str, record: dict, exclusions: list, why: str) -> bool:
    if digest in pool:
        exclusions.append({
            "reason": why,
            "kind": record.get("kind"),
            "class": record.get("class"),
            "source": record.get("source_path"),
            "kept": pool[digest].get("source_path"),
        })
        return False
    pool[digest] = record
    return True


def _split_by_hash(records: list[dict]) -> None:
    records.sort(key=lambda r: r["pixel_sha256"])
    n = len(records)
    if n < 3:
        raise RuntimeError(f"Need at least 3 unique positives for an honest split; got {n}.")
    n_test = max(1, int(round(n * 0.15)))
    n_val = max(1, int(round(n * 0.15)))
    if n_test + n_val >= n:
        n_test, n_val = 1, 1
    for i, rec in enumerate(records):
        if i < n - n_test - n_val:
            rec["split"] = "train"
        elif i < n - n_test:
            rec["split"] = "val"
        else:
            rec["split"] = "test"


def _ingest_manifest(pool, exclusions, root, source_name, sample_prefix):
    path = os.path.join(root, "manifest.csv")
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            img_path = os.path.join(root, row["image_path"].replace("/", os.sep))
            mask_path = os.path.join(root, row["mask_path"].replace("/", os.sep))
            img, mask, err = _read_pair(img_path, mask_path)
            if err:
                exclusions.append({"reason": err, "source": _rel(img_path), "class": row.get("class")})
                continue
            v_err = _verify_positive(mask)
            if v_err:
                exclusions.append({"reason": v_err, "source": _rel(img_path), "class": row.get("class")})
                continue
            digest = _pixel_sha256(img)
            rec = {
                "sample_id": sample_prefix + row["sample_id"],
                "class": (row.get("class") or "wound").strip().lower(),
                "kind": "positive",
                "split": None,
                "source_dataset": source_name,
                "source_path": _rel(img_path),
                "source_mask": _rel(mask_path),
                "provenance": "SYNTHETIC",
                "pixel_sha256": digest,
                "empty_mask": False,
                "mask_area": round(float((mask > 127).mean()), 6),
                "_img": img,
                "_mask": mask,
            }
            _add_unique(pool, digest, rec, exclusions, "exact_pixel_duplicate")


def build() -> dict:
    if os.path.exists(UNET_PROCESSED_ROOT):
        shutil.rmtree(UNET_PROCESSED_ROOT)
    img_root = os.path.join(UNET_PROCESSED_ROOT, "images")
    mask_root = os.path.join(UNET_PROCESSED_ROOT, "masks")
    os.makedirs(img_root, exist_ok=True)
    os.makedirs(mask_root, exist_ok=True)

    pool: dict[str, dict] = {}
    exclusions: list[dict] = []

    _ingest_manifest(pool, exclusions, os.path.join("data", "datasets", "public_wound_dataset"), "public_wound_dataset", "public_")
    _ingest_manifest(pool, exclusions, os.path.join("data", "datasets", "injury_dataset"), "injury_dataset", "injury_")

    records = list(pool.values())
    _split_by_hash(records)

    final_rows = []
    for rec in records:
        split = rec["split"]
        dest_img_dir = os.path.join(img_root, split)
        dest_msk_dir = os.path.join(mask_root, split)
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_msk_dir, exist_ok=True)
        dest_img = os.path.join(dest_img_dir, rec["sample_id"] + ".jpg")
        dest_msk = os.path.join(dest_msk_dir, rec["sample_id"] + ".png")
        cv2.imwrite(dest_img, rec["_img"])
        cv2.imwrite(dest_msk, rec["_mask"])
        row = {k: rec[k] for k in rec if not k.startswith("_")}
        row["image_path"] = _rel(dest_img)
        row["mask_path"] = _rel(dest_msk)
        final_rows.append(row)

    final_rows.sort(key=lambda r: (r["split"], r["kind"], r["sample_id"]))

    ood_eval = []
    for rel, stem, src_ds in EXISTING_NEGATIVES:
        if not os.path.isfile(rel):
            continue
        bgr = cv2.imread(rel)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)
        digest = _pixel_sha256(bgr)
        ood_eval.append({
            "sample_id": stem,
            "path": _rel(rel),
            "source_dataset": src_ds,
            "provenance": "EXISTING_FILE_NOT_A_TRAINING_CLASS",
            "pixel_sha256": digest,
            "empty_mask": True,
            "used_as_training_label": False,
            "note": "Too few existing no-injury files to train an empty-mask class. Eval-only. Empty mask is all-zero, not a fabricated wound shape.",
            "image_hw": [int(h), int(w)],
            "empty_mask_sum": int(empty.sum()),
        })

    split_hashes = defaultdict(set)
    for rec in final_rows:
        split_hashes[rec["split"]].add(rec["pixel_sha256"])
    overlap = {
        "train_val": sorted(split_hashes["train"] & split_hashes["val"]),
        "train_test": sorted(split_hashes["train"] & split_hashes["test"]),
        "val_test": sorted(split_hashes["val"] & split_hashes["test"]),
    }

    def _count(split):
        return sum(1 for r in final_rows if r["split"] == split)

    pos_areas = [r["mask_area"] for r in final_rows]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": _rel(UNET_PROCESSED_ROOT),
        "manifest": _rel(UNET_PROCESSED_MANIFEST),
        "did_not_fabricate_negatives": True,
        "did_not_generate_fake_medical_masks": True,
        "n": len(final_rows),
        "positives": len(final_rows),
        "negatives": 0,
        "empty_masks_trained": 0,
        "split_sizes": {s: _count(s) for s in ("train", "val", "test")},
        "class_counts": dict(Counter(r["class"] for r in final_rows)),
        "kind_counts": dict(Counter(r["kind"] for r in final_rows)),
        "mask_area_positives": {
            "min": min(pos_areas) if pos_areas else 0,
            "max": max(pos_areas) if pos_areas else 0,
            "mean": float(np.mean(pos_areas) if pos_areas else 0),
        },
        "existing_negative_files_eval_only": ood_eval,
        "hash_overlap": overlap,
        "leakage_free": all(len(v) == 0 for v in overlap.values()),
        "exclusions_n": len(exclusions),
        "provenance": "SYNTHETIC",
        "known_limitations": [
            "Positive pairs are unique synthetic drawings, not clinical photography.",
            "No legitimate labeled no-injury dataset was available. blank_skin.jpg and dummy_test.jpg are eval-only (n=2).",
            "An empty-mask class was not trained because negatives were not fabricated.",
            "Closed-set U-Net can still paint foreground on OOD fills. Overlay gates remain required.",
        ],
    }

    os.makedirs(UNET_PROCESSED_ROOT, exist_ok=True)
    fieldnames = [
        "sample_id", "split", "kind", "class", "empty_mask", "mask_area",
        "image_path", "mask_path", "source_dataset", "source_path",
        "source_mask", "provenance", "pixel_sha256",
    ]
    with open(UNET_PROCESSED_MANIFEST, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)
    with open(os.path.join(UNET_PROCESSED_ROOT, "split.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "kind", "class", "empty_mask", "pixel_sha256", "image_path", "mask_path"])
        writer.writeheader()
        for rec in final_rows:
            writer.writerow({k: rec[k] for k in ["sample_id", "split", "kind", "class", "empty_mask", "pixel_sha256", "image_path", "mask_path"]})
    with open(os.path.join(UNET_PROCESSED_ROOT, "ood_eval.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ood_eval[0].keys()) if ood_eval else ["sample_id"])
        writer.writeheader()
        writer.writerows(ood_eval)
    with open(os.path.join(UNET_PROCESSED_ROOT, "taxonomy.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "task": "binary_wound_mask",
            "trained_classes": ["wound_mask"],
            "empty_mask_class_trained": False,
            "existing_negatives_eval_only": [e["path"] for e in ood_eval],
            "reason_no_empty_class": (
                "Only two existing no-injury files were found. They are eval-only. "
                "Empty medical scenes and random empty textures were not generated."
            ),
        }, handle, indent=2)
    with open(os.path.join(UNET_PROCESSED_ROOT, "processing_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "exclusions_sample": exclusions[:80]}, handle, indent=2)

    print(json.dumps(summary, indent=2))
    if not summary["leakage_free"]:
        raise RuntimeError("Hash overlap across splits; refusing to write a leaking set.")
    if _count("train") < 1 or _count("val") < 1 or _count("test") < 1:
        raise RuntimeError("A split is empty after unique-hash assignment.")
    return summary


if __name__ == "__main__":
    build()
