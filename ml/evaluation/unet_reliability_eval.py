"""U-Net dataset audit + blank/OOD segmentation evaluation.

Raw positive area is always recorded. Gated overlay is withheld on invalid input.
False-positive masks are not hidden.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.vision.unet_wrapper import UNetSegmenter, interpret_segmentation
from ml.vision.input_quality import assess_input_quality

ROOT = os.path.join("data", "datasets", "public_wound_dataset")
MANIFEST = os.path.join(ROOT, "manifest.csv")
OUT_DIR = os.path.join("ml", "models", "unet_reliability")
OUT_JSON = os.path.join(OUT_DIR, "UNET_RELIABILITY_REPORT.json")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dhash(img, size=16) -> str:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return "".join("1" if v else "0" for v in diff.flatten())


def audit_dataset():
    if not os.path.exists(MANIFEST):
        raise FileNotFoundError(MANIFEST)
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    by_split = defaultdict(list)
    by_class = Counter()
    split_class = defaultdict(Counter)
    subjects = defaultdict(set)
    hashes = defaultdict(list)
    dhashes = defaultdict(list)
    mask_hashes = defaultdict(list)
    missing_img = 0
    missing_mask = 0
    empty_masks = 0
    dim_mismatch = 0
    mask_areas = []
    nonbinary = 0
    alignment = []

    for row in rows:
        split = row["split"]
        cls = row["class"]
        by_split[split].append(row)
        by_class[cls] += 1
        split_class[split][cls] += 1
        subjects[split].add(row.get("subject_id") or "")
        img_path = os.path.join(ROOT, row["image_path"].replace("/", os.sep))
        mask_path = os.path.join(ROOT, row["mask_path"].replace("/", os.sep))
        if not os.path.exists(img_path):
            missing_img += 1
            continue
        if not os.path.exists(mask_path):
            missing_mask += 1
            continue
        with open(img_path, "rb") as handle:
            digest = _sha256_bytes(handle.read())
        hashes[digest].append((split, row["sample_id"], cls))
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            missing_img += 1
            continue
        dhashes[_dhash(img)].append((split, row["sample_id"], cls))
        mask_hashes[_sha256_bytes(mask.tobytes())].append((split, row["sample_id"], cls))
        if img.shape[:2] != mask.shape[:2]:
            dim_mismatch += 1
        unique = np.unique(mask)
        if not set(unique.tolist()).issubset({0, 255, 1}):
            if unique.size > 2:
                nonbinary += 1
        area = float((mask > 127).mean())
        mask_areas.append(area)
        if area == 0.0:
            empty_masks += 1
        # crude alignment: mask should overlap the darker/redder wound mark
        alignment.append({
            "id": row["sample_id"],
            "class": cls,
            "split": split,
            "mask_area": round(area, 6),
            "image_hw": [int(img.shape[0]), int(img.shape[1])],
            "mask_hw": [int(mask.shape[0]), int(mask.shape[1])],
        })

    exact_dup_groups = {h: v for h, v in hashes.items() if len(v) > 1}
    leak_exact = []
    for group in exact_dup_groups.values():
        splits = {s for s, _, _ in group}
        if len(splits) > 1:
            leak_exact.append([(s, sid, c) for s, sid, c in group])
    leak_near = 0
    for group in dhashes.values():
        if len({s for s, _, _ in group}) > 1:
            leak_near += 1

    areas = np.array(mask_areas, dtype=np.float64) if mask_areas else np.array([0.0])
    return {
        "n_manifest": len(rows),
        "missing_images": missing_img,
        "missing_masks": missing_mask,
        "dimension_mismatches": dim_mismatch,
        "empty_masks": empty_masks,
        "nonbinary_masks": nonbinary,
        "negative_no_wound_examples": empty_masks,
        "class_balance_all": dict(by_class),
        "split_sizes": {k: len(v) for k, v in by_split.items()},
        "split_class_counts": {k: dict(v) for k, v in split_class.items()},
        "subject_overlap": {
            "train_val": sorted(subjects["train"] & subjects["val"]),
            "train_test": sorted(subjects["train"] & subjects["test"]),
            "val_test": sorted(subjects["val"] & subjects["test"]),
        },
        "exact_duplicate_groups": len(exact_dup_groups),
        "exact_duplicate_images": sum(len(v) for v in exact_dup_groups.values()),
        "cross_split_exact_duplicate_groups": len(leak_exact),
        "cross_split_near_duplicate_group_count": leak_near,
        "mask_area": {
            "min": round(float(areas.min()), 6),
            "max": round(float(areas.max()), 6),
            "mean": round(float(areas.mean()), 6),
            "p50": round(float(np.median(areas)), 6),
        },
        "source_field": rows[0]["source"] if rows else None,
        "known_synthetic_generator": "ml/training/download_public_datasets.py generate_expanded_wound_dataset",
        "leakage_free": len(leak_exact) == 0,
        "image_mask_pairs_ok": missing_img == 0 and missing_mask == 0 and dim_mismatch == 0,
    }


def _probe_row(seg: UNetSegmenter, name: str, group: str, img: np.ndarray) -> dict:
    quality = assess_input_quality(img)
    raw = seg.segment_raw(img)
    mask, count, ratio, gated = seg.segment(img)
    parsed = interpret_segmentation(mask, count, ratio, gated)
    return {
        "name": name,
        "group": group,
        "quality_status": quality["status"],
        "quality_reason": quality["reason"],
        "quality_metrics": quality.get("metrics", {}),
        "raw": {
            "positive_ratio": raw.get("positive_ratio"),
            "mean_prob": raw.get("mean_prob"),
            "max_prob": raw.get("max_prob"),
            "min_prob": raw.get("min_prob"),
        },
        "gated": {
            "status": parsed["status"],
            "reason": parsed["reason"],
            "is_reliable": parsed["is_reliable"],
            "mask_withheld": parsed["mask_withheld"],
            "displayed_positive_ratio": float(mask.mean()) if mask is not None else 0.0,
            "displayed_pixel_count": parsed["pixel_count"],
            "affected_ratio": parsed["affected_ratio"],
            "raw_positive_ratio": parsed["raw_positive_ratio"],
            "false_positive_area": parsed["false_positive_area"],
        },
        "failure_hidden": bool(
            (raw.get("positive_ratio") or 0) > 0.25
            and parsed["raw_positive_ratio"] in (None, 0, 0.0)
        ),
    }


def evaluate_probes(seg: UNetSegmenter):
    rng = np.random.default_rng(0)
    probes = []

    def add(name, group, img):
        probes.append(_probe_row(seg, name, group, img))

    add("black", "uniform", np.zeros((224, 224, 3), dtype=np.uint8))
    add("white", "uniform", np.full((224, 224, 3), 255, dtype=np.uint8))
    add("gray", "uniform", np.full((224, 224, 3), 180, dtype=np.uint8))
    add(
        "noisy_gray_std5",
        "uniform",
        np.clip(np.full((224, 224, 3), 180) + rng.normal(0, 5, (224, 224, 3)), 0, 255).astype(np.uint8),
    )
    add("uniform_skin", "normal_skin", np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8))
    add(
        "noisy_skin",
        "normal_skin",
        np.clip(np.full((224, 224, 3), (185, 145, 125)) + rng.normal(0, 4, (224, 224, 3)), 0, 255).astype(np.uint8),
    )

    blank_path = os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")
    if os.path.exists(blank_path):
        bgr = cv2.imread(blank_path)
        add("blank_skin.jpg", "normal_skin", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    id_img = np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8)
    cv2.line(id_img, (70, 40), (150, 180), (190, 20, 20), 7)
    add("synthetic_cut_template", "wound", id_img)
    add("blurred_cut", "blurred", cv2.GaussianBlur(id_img, (31, 31), 8))
    add("heavily_blurred_cut", "blurred", cv2.GaussianBlur(id_img, (51, 51), 16))

    add("blue_unrelated", "unrelated", np.full((224, 224, 3), (20, 60, 200), dtype=np.uint8))
    add("green_unrelated", "unrelated", np.full((224, 224, 3), (20, 180, 40), dtype=np.uint8))
    striped = np.zeros((224, 224, 3), dtype=np.uint8)
    striped[:, :, 2] = 200
    striped[::6, :, 1] = 180
    add("blue_striped_unrelated", "unrelated", striped)
    add("high_frequency_noise", "unrelated", rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))

    demo = os.path.join("data", "sample", "image", "football_injury.jpg")
    if os.path.exists(demo):
        bgr = cv2.imread(demo)
        add("football_injury.jpg", "wound", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    val_rows = [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["split"] == "val"]
    for row in val_rows[:3]:
        path = os.path.join(ROOT, row["image_path"].replace("/", os.sep))
        if os.path.exists(path):
            bgr = cv2.imread(path)
            add(f"val_{row['sample_id']}_{row['class']}", "wound", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    return probes


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    audit = audit_dataset()
    seg = UNetSegmenter()
    probes = evaluate_probes(seg) if seg.is_loaded else []

    still_reliable_invalid = [
        row["name"]
        for row in probes
        if row["group"] in {"uniform", "normal_skin", "unrelated", "blurred"}
        and row["gated"]["is_reliable"]
    ]
    hidden_fp = [row["name"] for row in probes if row["failure_hidden"]]
    blank_fp = {
        row["name"]: {
            "raw_positive_ratio": row["raw"]["positive_ratio"],
            "gated_status": row["gated"]["status"],
            "displayed_positive_ratio": row["gated"]["displayed_positive_ratio"],
            "false_positive_area": row["gated"]["false_positive_area"],
        }
        for row in probes
        if row["group"] in {"uniform", "normal_skin", "unrelated", "blurred"}
    }

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "not_clinical_accuracy": True,
        "weights_status": "MODEL_OUTPUT_NOT_TRUSTWORTHY",
        "inference_gates_only": True,
        "dataset_audit": audit,
        "loss_function": "BCE_plus_Dice (decoder fine-tune, encoder frozen)",
        "thresholding": "fixed 0.5; adaptive 0.85*max removed because it created extra false positives",
        "postprocessing": "3x3 morphological opening; largest connected component if it is >=80% of positives; max area 0.70",
        "probes": probes,
        "blank_ood_false_positive_area": blank_fp,
        "still_reliable_on_invalid": still_reliable_invalid,
        "hidden_false_positives": hidden_fp,
        "remaining_dataset_gaps": [
            "Zero empty masks and zero labeled no-wound examples in public_wound_dataset.",
            "All 200 images are PIL drawings from generate_expanded_wound_dataset, not clinical photos.",
            "Exact/near-duplicate templates span train/val/test (subject IDs are disjoint; pixels are not).",
            "Abrasion/laceration/burn were remapped to swelling in the manifest class field.",
            "Held-out Dice ~0.98 is template reconstruction, not generalization.",
        ],
        "retraining_recommendation": [
            "Do not promote new U-Net weights until blank/white/black raw positive_ratio is < 0.05 without inference gates.",
            "Add empty-mask negatives: uniform fills, blank skin, cluttered non-injury photos.",
            "Keep image/mask pairs unique across splits; drop exact duplicates.",
            "Train with empty-mask examples in the Dice+BCE loss so background is a first-class class.",
            "Keep current production weights plus these gates until that dataset exists.",
        ],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {OUT_JSON}")
    print("empty_masks", audit["empty_masks"], "negatives", audit["negative_no_wound_examples"])
    print("mask_area", audit["mask_area"])
    print("cross_split_exact", audit["cross_split_exact_duplicate_groups"], "near", audit["cross_split_near_duplicate_group_count"])
    print("still_reliable_on_invalid", still_reliable_invalid)
    print("hidden_false_positives", hidden_fp)
    for row in probes:
        r = row["raw"]
        g = row["gated"]
        print(
            f"{row['name']:32s} raw_pos={r.get('positive_ratio')}  "
            f"gated={g['status']} displayed={g['displayed_positive_ratio']:.4f} "
            f"fp_area={g['false_positive_area']} reliable={g['is_reliable']}"
        )


if __name__ == "__main__":
    main()
