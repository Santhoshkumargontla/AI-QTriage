"""
AI-QTriage — Real Dataset Preparation & Deduplication Script (Phase 2)
Builds data/datasets/yolo_real_wound from real photographic wound datasets.
Enforces SHA256 exact deduplication, perceptual hashing (ImageHash), bounding box validation,
and strict 70/15/15 train/val/held-out test split with zero data leakage.
"""

import os
import sys
import glob
import hashlib
import cv2
import numpy as np
import pandas as pd
import imagehash
from PIL import Image
import yaml
import shutil

# Random Seed for Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(ROOT_DIR, "data", "datasets", "public_wound_dataset")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "datasets", "yolo_real_wound")

REAL_TAXONOMY = {
    "cut": 0,
    "bruise": 1,
    "swelling": 2, # Mapped to general wound / lesion region
    "wound": 2
}

CLASS_NAMES = ["cut", "bruise", "wound"]

def compute_sha256(file_path: str) -> str:
    """Computes exact SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_phash(file_path: str) -> str:
    """Computes perceptual hash (pHash) of an image."""
    try:
        with Image.open(file_path) as img:
            return str(imagehash.phash(img))
    except (OSError, ValueError, Image.UnidentifiedImageError):
        return "0000000000000000"

def extract_yolo_bbox(mask_path: str, img_w: int, img_h: int) -> list:
    """
    Extracts normalized YOLO bounding boxes [class_id, xc, yc, w, h] from ground-truth mask.
    """
    if not os.path.exists(mask_path):
        return []

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or np.sum(mask > 0) == 0:
        return []

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 2 or h < 2:
            continue
        # Normalize coordinates
        xc = round((x + w / 2.0) / img_w, 6)
        yc = round((y + h / 2.0) / img_h, 6)
        nw = round(w / float(img_w), 6)
        nh = round(h / float(img_h), 6)

        # Bounds validation
        if 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < nw <= 1.0 and 0.0 < nh <= 1.0:
            boxes.append((xc, yc, nw, nh))

    return boxes

def prepare_dataset():
    print("=================================================================")
    print("AI-QTriage — REAL DATASET PREPARATION & DEDUPLICATION (PHASE 2)")
    print("=================================================================")

    manifest_path = os.path.join(SOURCE_DIR, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Source manifest not found at {manifest_path}")

    df_src = pd.read_csv(manifest_path)
    print(f"Loaded source dataset manifest: {len(df_src)} samples.")

    # 1. Deduplication via SHA256 & pHash
    sha256_seen = set()
    phash_seen = set()
    cleaned_samples = []

    duplicates_removed = 0
    corrupt_removed = 0
    invalid_boxes_removed = 0

    for idx, row in df_src.iterrows():
        img_rel = row["image_path"]
        img_full = os.path.join(SOURCE_DIR, img_rel) if not os.path.isabs(img_rel) else img_rel

        if not os.path.exists(img_full):
            corrupt_removed += 1
            continue

        # Check image validity & size
        img = cv2.imread(img_full)
        if img is None:
            corrupt_removed += 1
            continue

        img_h, img_w = img.shape[:2]

        # SHA256 Hash
        sha_hash = compute_sha256(img_full)
        if sha_hash in sha256_seen:
            duplicates_removed += 1
            continue
        sha256_seen.add(sha_hash)

        # pHash Hash
        p_hash = compute_phash(img_full)
        if p_hash in phash_seen:
            duplicates_removed += 1
            continue
        phash_seen.add(p_hash)

        # Extract & Validate Bounding Boxes
        mask_rel = row["mask_path"]
        mask_full = os.path.join(SOURCE_DIR, mask_rel) if not os.path.isabs(mask_rel) else mask_rel
        raw_class = str(row["class"]).lower()

        class_id = REAL_TAXONOMY.get(raw_class, 2)
        final_class = CLASS_NAMES[class_id]

        bboxes = extract_yolo_bbox(mask_full, img_w, img_h)
        if not bboxes:
            invalid_boxes_removed += 1
            continue

        subject_id = row.get("subject_id", f"subj_{idx:03d}")
        cleaned_samples.append({
            "sample_id": row["sample_id"],
            "subject_id": subject_id,
            "orig_path": img_full,
            "raw_class": raw_class,
            "final_class": final_class,
            "class_id": class_id,
            "bboxes": bboxes,
            "img_w": img_w,
            "img_h": img_h,
            "sha256": sha_hash,
            "phash": p_hash
        })

    print(f"[QUALITY FILTERING RESULTS]")
    print(f"  - Cleaned Valid Samples : {len(cleaned_samples)}")
    print(f"  - Duplicates Removed   : {duplicates_removed}")
    print(f"  - Corrupt Images       : {corrupt_removed}")
    print(f"  - Invalid BBox Removed : {invalid_boxes_removed}")

    # 2. Subject-Level 70 / 15 / 15 Splitting
    subjects = list(set(s["subject_id"] for s in cleaned_samples))
    np.random.shuffle(subjects)

    num_subjects = len(subjects)
    n_train = int(round(num_subjects * 0.70))
    n_val = int(round(num_subjects * 0.15))

    train_subjs = set(subjects[:n_train])
    val_subjs = set(subjects[n_train:n_train + n_val])
    test_subjs = set(subjects[n_train + n_val:])

    for s in cleaned_samples:
        sid = s["subject_id"]
        if sid in train_subjs:
            s["split"] = "train"
        elif sid in val_subjs:
            s["split"] = "val"
        else:
            s["split"] = "test"

    # Verify Data Leakage
    assert len(train_subjs.intersection(val_subjs)) == 0
    assert len(train_subjs.intersection(test_subjs)) == 0
    assert len(val_subjs.intersection(test_subjs)) == 0

    print(f"\n[SUBJECT-LEVEL DATASET SPLIT]")
    print(f"  - Total Subjects       : {num_subjects}")
    print(f"  - Train Split Subjects : {len(train_subjs)}")
    print(f"  - Val Split Subjects   : {len(val_subjs)}")
    print(f"  - Test Split Subjects  : {len(test_subjs)}")

    # 3. Create Output Directory Structure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    manifest_rows = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    class_counts = {"cut": 0, "bruise": 0, "wound": 0}

    for s in cleaned_samples:
        split = s["split"]
        s_id = s["sample_id"]

        # Copy Image
        dst_img_name = f"{s_id}.jpg"
        dst_img_path = os.path.join(OUTPUT_DIR, "images", split, dst_img_name)
        shutil.copy2(s["orig_path"], dst_img_path)

        # Write Label
        dst_lbl_name = f"{s_id}.txt"
        dst_lbl_path = os.path.join(OUTPUT_DIR, "labels", split, dst_lbl_name)

        with open(dst_lbl_path, "w") as f_lbl:
            for (xc, yc, nw, nh) in s["bboxes"]:
                f_lbl.write(f"{s['class_id']} {xc} {yc} {nw} {nh}\n")

        split_counts[split] += 1
        class_counts[s["final_class"]] += 1

        manifest_rows.append({
            "sample_id": s_id,
            "subject_id": s["subject_id"],
            "split": split,
            "raw_class": s["raw_class"],
            "final_class": s["final_class"],
            "class_id": s["class_id"],
            "image_path": dst_img_path,
            "label_path": dst_lbl_path,
            "sha256": s["sha256"],
            "phash": s["phash"]
        })

    # Save Cleaned Manifest
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(os.path.join(OUTPUT_DIR, "manifest.csv"), index=False)

    # Save yolo_real.yaml
    yolo_cfg = {
        "path": OUTPUT_DIR,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    with open(os.path.join(OUTPUT_DIR, "yolo_real.yaml"), "w") as f_yaml:
        yaml.dump(yolo_cfg, f_yaml, default_flow_style=False)

    print(f"\n[FINAL CLEANED SPLIT SAMPLE COUNTS]")
    print(f"  - Train Split Images   : {split_counts['train']}")
    print(f"  - Val Split Images     : {split_counts['val']}")
    print(f"  - Test Split Images    : {split_counts['test']}")

    print(f"\n[CLASS DISTRIBUTION]")
    for c_name, count in class_counts.items():
        print(f"  - {c_name:10s} : {count} samples")

    print(f"\nSaved config to: {os.path.join(OUTPUT_DIR, 'yolo_real.yaml')}")
    print("=================================================================")
    print("REAL DATASET PREPARATION COMPLETE [OK]")
    print("=================================================================")

if __name__ == "__main__":
    prepare_dataset()
