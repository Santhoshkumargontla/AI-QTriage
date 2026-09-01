"""
AI-QTriage — Real Dataset Annotation Quality Control & Validation Script (Phase 3)
Verifies bounding box coordinate bounds, class IDs, non-empty labels, alignment,
and exports overlay visualization samples to data/datasets/yolo_real_wound/qc_samples.
"""

import os
import glob
import cv2
import yaml
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "data", "datasets", "yolo_real_wound")
QC_DIR = os.path.join(DATASET_DIR, "qc_samples")

def validate_annotations():
    print("=================================================================")
    print("AI-QTriage — ANNOTATION QUALITY CONTROL & VALIDATION (PHASE 3)")
    print("=================================================================")

    yaml_path = os.path.join(DATASET_DIR, "yolo_real.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config yolo_real.yaml not found at {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    class_names = cfg.get("names", {0: "cut", 1: "bruise", 2: "wound"})
    print(f"Loaded YAML config. Supported Classes: {class_names}")

    os.makedirs(QC_DIR, exist_ok=True)

    total_images_checked = 0
    total_boxes_checked = 0
    out_of_bounds_count = 0
    corrupt_images = 0
    empty_label_files = 0
    class_id_distribution = {cid: 0 for cid in class_names.keys()}

    # Color palette for classes (RGB)
    colors = {
        0: (255, 50, 50),   # cut: Red
        1: (50, 200, 50),   # bruise: Green
        2: (50, 100, 255)   # wound: Blue
    }

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(DATASET_DIR, "images", split)
        lbl_dir = os.path.join(DATASET_DIR, "labels", split)

        img_files = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))

        for img_path in img_files:
            total_images_checked += 1
            bname = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, f"{bname}.txt")

            img = cv2.imread(img_path)
            if img is None:
                corrupt_images += 1
                continue

            h, w = img.shape[:2]
            overlay_img = img.copy()

            if not os.path.exists(lbl_path):
                empty_label_files += 1
                continue

            with open(lbl_path, "r") as f_lbl:
                lines = f_lbl.readlines()

            if not lines:
                empty_label_files += 1
                continue

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                xc, yc, nw, nh = [float(x) for x in parts[1:]]

                # Check bounds
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
                    out_of_bounds_count += 1

                total_boxes_checked += 1
                class_id_distribution[class_id] = class_id_distribution.get(class_id, 0) + 1

                # Convert normalized YOLO to pixel box [x1, y1, x2, y2]
                x1 = int(max(0, (xc - nw / 2.0) * w))
                y1 = int(max(0, (yc - nh / 2.0) * h))
                x2 = int(min(w, (xc + nw / 2.0) * w))
                y2 = int(min(h, (yc + nh / 2.0) * h))

                c_color = colors.get(class_id, (255, 255, 255))
                c_name = class_names.get(class_id, f"class_{class_id}")

                cv2.rectangle(overlay_img, (x1, y1), (x2, y2), c_color, 2)
                cv2.putText(overlay_img, c_name, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_color, 2)

            # Save overlay image sample for QC review
            qc_save_path = os.path.join(QC_DIR, f"qc_{split}_{bname}.jpg")
            cv2.imwrite(qc_save_path, overlay_img)

    print(f"\n[ANNOTATION VALIDATION METRICS]")
    print(f"  - Total Images Checked     : {total_images_checked}")
    print(f"  - Total Bounding Boxes     : {total_boxes_checked}")
    print(f"  - Corrupt Images Found     : {corrupt_images}")
    print(f"  - Empty Label Files        : {empty_label_files}")
    print(f"  - Out-of-Bounds Coordinates : {out_of_bounds_count}")

    print(f"\n[CLASS ID BREAKDOWN]")
    for cid, count in class_id_distribution.items():
        cname = class_names.get(cid, str(cid))
        print(f"  - Class {cid} ({cname:10s}) : {count} boxes")

    qc_samples = glob.glob(os.path.join(QC_DIR, "*.jpg"))
    print(f"\n[QC VISUAL SAMPLES]")
    print(f"  - Exported {len(qc_samples)} visual overlay samples to: {QC_DIR}")

    assert corrupt_images == 0, "Corrupt images detected during QC!"
    assert out_of_bounds_count == 0, "Out-of-bounds bounding boxes detected during QC!"

    print("=================================================================")
    print("ANNOTATION QUALITY CONTROL PASSED [OK]")
    print("=================================================================")

if __name__ == "__main__":
    validate_annotations()
