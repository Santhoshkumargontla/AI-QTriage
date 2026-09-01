import os
import json
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

DATASET_DIR = os.path.join("data", "datasets", "injury_dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

# Supported Academic Datasets:
# 1. Kaggle Wound Dataset (Classification: cut, bruise, laceration, abrasion)
# 2. Roboflow Injury Detection (Detection: bruise, cut, swelling, burn)
# 3. WOUNDSEG / WACV 2023 (Segmentation: wound masks)
# 4. GMU Bruise Data Project (George Mason University)
# EXCLUDED: JujubeBruiseNet (Fruit data, not human injury data)


def generate_synthetic_research_dataset(num_subjects: int = 15):
    """
    Creates a benchmark research dataset with strict subject-level splitting (train/val/test).
    Prevents subject data leakage across splits.
    """
    os.makedirs(os.path.join(DATASET_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "masks"), exist_ok=True)

    records = []
    classes = ["cut", "bruise", "swelling"]

    # Subject-level split (70% train, 15% val, 15% test)
    subjects = [f"subj_{i:03d}" for i in range(1, num_subjects + 1)]
    train_count = int(num_subjects * 0.70)
    val_count = int(num_subjects * 0.15)
    
    train_subjs = set(subjects[:train_count])
    val_subjs = set(subjects[train_count:train_count + val_count])
    test_subjs = set(subjects[train_count + val_count:])

    sample_counter = 1
    for subj in subjects:
        if subj in train_subjs:
            split = "train"
        elif subj in val_subjs:
            split = "val"
        else:
            split = "test"

        # Generate 2 samples per subject
        for _ in range(2):
            cls = classes[(sample_counter - 1) % len(classes)]
            img_filename = f"images/sample_{sample_counter:04d}.jpg"
            mask_filename = f"masks/mask_{sample_counter:04d}.png"
            
            img_full = os.path.join(DATASET_DIR, img_filename)
            mask_full = os.path.join(DATASET_DIR, mask_filename)

            # Create synthetic image (224x224 RGB)
            img = Image.new("RGB", (224, 224), color=(180, 140, 120))
            draw = ImageDraw.Draw(img)
            
            # Create synthetic mask (224x224 grayscale)
            mask = Image.new("L", (224, 224), color=0)
            mask_draw = ImageDraw.Draw(mask)

            if cls == "cut":
                draw.line([(80, 50), (140, 170)], fill=(180, 20, 20), width=6)
                mask_draw.line([(80, 50), (140, 170)], fill=255, width=10)
            elif cls == "bruise":
                draw.ellipse([(60, 60), (160, 160)], fill=(70, 30, 90))
                mask_draw.ellipse([(60, 60), (160, 160)], fill=255)
            else:  # swelling
                draw.ellipse([(50, 50), (170, 170)], fill=(200, 160, 140), outline=(220, 180, 160), width=4)
                mask_draw.ellipse([(50, 50), (170, 170)], fill=255)

            img.save(img_full)
            mask.save(mask_full)

            records.append({
                "sample_id": f"sample_{sample_counter:04d}",
                "subject_id": subj,
                "image_path": img_filename,
                "class": cls,
                "mask_path": mask_filename,
                "source": "AI-QTriage Synthetic Research Benchmark",
                "license": "CC-BY-4.0 Academic Research",
                "split": split
            })
            sample_counter += 1

    df = pd.DataFrame(records)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Generated dataset at {DATASET_DIR} with {len(df)} samples across {num_subjects} subjects.")
    print(f"Split distribution: {df['split'].value_counts().to_dict()}")

if __name__ == "__main__":
    generate_synthetic_research_dataset()
