import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import hashlib
import glob
from collections import Counter

evidence = {}

# 1. Inspect canonical paths & registry
from ml.models.canonical_paths import (
    YOLO_CANONICAL, EFFNET_CANONICAL, UNET_CANONICAL, SENSOR_MODEL, XGB_CANONICAL, VQC_WEIGHTS, REGISTRY_PATH
)

print("=== CHECKPOINT DISK & HASH AUDIT ===")
checkpoints = {
    "yolo11": YOLO_CANONICAL,
    "efficientnetv2": EFFNET_CANONICAL,
    "unet": UNET_CANONICAL,
    "sensor": SENSOR_MODEL,
    "xgboost": XGB_CANONICAL,
    "vqc": VQC_WEIGHTS
}

registry = json.load(open(REGISTRY_PATH, encoding="utf-8"))

for name, path in checkpoints.items():
    exists = os.path.exists(path)
    sz = os.path.getsize(path) if exists else 0
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest() if exists else "MISSING"
    print(f"[{name}] {path} exists={exists} size={sz:,} sha256={sha[:16]}...")
    evidence[f"{name}_checkpoint"] = {
        "path": path, "exists": exists, "size": sz, "sha256": sha
    }

# 2. Inspect YOLO dataset
print("\n=== YOLO11 DATASET AUDIT (yolo_expanded_v1) ===")
yolo_dir = "data/datasets/yolo_expanded_v1"
yolo_yaml = os.path.join(yolo_dir, "data.yaml")
yolo_exists = os.path.exists(yolo_dir)
print(f"Directory {yolo_dir} exists: {yolo_exists}")
if yolo_exists:
    train_imgs = glob.glob(os.path.join(yolo_dir, "images", "train", "*.*"))
    val_imgs = glob.glob(os.path.join(yolo_dir, "images", "val", "*.*"))
    test_imgs = glob.glob(os.path.join(yolo_dir, "images", "test", "*.*"))
    train_lbls = glob.glob(os.path.join(yolo_dir, "labels", "train", "*.*"))
    val_lbls = glob.glob(os.path.join(yolo_dir, "labels", "val", "*.*"))
    test_lbls = glob.glob(os.path.join(yolo_dir, "labels", "test", "*.*"))
    print(f"Images: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}, total={len(train_imgs)+len(val_imgs)+len(test_imgs)}")
    print(f"Labels: train={len(train_lbls)}, val={len(val_lbls)}, test={len(test_lbls)}")
    yaml_content = open(yolo_yaml, encoding="utf-8").read() if os.path.exists(yolo_yaml) else "NO_YAML"
    print("YAML config preview:")
    for line in yaml_content.strip().split("\n")[:10]:
        print("  ", line)

# 3. Inspect EfficientNet dataset
print("\n=== EFFICIENTNET DATASET AUDIT (efficientnet_kaggle_v1) ===")
eff_dir = "data/datasets/efficientnet_kaggle_v1"
eff_exists = os.path.exists(eff_dir)
print(f"Directory {eff_dir} exists: {eff_exists}")
if eff_exists:
    eff_train = glob.glob(os.path.join(eff_dir, "train", "*", "*.*"))
    eff_val = glob.glob(os.path.join(eff_dir, "val", "*", "*.*"))
    eff_test = glob.glob(os.path.join(eff_dir, "test", "*", "*.*"))
    print(f"Photos: train={len(eff_train)}, val={len(eff_val)}, test={len(eff_test)}, total={len(eff_train)+len(eff_val)+len(eff_test)}")
    classes_train = [os.path.basename(os.path.dirname(p)) for p in eff_train]
    classes_test = [os.path.basename(os.path.dirname(p)) for p in eff_test]
    print(f"Classes in train: {sorted(list(set(classes_train)))}")
    print(f"Test class distribution: {dict(Counter(classes_test))}")

# 4. Inspect U-Net dataset
print("\n=== U-NET DATASET AUDIT (unet_deduped_subject) ===")
unet_dir = "data/datasets/unet_deduped_subject"
unet_exists = os.path.exists(unet_dir)
print(f"Directory {unet_dir} exists: {unet_exists}")
if unet_exists:
    unet_train_img = glob.glob(os.path.join(unet_dir, "train", "images", "*.*"))
    unet_train_msk = glob.glob(os.path.join(unet_dir, "train", "masks", "*.*"))
    unet_val_img = glob.glob(os.path.join(unet_dir, "val", "images", "*.*"))
    unet_val_msk = glob.glob(os.path.join(unet_dir, "val", "masks", "*.*"))
    unet_test_img = glob.glob(os.path.join(unet_dir, "test", "images", "*.*"))
    unet_test_msk = glob.glob(os.path.join(unet_dir, "test", "masks", "*.*"))
    print(f"Pairs: train={len(unet_train_img)}/{len(unet_train_msk)}, val={len(unet_val_img)}/{len(unet_val_msk)}, test={len(unet_test_img)}/{len(unet_test_msk)}")
    print(f"Total test pairs: {len(unet_test_img)}")

# 5. Inspect Sensor dataset & manifest
print("\n=== SENSOR DATASET AUDIT (sensor_real_manifest.csv) ===")
sensor_manifest = "data/datasets/manifests/sensor_real_manifest.csv"
sensor_exists = os.path.exists(sensor_manifest)
print(f"Manifest {sensor_manifest} exists: {sensor_exists}")
if sensor_exists:
    import pandas as pd
    df_sensor = pd.read_csv(sensor_manifest)
    print(f"Total rows in manifest: {len(df_sensor)}")
    print("Columns:", list(df_sensor.columns))
    if "split" in df_sensor.columns:
        print("Split distribution:\n", df_sensor["split"].value_counts().to_dict())
    if "source" in df_sensor.columns:
        print("Source distribution:\n", df_sensor["source"].value_counts().to_dict())
    if "label" in df_sensor.columns:
        print("Label distribution:\n", df_sensor["label"].value_counts().to_dict())

with open("scratch/audit_evidence/initial_inspection.json", "w", encoding="utf-8") as f:
    json.dump(evidence, f, indent=2)
print("\nSaved inspection record to scratch/audit_evidence/initial_inspection.json")
