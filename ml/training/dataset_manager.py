import os
import json
import pandas as pd
from PIL import Image
from typing import Dict, Any, List

def validate_dataset_quality(manifest_path: str, dataset_dir: str) -> Dict[str, Any]:
    """
    Validates the dataset manifest and checks files for corruption, subject leakage,
    class imbalance, and other data quality parameters.
    
    Generates a quality report dictionary.
    """
    report: Dict[str, Any] = {
        "status": "passed",
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_records": 0,
        "class_distribution": {},
        "split_distribution": {},
        "subject_leakage_detected": False,
        "corrupted_images": [],
        "missing_images": [],
        "missing_masks": [],
        "invalid_labels": [],
        "validation_errors": []
    }
    
    if not os.path.exists(manifest_path):
        report["status"] = "failed"
        report["validation_errors"].append(f"Manifest file not found at: {manifest_path}")
        return report
        
    try:
        df = pd.read_csv(manifest_path)
    except (OSError, ValueError, pd.errors.ParserError) as e:
        report["status"] = "failed"
        report["validation_errors"].append(f"Failed to read CSV manifest: {str(e)}")
        return report
        
    report["total_records"] = len(df)
    
    # 1. Required Columns Check
    required_cols = ["sample_id", "subject_id", "image_path", "class", "mask_path", "source", "license", "split"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        report["status"] = "failed"
        report["validation_errors"].append(f"Missing required columns in manifest: {missing_cols}")
        return report

    # 2. Duplicate Check
    duplicate_samples = df[df.duplicated(subset=["sample_id"], keep=False)]["sample_id"].unique().tolist()
    if duplicate_samples:
        report["status"] = "failed"
        report["validation_errors"].append(f"Duplicate sample_ids found: {duplicate_samples}")

    # 3. Class and Split Distributions
    report["class_distribution"] = df["class"].value_counts().to_dict()
    report["split_distribution"] = df["split"].value_counts().to_dict()

    # Valid categories & splits
    valid_classes = {"cut", "bruise", "swelling"}
    valid_splits = {"train", "val", "test"}

    # 4. Record Level Validations
    train_subjects = set()
    test_subjects = set()
    val_subjects = set()

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        subj_id = str(row["subject_id"])
        img_rel_path = row["image_path"]
        cls = row["class"]
        mask_rel_path = row["mask_path"]
        split = row["split"]
        
        # Track subjects per split
        if split == "train":
            train_subjects.add(subj_id)
        elif split == "test":
            test_subjects.add(subj_id)
        elif split == "val":
            val_subjects.add(subj_id)

        # Class validation
        if cls not in valid_classes:
            report["invalid_labels"].append({
                "sample_id": sample_id,
                "label": cls,
                "reason": "Not in supported research classes (cut, bruise, swelling)"
            })
            
        # Split validation
        if split not in valid_splits:
            report["status"] = "failed"
            report["validation_errors"].append(f"Invalid split name '{split}' for sample {sample_id}")

        # Image file validation
        img_full_path = os.path.join(dataset_dir, img_rel_path)
        if not os.path.exists(img_full_path):
            report["missing_images"].append({
                "sample_id": sample_id,
                "path": img_rel_path
            })
        else:
            # Check image file format and integrity
            try:
                with Image.open(img_full_path) as img:
                    img.verify()
            except (OSError, ValueError, Image.UnidentifiedImageError) as e:
                report["corrupted_images"].append({
                    "sample_id": sample_id,
                    "path": img_rel_path,
                    "error": str(e)
                })

        # Mask file validation (optional depending on layout, but must exist if specified)
        if pd.notna(mask_rel_path) and str(mask_rel_path).strip() != "":
            mask_full_path = os.path.join(dataset_dir, mask_rel_path)
            if not os.path.exists(mask_full_path):
                report["missing_masks"].append({
                    "sample_id": sample_id,
                    "path": mask_rel_path
                })
            else:
                try:
                    with Image.open(mask_full_path) as mask:
                        mask.verify()
                except (OSError, ValueError, Image.UnidentifiedImageError) as e:
                    report["status"] = "failed"
                    report["validation_errors"].append(f"Corrupted mask file for {sample_id}: {str(e)}")

    # 5. Subject Leakage Check
    overlap_train_test = train_subjects.intersection(test_subjects)
    overlap_train_val = train_subjects.intersection(val_subjects)
    overlap_val_test = val_subjects.intersection(test_subjects)
    
    leakage_details = []
    if overlap_train_test:
        leakage_details.append(f"Train-Test overlap subjects: {list(overlap_train_test)}")
    if overlap_train_val:
        leakage_details.append(f"Train-Val overlap subjects: {list(overlap_train_val)}")
    if overlap_val_test:
        leakage_details.append(f"Val-Test overlap subjects: {list(overlap_val_test)}")
        
    if leakage_details:
        report["subject_leakage_detected"] = True
        report["status"] = "failed"
        for detail in leakage_details:
            report["validation_errors"].append(detail)

    # 6. Set Final Status based on validation findings
    if (report["corrupted_images"] or 
        report["missing_images"] or 
        report["missing_masks"] or 
        report["invalid_labels"]):
        report["status"] = "failed"

    return report

def run_dataset_quality_check(manifest_path: str, dataset_dir: str, report_output_path: str) -> bool:
    """Runs the validation and saves the report. Returns True if passed, False otherwise."""
    report = validate_dataset_quality(manifest_path, dataset_dir)
    
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Dataset quality report saved to {report_output_path}")
    print(f"Validation Status: {report['status'].upper()}")
    
    return report["status"] == "passed"
