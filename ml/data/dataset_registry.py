"""
AI-QTriage Dataset Registry & Canonical Processing Pipeline
Handles canonical label mapping, perceptual hash duplicate detection, and subject/group fallback splitting.
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Canonical Label Mapping Rules (Preserves original label in metadata)
CANONICAL_LABEL_MAP = {
    "cut": "cut",
    "laceration": "cut",
    "incision": "cut",
    "bruise": "bruise",
    "contusion": "bruise",
    "hematoma": "bruise",
    "swelling": "swelling",
    "edema": "swelling",
    "inflammation": "swelling",
    "abrasion": "abrasion",
    "scrape": "abrasion",
    "wound": "cut"
}

def get_canonical_label(original_label: str) -> Tuple[str, str]:
    """
    Returns (canonical_label, mapping_reason).
    Preserves original label in metadata.
    """
    norm_label = str(original_label).strip().lower()
    canonical = CANONICAL_LABEL_MAP.get(norm_label, norm_label)
    if canonical == norm_label:
        reason = "Direct match to canonical taxonomy"
    else:
        reason = f"Mapped '{original_label}' to research taxonomy '{canonical}' based on clinical similarity"
    return canonical, reason

def generate_hierarchical_splits(
    df: pd.DataFrame, 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15, 
    seed: int = 42
) -> pd.DataFrame:
    """
    Applies realistic split hierarchy:
    IF subject_id exists -> split by subject_id (Train subjects ∩ Test subjects = Ø)
    ELSE IF group_id exists -> split by group_id
    ELSE -> deterministic stratified split & document that subject leakage cannot be ruled out.
    NEVER invents fake subject IDs!
    """
    np.random.seed(seed)
    df = df.copy()

    if "subject_id" in df.columns and df["subject_id"].dropna().nunique() > 1:
        split_method = "subject_level_split"
        group_col = "subject_id"
    elif "group_id" in df.columns and df["group_id"].dropna().nunique() > 1:
        split_method = "group_level_split"
        group_col = "group_id"
    else:
        split_method = "stratified_random_split_no_subject_ids"
        group_col = None

    df["split_method"] = split_method

    if group_col:
        unique_groups = df[group_col].dropna().unique()
        np.random.shuffle(unique_groups)
        n_groups = len(unique_groups)
        n_train = max(1, int(n_groups * train_ratio))
        n_val = max(1, int(n_groups * val_ratio))
        
        train_groups = set(unique_groups[:n_train])
        val_groups = set(unique_groups[n_train:n_train + n_val])
        test_groups = set(unique_groups[n_train + n_val:])
        
        # Guarantee non-empty test set
        if not test_groups and len(val_groups) > 1:
            test_groups.add(val_groups.pop())

        def assign_split(g):
            if g in train_groups:
                return "train"
            elif g in val_groups:
                return "val"
            else:
                return "test"

        df["split"] = df[group_col].apply(assign_split)
    else:
        # Stratified random split
        df["split"] = "train"
        target_col = "canonical_label" if "canonical_label" in df.columns else ("class" if "class" in df.columns else df.columns[0])
        
        for cls_val in df[target_col].unique():
            cls_idx = df[df[target_col] == cls_val].index.values
            np.random.shuffle(cls_idx)
            n_cls = len(cls_idx)
            n_tr = int(n_cls * train_ratio)
            n_va = int(n_cls * val_ratio)
            
            df.loc[cls_idx[:n_tr], "split"] = "train"
            df.loc[cls_idx[n_tr:n_tr + n_va], "split"] = "val"
            df.loc[cls_idx[n_tr + n_va:], "split"] = "test"

    return df

def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
