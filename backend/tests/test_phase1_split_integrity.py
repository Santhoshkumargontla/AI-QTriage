"""Automated tests for Phase 1 split integrity, duplicate group isolation, and leak-free evaluation."""
import os
import hashlib
import numpy as np
import pytest

from ml.training.train_xgboost import generate_multimodal_dataset, get_leakage_free_splits
from ml.models.canonical_paths import XGB_CANONICAL, VQC_DIR, resolve_existing
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.classifiers.vqc_classifier import VQCClassifier


@pytest.fixture(scope="module")
def multimodal_data():
    X, y = generate_multimodal_dataset(num_samples=200, seed=42)
    train_idx, val_idx, test_idx = get_leakage_free_splits(num_samples=200)
    return X, y, train_idx, val_idx, test_idx


def test_splits_disjoint_and_complete(multimodal_data):
    X, y, train_idx, val_idx, test_idx = multimodal_data
    
    assert len(train_idx) == 140, f"Expected 140 train samples, got {len(train_idx)}"
    assert len(val_idx) == 30, f"Expected 30 val samples, got {len(val_idx)}"
    assert len(test_idx) == 30, f"Expected 30 test samples, got {len(test_idx)}"
    
    set_train = set(train_idx)
    set_val = set(val_idx)
    set_test = set(test_idx)
    
    # Strictly pairwise disjoint
    assert set_train.isdisjoint(set_val), "Train and Val splits share indices"
    assert set_train.isdisjoint(set_test), "Train and Test splits share indices"
    assert set_val.isdisjoint(set_test), "Val and Test splits share indices"
    
    # Complete coverage of 0..199
    all_indices = set_train | set_val | set_test
    assert all_indices == set(range(200)), "Splits do not cleanly partition 0..199"


def test_duplicate_feature_groups_isolated_in_train(multimodal_data):
    X, y, train_idx, val_idx, test_idx = multimodal_data
    
    # Map each feature vector hash to its row indices
    hash_to_rows = {}
    for i, row in enumerate(X):
        h = hashlib.sha256(row.tobytes()).hexdigest()
        hash_to_rows.setdefault(h, []).append(i)
        
    dup_groups = {h: idxs for h, idxs in hash_to_rows.items() if len(idxs) > 1}
    assert len(dup_groups) == 2, f"Expected 2 duplicate groups, found {len(dup_groups)}"
    
    set_train = set(train_idx)
    set_val = set(val_idx)
    set_test = set(test_idx)
    
    for h, idxs in dup_groups.items():
        # Every member of duplicate group must be in Train
        for idx in idxs:
            assert idx in set_train, f"Duplicate row {idx} is not in train split"
            assert idx not in set_val, f"Duplicate row {idx} leaked into val split"
            assert idx not in set_test, f"Duplicate row {idx} leaked into test split"


def test_zero_feature_vector_cross_split_hash_collision(multimodal_data):
    X, y, train_idx, val_idx, test_idx = multimodal_data
    
    train_hashes = set(hashlib.sha256(X[i].tobytes()).hexdigest() for i in train_idx)
    val_hashes = set(hashlib.sha256(X[i].tobytes()).hexdigest() for i in val_idx)
    test_hashes = set(hashlib.sha256(X[i].tobytes()).hexdigest() for i in test_idx)
    
    # Cross-split hash collisions must be strictly 0
    assert len(train_hashes & val_hashes) == 0, "Train and Val share identical feature vectors"
    assert len(train_hashes & test_hashes) == 0, "Train and Test share identical feature vectors"
    assert len(val_hashes & test_hashes) == 0, "Val and Test share identical feature vectors"


def test_no_held_out_test_sample_in_training(multimodal_data):
    X, y, train_idx, val_idx, test_idx = multimodal_data
    
    # Specifically check row 139 (previously boundary ambiguity)
    assert 139 in test_idx, "Row 139 should be assigned to held-out test split"
    assert 139 not in train_idx, "Row 139 must not be in training indices"
    
    # Check that rows 143 and 191 are in train, not in test
    assert 143 in train_idx and 143 not in test_idx
    assert 191 in train_idx and 191 not in test_idx


def test_retrained_models_evaluate_on_test_split(multimodal_data):
    X, y, train_idx, val_idx, test_idx = multimodal_data
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    # Test canonical XGBoost
    xgb = XGBoostClassifier(resolve_existing(XGB_CANONICAL))
    xgb_preds = [xgb.predict(row)[0] for row in X_test]
    assert len(xgb_preds) == 30
    assert all(p in (0, 1, 2) for p in xgb_preds)
    correct_xgb = sum(p == y for p, y in zip(xgb_preds, y_test))
    assert correct_xgb == 25, f"Expected 25/30 for XGBoost on clean test split, got {correct_xgb}"
    
    # Test canonical VQC
    vqc = VQCClassifier(resolve_existing(VQC_DIR))
    vqc_preds = [vqc.predict(row)[0] for row in X_test]
    assert len(vqc_preds) == 30
    assert all(p in (0, 1, 2) for p in vqc_preds)
    correct_vqc = sum(p == y for p, y in zip(vqc_preds, y_test))
    assert correct_vqc >= 15, f"Expected VQC score >= 15/30, got {correct_vqc}"
