"""Phase 1: recompute XGB/VQC held-out metrics from live artifacts. No copied JSON."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from ml.classifiers.vqc_classifier import VQCClassifier
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.evaluation.run_all import _multiclass_brier, calculate_ece, run_complete_evaluation
from ml.models.canonical_paths import EVAL_COMPARE, EVAL_HELD_OUT, VQC_DIR, XGB_CANONICAL, sha256_file
from ml.training.train_xgboost import generate_multimodal_dataset


def row_hash(row) -> str:
    return hashlib.sha256(np.asarray(row, dtype=np.float64).tobytes()).hexdigest()


def main() -> None:
    X, y = generate_multimodal_dataset(num_samples=200, seed=42)
    X_train, y_train = X[:140], y[:140]
    X_val, y_val = X[140:170], y[140:170]
    X_test, y_test = X[170:], y[170:]
    print("len(y_train)", len(y_train), "len(y_val)", len(y_val), "len(y_test)", len(y_test))
    print("label counts train", Counter(y_train.tolist()), "val", Counter(y_val.tolist()), "test", Counter(y_test.tolist()))

    h_train = {row_hash(r) for r in X_train}
    h_val = {row_hash(r) for r in X_val}
    h_test = {row_hash(r) for r in X_test}
    print("exact-hash overlap train intersect val", len(h_train & h_val))
    print("exact-hash overlap train intersect test", len(h_train & h_test))
    print("exact-hash overlap val intersect test", len(h_val & h_test))

    xgb = XGBoostClassifier(XGB_CANONICAL)
    xgb_preds, xgb_probs = [], []
    for row in X_test:
        idx, p = xgb.predict(row)
        xgb_preds.append(idx)
        xgb_probs.append(p)
    xgb_preds = np.asarray(xgb_preds)
    xgb_probs = np.asarray(xgb_probs)

    vqc = VQCClassifier(VQC_DIR)
    vqc_preds, vqc_probs = [], []
    for row in X_test:
        idx, p = vqc.predict(row)
        vqc_preds.append(idx)
        vqc_probs.append(p)
    vqc_preds = np.asarray(vqc_preds)
    vqc_probs = np.asarray(vqc_probs)

    print("len(y_test)", len(y_test))
    print("len(y_pred_xgb)", len(xgb_preds))
    print("len(y_pred_vqc)", len(vqc_preds))
    assert len(y_test) == len(xgb_preds) == len(vqc_preds)

    xgb_cm = confusion_matrix(y_test, xgb_preds, labels=[0, 1, 2])
    vqc_cm = confusion_matrix(y_test, vqc_preds, labels=[0, 1, 2])
    print("XGB confusion\n", xgb_cm)
    print("VQC confusion\n", vqc_cm)
    xgb_correct = int((xgb_preds == y_test).sum())
    vqc_correct = int((vqc_preds == y_test).sum())
    print("XGB correct", xgb_correct, "/", len(y_test), "ratio", xgb_correct / len(y_test), "sklearn", accuracy_score(y_test, xgb_preds))
    print("VQC correct", vqc_correct, "/", len(y_test), "ratio", vqc_correct / len(y_test), "sklearn", accuracy_score(y_test, vqc_preds))
    print("impossible 0.9444 xgb?", np.isclose(accuracy_score(y_test, xgb_preds), 0.9444))
    print("impossible 0.2222 vqc?", np.isclose(accuracy_score(y_test, vqc_preds), 0.2222))

    for name, preds, probs in (("XGB", xgb_preds, xgb_probs), ("VQC", vqc_preds, vqc_probs)):
        print(name, "macroP", precision_score(y_test, preds, average="macro", zero_division=0))
        print(name, "macroR", recall_score(y_test, preds, average="macro", zero_division=0))
        print(name, "macroF1", f1_score(y_test, preds, average="macro", zero_division=0))
        print(name, "MCC", matthews_corrcoef(y_test, preds))
        print(name, "Brier", _multiclass_brier(probs, y_test))
        print(name, "ECE", calculate_ece(probs, y_test))

    print("scaler n_samples_seen_", getattr(vqc.scaler, "n_samples_seen_", None))
    print("pca n_samples_", getattr(vqc.pca, "n_samples_", None), "n_features_in_", getattr(vqc.pca, "n_features_in_", None))
    print("XGB n_features_in_", getattr(xgb.model, "n_features_in_", None))
    print("xgb sha", sha256_file(XGB_CANONICAL))

    print("\n=== regenerating canonical eval from these live predictions ===")
    written = run_complete_evaluation()
    stored = json.load(open(os.path.join(ROOT, EVAL_HELD_OUT), encoding="utf-8"))
    mc = stored["evaluation"]["metrics_comparison"]
    print("canonical xgb_correct", mc["xgb_correct"], "live", f"{xgb_correct} / {len(y_test)}")
    print("canonical vqc_correct", mc["vqc_correct"], "live", f"{vqc_correct} / {len(y_test)}")
    print("match xgb", mc["xgb_correct"] == f"{xgb_correct} / {len(y_test)}")
    print("match vqc", mc["vqc_correct"] == f"{vqc_correct} / {len(y_test)}")
    print("written test_sample_count", written.get("test_sample_count"))


if __name__ == "__main__":
    main()
