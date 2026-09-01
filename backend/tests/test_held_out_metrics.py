"""Live held-out metrics must match sklearn and the canonical artifact. No hardcoded scores."""
import os
import inspect

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from ml.classifiers.vqc_classifier import VQCClassifier
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.evaluation.run_all import _multiclass_brier
from ml.fusion.feature_fusion import N_FEATURES
from ml.models.canonical_paths import EVAL_HELD_OUT, VQC_DIR, XGB_CANONICAL, abs_path
from ml.training.train_xgboost import generate_multimodal_dataset


def test_live_held_out_metrics_match_sklearn_and_canonical_artifact():
    X, y = generate_multimodal_dataset(num_samples=200, seed=42)
    assert X.shape[1] == N_FEATURES
    y_train, y_val, y_test = y[:140], y[140:170], y[170:]
    X_test = X[170:]
    assert len(y_train) == 140 and len(y_val) == 30 and len(y_test) == 30
    assert np.array_equal(X_test, X[170:])

    xgb = XGBoostClassifier(XGB_CANONICAL)
    vqc = VQCClassifier(VQC_DIR)
    xgb_preds, xgb_probs, vqc_preds, vqc_probs = [], [], [], []
    for row in X_test:
        xi, xp = xgb.predict(row)
        vi, vp = vqc.predict(row)
        xgb_preds.append(xi)
        xgb_probs.append(xp)
        vqc_preds.append(vi)
        vqc_probs.append(vp)
    xgb_preds = np.asarray(xgb_preds)
    vqc_preds = np.asarray(vqc_preds)
    xgb_probs = np.asarray(xgb_probs)
    vqc_probs = np.asarray(vqc_probs)

    assert len(y_test) == len(xgb_preds) == len(vqc_preds) == len(xgb_probs) == len(vqc_probs)
    assert xgb_probs.shape == (len(y_test), 3)
    assert vqc_probs.shape == (len(y_test), 3)
    assert np.allclose(xgb_probs.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(vqc_probs.sum(axis=1), 1.0, atol=1e-4)

    xgb_correct = int((xgb_preds == y_test).sum())
    vqc_correct = int((vqc_preds == y_test).sum())
    n = int(len(y_test))
    assert np.isclose(accuracy_score(y_test, xgb_preds), xgb_correct / n)
    assert np.isclose(accuracy_score(y_test, vqc_preds), vqc_correct / n)

    assert getattr(vqc.scaler, "n_samples_seen_", None) == 140
    assert getattr(vqc.pca, "n_samples_", None) == 140
    assert int(xgb.model.n_features_in_) == 23

    import json
    stored = json.load(open(abs_path(EVAL_HELD_OUT), encoding="utf-8"))
    mc = stored["evaluation"]["metrics_comparison"]
    assert stored["evaluation"]["test_sample_count"] == n
    assert mc["xgb_correct"] == f"{xgb_correct} / {n}"
    assert mc["vqc_correct"] == f"{vqc_correct} / {n}"
    assert stored["evaluation"]["metrics_source"] == "live_held_out_predictions"
    _multiclass_brier(xgb_probs, y_test)
    _multiclass_brier(vqc_probs, y_test)


def test_vqc_predict_transform_only_and_frontend_has_no_stale_metrics():
    predict_src = inspect.getsource(VQCClassifier.predict)
    train_src = inspect.getsource(VQCClassifier.train)
    assert "self.scaler.transform" in predict_src
    assert "fit_transform" not in predict_src
    assert "self.scaler.fit_transform" in train_src
    assert "self.pca.fit_transform" in train_src

    page = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "app", "research", "page.tsx")
    text = open(page, encoding="utf-8").read()
    for token in ("0.9444", "0.2222", "xgb_correct = 17", "vqc_correct = 4", "0.958333", "0.317460"):
        assert token not in text
    assert "unavailable" in text
    assert "getComparison" in text
