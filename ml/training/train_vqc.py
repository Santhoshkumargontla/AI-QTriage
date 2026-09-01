"""Train VQC on the shared QNode, evaluate held-out, compare to XGBoost. Experimental only."""
import os
import sys
import json
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.classifiers.vqc_classifier import EXPERIMENTAL_ONLY, VQCClassifier
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.models.model_registry_manager import register_model_artifact
from ml.training.train_xgboost import generate_multimodal_dataset, compute_brier_and_ece
from ml.models.canonical_paths import VQC_DIR, XGB_CANONICAL, sha256_file

METADATA_SAVE_PATH = os.path.join(VQC_DIR, "vqc_metadata.json")
COMPARE_SAVE_PATH = os.path.join("data", "results", "vqc_xgb_comparison.json")


def _metrics(y_true, preds, probs, label):
    preds = np.asarray(preds)
    y_true = np.asarray(y_true)
    prec, rec, f1_per, supp = precision_recall_fscore_support(
        y_true, preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    brier, ece = compute_brier_and_ece(np.asarray(probs), y_true)
    return {
        "model": label,
        "correct_predictions": f"{int((preds == y_true).sum())} / {len(y_true)}",
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "macro_f1": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 6),
        "mcc": round(float(matthews_corrcoef(y_true, preds)), 6),
        "macro_precision": round(float(np.mean(prec)), 6),
        "macro_recall": round(float(np.mean(rec)), 6),
        "per_class_support": [int(x) for x in supp],
        "brier_score": brier,
        "ece": ece,
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1, 2]).tolist(),
    }


def train_vqc(epochs: int = 15, lr: float = 0.1):
    os.makedirs(VQC_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(COMPARE_SAVE_PATH), exist_ok=True)

    # Same generator and split as XGBoost: 200 samples, 140/30/30, seed 42.
    X, y = generate_multimodal_dataset(num_samples=200, seed=42)
    X_train, y_train = X[:140], y[:140]
    X_test, y_test = X[170:], y[170:]

    vqc = VQCClassifier()
    t_train = time.time()
    vqc.train(X_train, y_train, epochs=epochs, lr=lr)
    train_seconds = time.time() - t_train
    vqc.save_model(VQC_DIR)

    t0 = time.time()
    vqc_preds, vqc_probs = [], []
    for row in X_test:
        idx, p = vqc.predict(row)
        vqc_preds.append(idx)
        vqc_probs.append(p)
    vqc_latency_ms = (time.time() - t0) / max(len(X_test), 1) * 1000.0

    vqc_metrics = _metrics(y_test, vqc_preds, vqc_probs, "VQC")
    vqc_metrics["inference_latency_ms_per_sample"] = round(vqc_latency_ms, 2)

    xgb_metrics = None
    xgb_loaded = os.path.exists(XGB_CANONICAL)
    if xgb_loaded:
        xgb = XGBoostClassifier(XGB_CANONICAL)
        xgb_preds, xgb_probs = [], []
        for row in X_test:
            idx, p = xgb.predict(row)
            xgb_preds.append(idx)
            xgb_probs.append(p)
        xgb_metrics = _metrics(y_test, xgb_preds, xgb_probs, "XGBoost")

    vqc_acc = vqc_metrics["accuracy"]
    xgb_acc = xgb_metrics["accuracy"] if xgb_metrics else None
    vqc_mcc = vqc_metrics["mcc"]
    xgb_mcc = xgb_metrics["mcc"] if xgb_metrics else None
    outperforms = bool(
        xgb_metrics is not None and (vqc_acc > xgb_acc or vqc_mcc > xgb_mcc)
    )
    recommendation = "KEEP_EXPERIMENTAL" if outperforms else "DISABLE_FROM_MAIN_DECISION"

    opt = vqc.optimization or {}
    metrics = {
        "train_samples": len(y_train),
        "val_samples": 30,
        "test_samples": len(y_test),
        "epochs": epochs,
        "lr": lr,
        "optimizer": opt.get("optimizer"),
        "loss_start": opt.get("loss_start"),
        "loss_end": opt.get("loss_end"),
        "loss_decreased": opt.get("loss_decreased"),
        "loss_history": [round(x, 6) for x in (vqc.loss_history or [])],
        "train_seconds": round(train_seconds, 2),
        "circuit": vqc.circuit_spec(),
        "backend": "PennyLane default.qubit (classical simulator)",
        "metrics_source": "held_out_test_via_VQCClassifier.predict",
        "data_provenance_class": "SYNTHETIC",
        "genuinely_paired_clinical_samples": 0,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        **{f"vqc_{k}": v for k, v in vqc_metrics.items() if k != "model"},
        "xgb_comparison": xgb_metrics,
        "vqc_outperforms_xgboost": outperforms,
        "recommendation": recommendation,
        "isolation": EXPERIMENTAL_ONLY,
        "used_in_main_decision": False,
    }

    metadata = {
        "model_name": "Experimental 4-Qubit VQC",
        "version": "v1.4.0",
        "status": EXPERIMENTAL_ONLY,
        "data_provenance_class": "SYNTHETIC",
        "used_in_main_decision": False,
        "recommendation": recommendation,
        "disclaimer": (
            "PennyLane default.qubit is a classical CPU simulator. No quantum advantage. "
            "EXPERIMENTAL_ONLY. Excluded from SOS, first-aid, and main case decisions."
        ),
        "metrics": metrics,
    }
    with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    comparison = {
        "held_out_test_samples": len(y_test),
        "split": "generate_multimodal_dataset(200, seed=42) train[:140] test[170:]",
        "data_provenance_class": "SYNTHETIC",
        "vqc": vqc_metrics,
        "xgboost": xgb_metrics,
        "vqc_outperforms_xgboost": outperforms,
        "recommendation": recommendation,
        "used_in_main_decision": False,
        "circuit_match": vqc.circuit_spec(),
        "optimization": opt,
        "artifact_sha256": sha256_file(os.path.join(VQC_DIR, "vqc_weights.npz")),
        "compared_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(COMPARE_SAVE_PATH, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    register_model_artifact(
        model_name="Experimental 4-Qubit VQC",
        version="v1.4.0",
        artifact_path=os.path.join(VQC_DIR, "vqc_weights.npz"),
        training_dataset="synthetic_multimodal_fusion",
        sample_count=len(y_train),
        classes=["LOW", "MODERATE", "HIGH"],
        metrics=metrics,
        training_command=r"backend\venv\Scripts\python.exe ml\training\train_vqc.py",
        notes=(
            f"DATA_PROVENANCE=SYNTHETIC. EXPERIMENTAL_ONLY. recommendation={recommendation}. "
            "Training circuit equals inference circuit. Isolated from main decision."
        ),
    )
    print(
        f"[OK] VQC experimental train. acc={vqc_acc} xgb_acc={xgb_acc} "
        f"recommendation={recommendation}"
    )
    return metadata


if __name__ == "__main__":
    train_vqc()
