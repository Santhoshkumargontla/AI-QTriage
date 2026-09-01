"""Evaluate saved XGBoost vs VQC on the shared held-out split. Never auto-trains.

Always recomputes from live artifacts + generate_multimodal_dataset(200, seed=42)
split train[:140] / val[140:170] / test[170:]. Does not copy stale JSON metrics.
"""
import os
import sys
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.models.canonical_paths import abs_path, exists, EVAL_COMPARE, EVAL_HELD_OUT, EVAL_RESULTS

RESULTS_DIR = abs_path(os.path.join("data", "results"))
EVALUATION_JSON_PATH = abs_path(EVAL_RESULTS)
COMPARE_PATH = abs_path(EVAL_COMPARE)
CANONICAL_LIVE_PATH = abs_path(EVAL_HELD_OUT)


def calculate_ece(y_probs: np.ndarray, y_true: np.ndarray, n_bins: int = 5) -> float:
    if len(y_probs) == 0 or len(y_true) == 0:
        return 0.0
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = predictions == y_true
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0:
            ece += abs(float(np.mean(accuracies[in_bin])) - float(np.mean(confidences[in_bin]))) * prop_in_bin
    return float(ece)


def _multiclass_brier(probs, y_true, n_classes=3):
    y = np.asarray(y_true, dtype=int)
    onehot = np.zeros((len(y), n_classes), dtype=float)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((np.asarray(probs) - onehot) ** 2, axis=1)))


def _block(name, y_true, preds, probs):
    preds = np.asarray(preds)
    y_true = np.asarray(y_true)
    correct = int((preds == y_true).sum())
    n = int(len(y_true))
    acc = float(accuracy_score(y_true, preds))
    ratio = correct / n if n else 0.0
    if not np.isclose(acc, ratio):
        raise RuntimeError(f"{name}: accuracy_score {acc} != correct/n {ratio}")
    return {
        "model": name,
        "correct_predictions": f"{correct} / {n}",
        "correct_count": correct,
        "test_sample_count": n,
        "accuracy": round(acc, 6),
        "macro_precision": round(float(precision_score(y_true, preds, average="macro", zero_division=0)), 6),
        "macro_recall": round(float(recall_score(y_true, preds, average="macro", zero_division=0)), 6),
        "macro_f1": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 6),
        "mcc": round(float(matthews_corrcoef(y_true, preds)), 6),
        "brier_score": round(_multiclass_brier(probs, y_true), 6),
        "ece": round(calculate_ece(np.asarray(probs), y_true), 6),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1, 2]).tolist(),
    }


def run_complete_evaluation():
    """Score existing artifacts on the XGBoost held-out split. Does not train or fabricate metrics."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from ml.classifiers.xgboost_classifier import XGBoostClassifier
    from ml.classifiers.vqc_classifier import MODEL_UNAVAILABLE, VQCClassifier
    from ml.models.canonical_paths import VQC_DIR, VQC_WEIGHTS, XGB_CANONICAL, exists, posix, sha256_file
    from ml.training.train_xgboost import generate_multimodal_dataset

    if not exists(XGB_CANONICAL):
        return {"status": "not_evaluated", "message": "XGBoost canonical artifact missing."}

    X, y = generate_multimodal_dataset(num_samples=200, seed=42)
    X_train, y_train = X[:140], y[:140]
    X_val, y_val = X[140:170], y[140:170]
    X_test, y_test = X[170:], y[170:]
    if len(y_test) != 30:
        print(f"Notice: held-out length is {len(y_test)}, not a hardcoded 30.")

    xgb = XGBoostClassifier(XGB_CANONICAL)
    xgb_preds, xgb_probs = [], []
    for row in X_test:
        idx, p = xgb.predict(row)
        xgb_preds.append(idx)
        xgb_probs.append(p)
    xgb_block = _block("XGBoost", y_test, xgb_preds, xgb_probs)

    vqc_block = {"status": MODEL_UNAVAILABLE}
    try:
        vqc = VQCClassifier(VQC_DIR)
        vqc_preds, vqc_probs = [], []
        for row in X_test:
            idx, p = vqc.predict(row)
            vqc_preds.append(idx)
            vqc_probs.append(p)
        vqc_block = _block("VQC", y_test, vqc_preds, vqc_probs)
        vqc_block["status"] = "EXPERIMENTAL_ONLY"
    except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
        vqc_block = {"status": MODEL_UNAVAILABLE, "error": str(exc)[:300], "accuracy": None}

    n = int(len(y_test))
    compared_at = datetime.now(timezone.utc).isoformat()

    def _row_hash(row):
        import hashlib
        return hashlib.sha256(np.asarray(row, dtype=np.float64).tobytes()).hexdigest()

    h_train = {_row_hash(r) for r in X_train}
    h_val = {_row_hash(r) for r in X_val}
    h_test = {_row_hash(r) for r in X_test}
    comparison = {
        "held_out_test_samples": n,
        "split": "generate_multimodal_dataset(200, seed=42) train[:140] val[140:170] test[170:]",
        "class_order": ["LOW", "MODERATE", "HIGH"],
        "class_indices": [0, 1, 2],
        "data_provenance_class": "SYNTHETIC",
        "leakage_audit": {
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "test_samples": n,
            "val_excluded_from_test": True,
            "same_rows_for_xgb_and_vqc": True,
            "vqc_scaler_pca_fit_on": "train[:140] only (saved in vqc/scaler.pkl, vqc/pca.pkl)",
            "xgboost_inference_uses_unscaled_23d": True,
            "xgboost_train_script_scaler_pca": "fitted on train only; not used by XGBoostClassifier.predict",
            "subjects": "synthetic vectors have no subject_id; SUBJECT_LEAKAGE_NOT_VERIFIABLE",
            "exact_row_hash_overlap": {
                "train_val": int(len(h_train & h_val)),
                "train_test": int(len(h_train & h_test)),
                "val_test": int(len(h_val & h_test)),
            },
        },
        "vqc": vqc_block,
        "xgboost": xgb_block,
        "vqc_outperforms_xgboost": bool(
            vqc_block.get("accuracy") is not None
            and (
                vqc_block["accuracy"] > xgb_block["accuracy"]
                or (vqc_block.get("mcc") or 0) > xgb_block["mcc"]
            )
        ),
        "recommendation": "DISABLE_FROM_MAIN_DECISION",
        "used_in_main_decision": False,
        "compared_at": compared_at,
        "metrics_source": "live_held_out_predictions",
        "artifacts": {
            "xgboost_path": posix(XGB_CANONICAL),
            "xgboost_sha256": sha256_file(XGB_CANONICAL),
            "vqc_path": posix(VQC_WEIGHTS),
            "vqc_sha256": sha256_file(VQC_WEIGHTS) if exists(VQC_WEIGHTS) else None,
        },
    }

    evaluation_data = {
        "status": "evaluated",
        "timestamp": compared_at,
        "test_sample_count": n,
        "disclaimer": "SYNTHETIC held-out split. VQC is EXPERIMENTAL_ONLY.",
        "quantum_simulator_notice": "PennyLane default.qubit is a classical simulator. No quantum advantage claimed.",
        "metrics_comparison": {
            "xgb_correct": xgb_block["correct_predictions"],
            "vqc_correct": vqc_block.get("correct_predictions"),
            "test_sample_count": n,
            "xgb_accuracy": xgb_block["accuracy"],
            "vqc_accuracy": vqc_block.get("accuracy"),
            "xgb_precision": xgb_block["macro_precision"],
            "vqc_precision": vqc_block.get("macro_precision"),
            "xgb_recall": xgb_block["macro_recall"],
            "vqc_recall": vqc_block.get("macro_recall"),
            "xgb_macro_f1": xgb_block["macro_f1"],
            "vqc_macro_f1": vqc_block.get("macro_f1"),
            "xgb_mcc": xgb_block["mcc"],
            "vqc_mcc": vqc_block.get("mcc"),
            "xgb_brier": xgb_block["brier_score"],
            "vqc_brier": vqc_block.get("brier_score"),
            "xgb_ece": xgb_block["ece"],
            "vqc_ece": vqc_block.get("ece"),
            "xgb_confusion_matrix": xgb_block["confusion_matrix"],
            "vqc_confusion_matrix": vqc_block.get("confusion_matrix"),
        },
        "recommendation": comparison["recommendation"],
        "used_in_main_decision": False,
        "vqc_outperforms_xgboost": comparison["vqc_outperforms_xgboost"],
        "ablation_study": [],
        "metrics_source": "live_held_out_predictions",
        "class_order": ["LOW", "MODERATE", "HIGH"],
        "class_indices": [0, 1, 2],
        "leakage_audit": comparison["leakage_audit"],
        "artifacts": comparison["artifacts"],
    }

    with open(COMPARE_PATH, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    with open(EVALUATION_JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(evaluation_data, handle, indent=2)
    with open(CANONICAL_LIVE_PATH, "w", encoding="utf-8") as handle:
        json.dump({"evaluation": evaluation_data, "comparison": comparison}, handle, indent=2)
    print(f"[OK] Wrote {COMPARE_PATH}, {EVALUATION_JSON_PATH}, {CANONICAL_LIVE_PATH}")
    print(f"XGB {xgb_block['correct_predictions']} acc={xgb_block['accuracy']}")
    print(f"VQC {vqc_block.get('correct_predictions')} acc={vqc_block.get('accuracy')}")
    return evaluation_data


if __name__ == "__main__":
    run_complete_evaluation()
