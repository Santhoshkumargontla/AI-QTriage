"""Independent Phase 2 metrics + leakage audit. Does not copy stored JSON numbers."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
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
from ml.fusion.feature_fusion import FEATURE_NAMES, N_FEATURES
from ml.models.canonical_paths import (
    EVAL_COMPARE,
    EVAL_HELD_OUT,
    EVAL_RESULTS,
    VQC_DIR,
    VQC_PCA,
    VQC_SCALER,
    VQC_WEIGHTS,
    XGB_CANONICAL,
    abs_path,
    sha256_file,
)
from ml.training.train_xgboost import generate_multimodal_dataset


CLASS_ORDER = ["LOW", "MODERATE", "HIGH"]
N_BINS = 5


def row_hash(row) -> str:
    return hashlib.sha256(np.asarray(row, dtype=np.float64).tobytes()).hexdigest()


def ece_breakdown(probs: np.ndarray, y_true: np.ndarray, n_bins: int = N_BINS) -> dict:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == y_true)
    boundaries = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = float(boundaries[i]), float(boundaries[i + 1])
        in_bin = (confidences > lo) & (confidences <= hi)
        n = int(in_bin.sum())
        if n == 0:
            bins.append({"bin": i, "lo": lo, "hi": hi, "n": 0, "weight": 0.0, "acc": None, "conf": None, "abs_gap": None})
            continue
        acc = float(np.mean(correct[in_bin]))
        conf = float(np.mean(confidences[in_bin]))
        weight = n / len(y_true)
        gap = abs(acc - conf)
        ece += gap * weight
        bins.append({"bin": i, "lo": lo, "hi": hi, "n": n, "weight": weight, "acc": acc, "conf": conf, "abs_gap": gap})
    return {
        "formula": "sum_b (|acc_b - conf_b| * n_b / N); conf=max(p); empty bins skipped; left-open right-closed except bin0 uses >0",
        "n_bins": n_bins,
        "boundaries": boundaries.tolist(),
        "ece": float(ece),
        "confidences": confidences.tolist(),
        "predicted": predictions.tolist(),
        "correct": correct.astype(int).tolist(),
        "bins": bins,
    }


def metrics_block(name, y_true, preds, probs) -> dict:
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    probs = np.asarray(probs)
    n = int(len(y_true))
    correct = int((preds == y_true).sum())
    ratio = correct / n
    acc = float(accuracy_score(y_true, preds))
    row_sums = probs.sum(axis=1)
    return {
        "name": name,
        "n": n,
        "correct": correct,
        "correct_over_n": f"{correct} / {n}",
        "ratio_unrounded": ratio,
        "sklearn_accuracy_unrounded": acc,
        "ratio_equals_sklearn": bool(np.isclose(acc, ratio)),
        "macro_precision_unrounded": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "macro_recall_unrounded": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "macro_f1_unrounded": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "mcc_unrounded": float(matthews_corrcoef(y_true, preds)),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1, 2]).tolist(),
        "prob_shape": list(probs.shape),
        "prob_row_sum_min": float(row_sums.min()),
        "prob_row_sum_max": float(row_sums.max()),
        "brier_unrounded": _multiclass_brier(probs, y_true),
        "brier_formula": "mean_i sum_k (p_ik - y_ik)^2 with one-hot y, class order [0=LOW,1=MODERATE,2=HIGH]",
        "ece": ece_breakdown(probs, y_true),
    }


def hash_overlap(a, b):
    return sorted(a & b)


def csv_split_audit(path, split_col="split", subject_col=None, hash_col=None):
    if not os.path.isfile(path):
        return {"path": path, "exists": False}
    df = pd.read_csv(path)
    out = {
        "path": path,
        "exists": True,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "has_subject_id": "subject_id" in df.columns or bool(subject_col),
        "splits": {},
        "subject_overlap": None,
        "hash_overlap": None,
        "subject_leakage_status": None,
    }
    if split_col not in df.columns:
        out["error"] = f"missing {split_col}"
        return out
    for split, g in df.groupby(split_col):
        rec = {"samples": int(len(g))}
        subj_col = subject_col or ("subject_id" if "subject_id" in g.columns else None)
        if subj_col:
            rec["subjects"] = int(g[subj_col].nunique())
            rec["subject_ids"] = sorted(g[subj_col].astype(str).unique().tolist())
        if hash_col and hash_col in g.columns:
            rec["unique_hashes"] = int(g[hash_col].nunique())
        out["splits"][str(split)] = rec
    groups = {str(s): g for s, g in df.groupby(split_col)}
    subj_col = subject_col or ("subject_id" if "subject_id" in df.columns else None)
    if subj_col:
        sets = {k: set(v[subj_col].astype(str)) for k, v in groups.items()}
        tv = sets.get("train", set()) & sets.get("val", set())
        tt = sets.get("train", set()) & sets.get("test", set())
        vt = sets.get("val", set()) & sets.get("test", set())
        out["subject_overlap"] = {
            "train_val": len(tv),
            "train_test": len(tt),
            "val_test": len(vt),
            "train_val_ids": sorted(tv),
            "train_test_ids": sorted(tt),
            "val_test_ids": sorted(vt),
            "n_train_subjects": len(sets.get("train", set())),
            "n_val_subjects": len(sets.get("val", set())),
            "n_test_subjects": len(sets.get("test", set())),
        }
        out["subject_leakage_status"] = (
            "ZERO_SUBJECT_OVERLAP" if not (tv or tt or vt) else "SUBJECT_OVERLAP_PRESENT"
        )
        # generator IDs are not patients
        sample_ids = df[subj_col].astype(str).head(3).tolist()
        if any(str(s).startswith("subj_") for s in sample_ids):
            out["subject_id_provenance"] = "SYNTHETIC_GENERATOR_IDS_NOT_PATIENTS"
            if out["subject_leakage_status"] == "ZERO_SUBJECT_OVERLAP":
                out["clinical_subject_claim"] = "SUBJECT_LEAKAGE_NOT_VERIFIABLE"
    else:
        out["subject_leakage_status"] = "SUBJECT_LEAKAGE_NOT_VERIFIABLE"
        out["clinical_subject_claim"] = "SUBJECT_LEAKAGE_NOT_VERIFIABLE"
    if hash_col and hash_col in df.columns:
        sets_h = {k: set(v[hash_col].astype(str)) for k, v in groups.items()}
        out["hash_overlap"] = {
            "train_val": len(sets_h.get("train", set()) & sets_h.get("val", set())),
            "train_test": len(sets_h.get("train", set()) & sets_h.get("test", set())),
            "val_test": len(sets_h.get("val", set()) & sets_h.get("test", set())),
        }
    return out


def stale_inventory():
    paths = [
        "data/results/canonical_held_out_evaluation.json",
        "data/results/evaluation_results.json",
        "data/results/vqc_xgb_comparison.json",
        "ml/evaluation/metric_consistency_audit.json",
        "ml/evaluation/yolo11/results.json",
        "independent_xgboost_evaluation.json",
        "ml/models/_archive/classical/xgboost_metadata.json",
        "baseline/pre_remediation_2026-08-28/metadata/ml_models_classical_xgboost_metadata.json",
        "ml/models/xgboost_metadata.json",
        "ml/models/vqc/vqc_metadata.json",
    ]
    rows = []
    for rel in paths:
        p = abs_path(rel) if not os.path.isabs(rel) else rel
        if not os.path.isfile(p):
            p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            rows.append({"path": rel, "exists": False})
            continue
        st = os.stat(p)
        text = open(p, encoding="utf-8", errors="replace").read()
        rows.append({
            "path": rel.replace("\\", "/"),
            "exists": True,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "bytes": st.st_size,
            "contains_0_9444": "0.9444" in text,
            "contains_0_2222": "0.2222" in text,
            "contains_25_30": "25 / 30" in text or '"correct_predictions": 25' in text,
            "contains_27_30": "27" in text and "0.9" in text,
        })
    return rows


def frontend_hardcode_scan():
    page = os.path.join(ROOT, "frontend", "app", "research", "page.tsx")
    src = open(page, encoding="utf-8").read()
    hits = []
    for token in ("0.9444", "0.2222", "xgb_correct = 17", "vqc_correct = 4", "0.958333", "0.317460", "0.928571", "0.155844", "25 / 30", "16 / 30"):
        if token in src:
            hits.append(token)
    return {
        "file": "frontend/app/research/page.tsx",
        "hardcoded_metric_tokens": hits,
        "uses_unavailable_fallback": "?? \"unavailable\"" in src or "?? 'unavailable'" in src,
        "reads_api_getComparison": "getComparison" in src,
    }


def main():
    print("=== STEP 2 live held-out recompute ===")
    X, y = generate_multimodal_dataset(num_samples=200, seed=42)
    assert X.shape[1] == N_FEATURES == 23
    X_train, y_train = X[:140], y[:140]
    X_val, y_val = X[140:170], y[140:170]
    X_test, y_test = X[170:], y[170:]
    train_idx = list(range(0, 140))
    val_idx = list(range(140, 170))
    test_idx = list(range(170, 200))
    print("source generate_multimodal_dataset(num_samples=200, seed=42)")
    print("split train[:140] val[140:170] test[170:]")
    print("len y_train/val/test", len(y_train), len(y_val), len(y_test))
    print("feature names", FEATURE_NAMES)
    print("label counts", {"train": dict(Counter(y_train.tolist())), "val": dict(Counter(y_val.tolist())), "test": dict(Counter(y_test.tolist()))})

    h_train = [row_hash(r) for r in X_train]
    h_val = [row_hash(r) for r in X_val]
    h_test = [row_hash(r) for r in X_test]
    s_train, s_val, s_test = set(h_train), set(h_val), set(h_test)
    overlap = {
        "train_val_hashes": hash_overlap(s_train, s_val),
        "train_test_hashes": hash_overlap(s_train, s_test),
        "val_test_hashes": hash_overlap(s_val, s_test),
    }
    print("hash overlap train-val", len(overlap["train_val_hashes"]), "train-test", len(overlap["train_test_hashes"]), "val-test", len(overlap["val_test_hashes"]))

    def locate(hashes, target):
        return [i for i, h in enumerate(hashes) if h == target]

    overlap_detail = {}
    for key, vals in overlap.items():
        overlap_detail[key] = []
        for h in vals:
            overlap_detail[key].append({
                "hash": h,
                "train_local_idx": locate(h_train, h),
                "val_local_idx": locate(h_val, h),
                "test_local_idx": locate(h_test, h),
                "global_train": [train_idx[i] for i in locate(h_train, h)],
                "global_val": [val_idx[i] for i in locate(h_val, h)],
                "global_test": [test_idx[i] for i in locate(h_test, h)],
            })
            print(key, overlap_detail[key][-1])

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

    print("len y_test", len(y_test), "y_pred_xgb", len(xgb_preds), "y_pred_vqc", len(vqc_preds))
    print("len probs xgb", len(xgb_probs), "vqc", len(vqc_probs))
    assert len(y_test) == len(xgb_preds) == len(vqc_preds) == len(xgb_probs) == len(vqc_probs)

    xgb_m = metrics_block("XGBoost", y_test, xgb_preds, xgb_probs)
    vqc_m = metrics_block("VQC", y_test, vqc_preds, vqc_probs)
    print("XGB CM", xgb_m["confusion_matrix"], xgb_m["correct_over_n"], "acc", xgb_m["sklearn_accuracy_unrounded"])
    print("VQC CM", vqc_m["confusion_matrix"], vqc_m["correct_over_n"], "acc", vqc_m["sklearn_accuracy_unrounded"])
    print("XGB Brier", xgb_m["brier_unrounded"], "ECE", xgb_m["ece"]["ece"])
    print("VQC Brier", vqc_m["brier_unrounded"], "ECE", vqc_m["ece"]["ece"])
    print("impossible 0.9444?", bool(np.isclose(xgb_m["sklearn_accuracy_unrounded"], 0.9444)))
    print("impossible 0.2222?", bool(np.isclose(vqc_m["sklearn_accuracy_unrounded"], 0.2222)))
    print("XGB ECE bins", json.dumps(xgb_m["ece"]["bins"], indent=2))
    print("VQC ECE bins", json.dumps(vqc_m["ece"]["bins"], indent=2))

    scaler_n = getattr(vqc.scaler, "n_samples_seen_", None)
    pca_n = getattr(vqc.pca, "n_samples_", None)
    print("VQC scaler n_samples_seen_", scaler_n, "pca n_samples_", pca_n, "n_features_in_", getattr(vqc.pca, "n_features_in_", None))
    print("XGB n_features_in_", getattr(xgb.model, "n_features_in_", None), "sha", sha256_file(XGB_CANONICAL))
    print("VQC weights sha", sha256_file(VQC_WEIGHTS), "scaler", sha256_file(VQC_SCALER), "pca", sha256_file(VQC_PCA))

    train_src = inspect.getsource(VQCClassifier.train)
    predict_src = inspect.getsource(VQCClassifier.predict)
    xgb_train_src = inspect.getsource(open(os.path.join(ROOT, "ml", "training", "train_xgboost.py")).read if False else generate_multimodal_dataset)
    xgb_script = open(os.path.join(ROOT, "ml", "training", "train_xgboost.py"), encoding="utf-8").read()
    vqc_script = open(os.path.join(ROOT, "ml", "training", "train_vqc.py"), encoding="utf-8").read()

    print("=== regenerate canonical from same live preds via run_all ===")
    written = run_complete_evaluation()
    stored = json.load(open(abs_path(EVAL_HELD_OUT), encoding="utf-8"))
    mc = stored["evaluation"]["metrics_comparison"]
    print("stored xgb", mc["xgb_correct"], "live", xgb_m["correct_over_n"], "match", mc["xgb_correct"] == xgb_m["correct_over_n"])
    print("stored vqc", mc["vqc_correct"], "live", vqc_m["correct_over_n"], "match", mc["vqc_correct"] == vqc_m["correct_over_n"])

    from backend.main import _comparison_api_payload
    payload = _comparison_api_payload(stored)
    print("API payload canonical", payload.get("canonical_artifact"), payload["classical_xgb"]["xgb_correct"], payload["quantum_vqc"]["vqc_correct"])

    sensor = csv_split_audit(os.path.join(ROOT, "data", "datasets", "manifests", "sensor_manifest.csv"))
    print("SENSOR subjects", sensor.get("subject_overlap"), sensor.get("subject_leakage_status"))

    public = csv_split_audit(os.path.join(ROOT, "data", "datasets", "public_wound_dataset", "manifest.csv"))
    # File-byte hashes across production public_wound splits (subject_id is a generator label).
    pub_path = os.path.join(ROOT, "data", "datasets", "public_wound_dataset", "manifest.csv")
    if os.path.isfile(pub_path):
        pdf = pd.read_csv(pub_path)
        hash_sets = defaultdict(set)
        for _, rec in pdf.iterrows():
            img = rec.get("image_path")
            if not isinstance(img, str):
                continue
            full = img if os.path.isabs(img) else os.path.join(ROOT, "data", "datasets", "public_wound_dataset", img)
            if not os.path.isfile(full):
                full = os.path.join(ROOT, img)
            if not os.path.isfile(full):
                continue
            h = hashlib.sha256(open(full, "rb").read()).hexdigest()
            hash_sets[str(rec["split"])].add(h)
        public["file_sha256_overlap"] = {
            "train_val": len(hash_sets.get("train", set()) & hash_sets.get("val", set())),
            "train_test": len(hash_sets.get("train", set()) & hash_sets.get("test", set())),
            "val_test": len(hash_sets.get("val", set()) & hash_sets.get("test", set())),
            "hashed_files": {k: len(v) for k, v in hash_sets.items()},
        }
    print("PUBLIC_WOUND subjects", public.get("subject_overlap"), public.get("subject_leakage_status"), public.get("clinical_subject_claim"))
    print("PUBLIC_WOUND file hashes", public.get("file_sha256_overlap"))

    eff_proc = csv_split_audit(os.path.join(ROOT, "data", "datasets", "efficientnet_processed", "manifest.csv"), hash_col="pixel_sha256")
    unet_proc = csv_split_audit(os.path.join(ROOT, "data", "datasets", "unet_processed", "manifest.csv"), hash_col="pixel_sha256")
    print("EFFNET processed hash overlap", eff_proc.get("hash_overlap"), "subject", eff_proc.get("subject_leakage_status"))
    print("UNET processed hash overlap", unet_proc.get("hash_overlap"), "subject", unet_proc.get("subject_leakage_status"))

    fe = frontend_hardcode_scan()
    print("frontend hardcodes", fe)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": "ml.training.train_xgboost.generate_multimodal_dataset",
            "num_samples": 200,
            "seed": 42,
            "split": "index slices train[:140] val[140:170] test[170:]",
            "schema": FEATURE_NAMES,
            "n_features": 23,
            "class_order": CLASS_ORDER,
            "len_y_train": int(len(y_train)),
            "len_y_val": int(len(y_val)),
            "len_y_test": int(len(y_test)),
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
            "same_test_rows_xgb_and_vqc": True,
            "subjects": "none — synthetic vectors; SUBJECT_LEAKAGE_NOT_VERIFIABLE",
        },
        "xgboost": xgb_m,
        "vqc": vqc_m,
        "row_hash_overlap": overlap_detail,
        "xgb_runtime": {
            "path": XGB_CANONICAL.replace("\\", "/"),
            "sha256": sha256_file(XGB_CANONICAL),
            "n_features_in": int(getattr(xgb.model, "n_features_in_", 0) or 0),
            "predict_uses_scaler": "scaler" in inspect.getsource(xgb.predict).lower() and "StandardScaler" in inspect.getsource(type(xgb).predict),
            "train_script_fits_scaler_but_fits_model_on_unscaled_X_train": "xgb.fit(X_train, y_train" in xgb_script and "scaler.fit_transform(X_train)" in xgb_script,
            "eval_set_used": "eval_set" in xgb_script,
            "val_used_for_fit": "X_val" in xgb_script and "xgb.fit" in xgb_script,
        },
        "vqc_runtime": {
            "dir": VQC_DIR.replace("\\", "/"),
            "weights_sha256": sha256_file(VQC_WEIGHTS),
            "scaler_sha256": sha256_file(VQC_SCALER),
            "pca_sha256": sha256_file(VQC_PCA),
            "scaler_n_samples_seen_": float(scaler_n) if scaler_n is not None else None,
            "pca_n_samples_": int(pca_n) if pca_n is not None else None,
            "scaler_fit_on_train_only": bool(scaler_n == 140),
            "pca_fit_on_train_only": bool(pca_n == 140),
            "train_uses_fit_transform": "fit_transform" in train_src,
            "predict_uses_transform_only": "self.scaler.transform" in predict_src and "fit_transform" not in predict_src,
            "train_script_uses_X_val_for_optimization": "X_val" in vqc_script,
            "train_script_test_is_X_170": "X[170:]" in vqc_script,
            "circuits_match": vqc.circuit_spec().get("circuits_match"),
        },
        "sensor": sensor,
        "efficientnet_unet_production_manifest": public,
        "efficientnet_processed": eff_proc,
        "unet_processed": unet_proc,
        "stale_artifacts": stale_inventory(),
        "frontend": fe,
        "canonical_match": {
            "xgb": mc["xgb_correct"] == xgb_m["correct_over_n"],
            "vqc": mc["vqc_correct"] == vqc_m["correct_over_n"],
            "test_sample_count": stored["evaluation"]["test_sample_count"] == int(len(y_test)),
            "api_canonical_artifact": payload.get("canonical_artifact"),
        },
        "written_eval_test_sample_count": written.get("test_sample_count"),
    }
    dest = os.path.join(ROOT, "scratch", "phase2_metrics_leakage_audit.json")
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, default=str)
    print("wrote", dest)


if __name__ == "__main__":
    main()
