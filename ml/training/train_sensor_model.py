"""
AI-QTriage Dedicated Sensor Motion / Fall / Impact Model Training Pipeline
Trains on REAL SisFall + UCI HAR features from prepare_sensor_real().
Archives prior synthetic artifacts under ml/models/_archive/ before promoting.
"""

import os
import sys
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.models.model_registry_manager import register_model_artifact, get_file_hash
from ml.data.prepare_sensor_real import prepare_sensor_real, MANIFEST_PATH, FEATURE_COLS
from ml.models.canonical_paths import SENSOR_MODEL, SENSOR_SCALER, SENSOR_METADATA

MODEL_DIR = os.path.join("ml", "models")
ARCHIVE_DIR = os.path.join(MODEL_DIR, "_archive")
SENSOR_MODEL_PATH = SENSOR_MODEL
SENSOR_SCALER_PATH = SENSOR_SCALER
METADATA_SAVE_PATH = SENSOR_METADATA
VERSION = "v2.0.0-real"


def _archive_synthetic_artifacts():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = []
    for path in (SENSOR_MODEL_PATH, SENSOR_SCALER_PATH, METADATA_SAVE_PATH):
        if os.path.exists(path):
            base = os.path.basename(path)
            dest = os.path.join(ARCHIVE_DIR, f"synthetic_sensor_v1.2.0_{stamp}_{base}")
            shutil.copy2(path, dest)
            archived.append(dest.replace("\\", "/"))
    return archived


def train_sensor_model(force_prepare: bool = False):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if force_prepare or not os.path.exists(MANIFEST_PATH):
        df_manifest = prepare_sensor_real()
    else:
        df_manifest = pd.read_csv(MANIFEST_PATH)

    archived = _archive_synthetic_artifacts()

    label_map = {"normal_activity": 0, "fall": 1, "impact": 2}
    df_manifest = df_manifest.copy()
    df_manifest["target"] = df_manifest["canonical_label"].map(label_map)
    df_manifest = df_manifest.dropna(subset=["target"])
    df_manifest["target"] = df_manifest["target"].astype(int)

    train_df = df_manifest[df_manifest["split"] == "train"]
    val_df = df_manifest[df_manifest["split"] == "val"]
    test_df = df_manifest[df_manifest["split"] == "test"]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["target"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["target"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SENSOR_SCALER_PATH)

    counts = np.bincount(y_train, minlength=3).astype(float)
    counts[counts == 0] = 1.0
    class_w = counts.sum() / (len(counts) * counts)
    sample_w = class_w[y_train]

    clf = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
        objective="multi:softprob",
        num_class=3,
    )
    clf.fit(X_train_scaled, y_train, sample_weight=sample_w)
    clf.save_model(SENSOR_MODEL_PATH)

    preds = clf.predict(X_test_scaled)
    correct_count = int((preds == y_test).sum())
    total_test = len(y_test)
    acc = float(correct_count / total_test) if total_test else 0.0
    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0)) if total_test else 0.0
    mcc = float(matthews_corrcoef(y_test, preds)) if total_test and len(np.unique(y_test)) > 1 else 0.0
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2]).tolist()

    y_test_bin = (y_test > 0).astype(int)
    preds_bin = (preds > 0).astype(int)
    if total_test and len(np.unique(y_test_bin)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_test_bin, preds_bin, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    else:
        fpr, fnr = 0.0, 0.0

    metrics = {
        "dataset_name": "sisfall_uci_har_real",
        "sampling_rate_hz": 50.0,
        "window_seconds": 2.5,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(val_df)),
        "test_samples": int(total_test),
        "correct_predictions": f"{correct_count} / {total_test}",
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "mcc": round(mcc, 6),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "confusion_matrix": cm,
        "label_counts_train": {
            "normal_activity": int((y_train == 0).sum()),
            "fall": int((y_train == 1).sum()),
            "impact": int((y_train == 2).sum()),
        },
        "sources": df_manifest["source"].value_counts().to_dict() if "source" in df_manifest.columns else {},
    }

    metadata = {
        "model_name": "Sensor Motion Event Classifier",
        "version": VERSION,
        "status": "TRAINED",
        "domain": "Motion / Fall / Impact Event Detection",
        "classes": ["normal_activity", "fall", "impact"],
        "metrics": metrics,
        "weights_path": SENSOR_MODEL_PATH.replace("\\", "/"),
        "canonical_path": SENSOR_MODEL_PATH.replace("\\", "/"),
        "artifact_sha256": get_file_hash(SENSOR_MODEL_PATH),
        "scaler_path": SENSOR_SCALER_PATH.replace("\\", "/"),
        "scaler_sha256": get_file_hash(SENSOR_SCALER_PATH),
        "data_provenance_class": "REAL",
        "dataset_provenance": (
            "SisFall (HF mirror Algo-rythmic/Sisfall_Dataset) + UCI HAR smartphones. "
            "F*→fall, D18/D19→impact, other D* + UCI→normal_activity. "
            "Subject-level split. Not synthetic np.random."
        ),
        "training_status": "TRAINED",
        "training_was_real": True,
        "archived_synthetic_artifacts": archived,
        "known_limitations": (
            "Wearable IMU placement differs from phone-in-pocket uploads. "
            "impact class uses SisFall stumble/jump ADLs (minority). "
            "Demo football_fall.csv remains simulated. Not a medical fall-detection device."
        ),
    }

    with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    register_model_artifact(
        model_name="Sensor Motion Event Classifier",
        version=VERSION,
        artifact_path=SENSOR_MODEL_PATH,
        training_dataset="sisfall_uci_har_real",
        sample_count=len(df_manifest),
        classes=["normal_activity", "fall", "impact"],
        metrics=metrics,
        training_command="backend\\venv\\Scripts\\python.exe ml\\training\\train_sensor_model.py",
        random_seed=42,
        notes=(
            "REAL provenance. SisFall+UCI HAR. Archived synthetic v1.2.0 under ml/models/_archive/. "
            f"Test {correct_count}/{total_test}."
        ),
    )

    registry_path = os.path.join(MODEL_DIR, "model_registry.json")
    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)
    entry = registry.get("Sensor Motion Event Classifier", {})
    entry.update(
        {
            "canonical_path": SENSOR_MODEL_PATH.replace("\\", "/"),
            "status": "TRAINED",
            "training_status": "TRAINED",
            "readiness_status": "TRAINED",
            "evaluation_artifact": METADATA_SAVE_PATH.replace("\\", "/"),
            "dataset_type": "real_public_motion",
            "dataset_provenance": metadata["dataset_provenance"],
            "data_provenance_class": "REAL",
            "evaluation_status": "evaluated",
            "last_evaluated": datetime.now(timezone.utc).isoformat(),
            "known_limitations": metadata["known_limitations"],
            "training_was_real": True,
            "scaler_path": SENSOR_SCALER_PATH.replace("\\", "/"),
            "scaler_sha256": get_file_hash(SENSOR_SCALER_PATH),
            "file_size": os.path.getsize(SENSOR_MODEL_PATH) if os.path.exists(SENSOR_MODEL_PATH) else 0,
            "artifact_size_mb": round(
                os.path.getsize(SENSOR_MODEL_PATH) / (1024 * 1024), 2
            )
            if os.path.exists(SENSOR_MODEL_PATH)
            else 0.0,
            "schema_version": VERSION,
        }
    )
    registry["Sensor Motion Event Classifier"] = entry
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(
        f"[OK] Retrained Sensor Motion Model ({VERSION}) on REAL data. "
        f"Test Accuracy: {correct_count}/{total_test} ({acc*100:.2f}%)"
    )
    return metadata


if __name__ == "__main__":
    train_sensor_model(force_prepare=True)
