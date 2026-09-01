"""Strict provenance and split-leakage audit. Computes hashes from disk. Does not invent REAL."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.models.canonical_paths import (
    EFFNET_METADATA,
    SENSOR_METADATA,
    UNET_METADATA,
    VQC_DIR,
    YOLO_METADATA,
    XGB_METADATA,
    read_json,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_JSON = os.path.join(ROOT, "data", "datasets", "CANONICAL_DATASET_MANIFEST.json")
OUT_CSV = os.path.join(ROOT, "data", "datasets", "canonical_dataset_records.csv")
AHASH_NEAR = 5


def _rel(path: str) -> str:
    return path.replace("\\", "/")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pixel_sha256(path: str) -> str | None:
    try:
        img = Image.open(path).convert("RGB")
    except (OSError, Image.UnidentifiedImageError):
        return None
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    return hashlib.sha256(arr.tobytes() + f"|{w}x{h}".encode()).hexdigest()


def ahash64(path: str) -> int | None:
    try:
        img = Image.open(path).convert("L").resize((8, 8))
    except (OSError, Image.UnidentifiedImageError):
        return None
    pixels = list(img.getdata())
    avg = sum(pixels) / max(len(pixels), 1)
    bits = 0
    for i, p in enumerate(pixels):
        if p >= avg:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def overlap(a: set, b: set) -> list:
    return sorted(a & b)


def split_sets(rows: list[dict], key: str) -> dict[str, set]:
    out = defaultdict(set)
    for row in rows:
        val = row.get(key)
        if val is None or val == "":
            continue
        out[row["split"]].add(val)
    return dict(out)


def cross_split(sets: dict[str, set]) -> dict:
    return {
        "train_val": overlap(sets.get("train", set()), sets.get("val", set())),
        "train_test": overlap(sets.get("train", set()), sets.get("test", set())),
        "val_test": overlap(sets.get("val", set()), sets.get("test", set())),
    }


def n_overlap(block: dict) -> dict:
    return {k: len(v) for k, v in block.items()}


def near_dup_cross_split(rows: list[dict], max_hamming: int = AHASH_NEAR) -> dict:
    by_split = defaultdict(list)
    for row in rows:
        ah = row.get("ahash")
        if ah is None:
            continue
        by_split[row["split"]].append(row)
    pairs = []
    seen = set()
    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for a, b in split_pairs:
        for ra in by_split.get(a, []):
            for rb in by_split.get(b, []):
                d = hamming(int(ra["ahash"]), int(rb["ahash"]))
                if d <= max_hamming:
                    key = tuple(sorted((ra["sample_id"], rb["sample_id"])))
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append({
                        "splits": f"{a}/{b}",
                        "hamming": d,
                        "a": ra["sample_id"],
                        "b": rb["sample_id"],
                    })
    return {
        "threshold_bits": max_hamming,
        "pair_count": len(pairs),
        "pairs_sample": pairs[:40],
    }


def subject_status(rows: list[dict], subject_kind: str | None) -> dict:
    has_field = any(r.get("subject_id") not in (None, "") for r in rows)
    if not has_field:
        return {
            "subject_id_field_present": False,
            "subject_id_kind": None,
            "unique_subject_ids": 0,
            "subjects_by_split": {},
            "subject_overlap": {"train_val": [], "train_test": [], "val_test": []},
            "subject_overlap_n": {"train_val": 0, "train_test": 0, "val_test": 0},
            "subject_leakage_status": "SUBJECT_LEAKAGE_NOT_VERIFIABLE",
            "note": "No subject_id field. Do not claim zero subject leakage.",
        }
    subj = split_sets(rows, "subject_id")
    ov = cross_split(subj)
    # Generator labels can be disjoint without proving real-patient independence.
    if subject_kind in {"SYNTHETIC_GENERATOR_LABEL", "SYNTHETIC"}:
        leak = "SUBJECT_LEAKAGE_NOT_VERIFIABLE"
        note = (
            "subject_id values exist but are synthetic generator labels, not patient IDs. "
            f"Generator-id overlap counts: {n_overlap(ov)}. Real-patient leakage cannot be verified."
        )
    else:
        leak = "SUBJECT_LEAKAGE_DETECTED" if any(n_overlap(ov).values()) else "NO_GENERATOR_SUBJECT_ID_OVERLAP"
        note = "Subject IDs present; see overlap counts."
        if leak == "NO_GENERATOR_SUBJECT_ID_OVERLAP":
            leak = "SUBJECT_LEAKAGE_NOT_VERIFIABLE"
            note = "IDs are present but not independently verified as real patients."
    return {
        "subject_id_field_present": True,
        "subject_id_kind": subject_kind,
        "unique_subject_ids": len({r["subject_id"] for r in rows if r.get("subject_id")}),
        "subjects_by_split": {k: len(v) for k, v in subj.items()},
        "subject_overlap": {k: v[:20] for k, v in ov.items()},
        "subject_overlap_n": n_overlap(ov),
        "subject_leakage_status": leak,
        "note": note,
    }


def hash_leakage(rows: list[dict], hash_key: str) -> dict:
    sets = split_sets(rows, hash_key)
    ov = cross_split(sets)
    groups = defaultdict(list)
    for row in rows:
        h = row.get(hash_key)
        if not h:
            continue
        groups[h].append({"split": row["split"], "sample_id": row["sample_id"]})
    within = sum(1 for g in groups.values() if len(g) > 1)
    cross = []
    for digest, members in groups.items():
        splits = {m["split"] for m in members}
        if len(splits) > 1:
            cross.append({"hash": digest, "n": len(members), "splits": sorted(splits), "ids": [m["sample_id"] for m in members[:8]]})
    return {
        "unique_hashes": len(groups),
        "duplicate_hash_groups": within,
        "cross_split_hash_groups": len(cross),
        "cross_split_overlap_n": n_overlap(ov),
        "cross_split_groups_sample": cross[:15],
        "leakage": any(n_overlap(ov).values()),
    }


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def image_rows_from_manifest(manifest_path: str, image_col: str, split_col: str = "split",
                             sample_col: str = "sample_id", subject_col: str = "subject_id",
                             source_col: str = None, root: str = None) -> list[dict]:
    rows = []
    base = root or os.path.dirname(manifest_path)
    for rec in load_csv(manifest_path):
        rel = rec[image_col].replace("\\", "/")
        path = rel if os.path.isabs(rel) else os.path.join(ROOT, rel) if os.path.exists(os.path.join(ROOT, rel)) else os.path.join(base, rel.replace("/", os.sep))
        if not os.path.isfile(path):
            path = os.path.join(ROOT, rel.replace("/", os.sep))
        row = {
            "sample_id": rec.get(sample_col) or os.path.basename(rel),
            "split": rec.get(split_col) or "",
            "subject_id": rec.get(subject_col) or "",
            "source_id": rec.get("source_path") or rec.get("source_stem") or rec.get(source_col) or rec.get("source_dataset") or rec.get("source") or "",
            "path": _rel(os.path.relpath(path, ROOT)) if os.path.isfile(path) else rel,
            "abs_path": path,
        }
        if os.path.isfile(path):
            row["file_sha256"] = sha256_file(path)
            row["pixel_sha256"] = pixel_sha256(path)
            row["ahash"] = ahash64(path)
        else:
            row["file_sha256"] = rec.get("byte_sha256") or rec.get("sha256") or ""
            row["pixel_sha256"] = rec.get("pixel_sha256") or ""
            row["ahash"] = None
            row["missing_file"] = True
        rows.append(row)
    return rows


def summarize_image_dataset(name: str, rows: list[dict], provenance: dict, subject_kind: str | None) -> dict:
    split_counts = dict(Counter(r["split"] for r in rows))
    file_l = hash_leakage(rows, "file_sha256")
    pix_l = hash_leakage(rows, "pixel_sha256")
    src_sets = split_sets(rows, "source_id")
    src_ov = cross_split(src_sets) if any(r.get("source_id") for r in rows) else None
    subj = subject_status(rows, subject_kind)
    near = near_dup_cross_split(rows)
    missing = sum(1 for r in rows if r.get("missing_file"))
    leak_status = []
    if file_l["leakage"] or pix_l["leakage"]:
        leak_status.append("EXACT_HASH_CROSS_SPLIT")
    if near["pair_count"] > 0:
        leak_status.append("NEAR_DUPLICATE_CROSS_SPLIT")
    if src_ov and any(n_overlap(src_ov).values()):
        leak_status.append("SOURCE_ID_CROSS_SPLIT")
    leak_status.append(subj["subject_leakage_status"])
    if not (file_l["leakage"] or pix_l["leakage"]) and near["pair_count"] == 0:
        exact = "NO_EXACT_HASH_CROSS_SPLIT"
    else:
        exact = "LEAKAGE_PRESENT"
    return {
        "dataset_name": name,
        "n": len(rows),
        "missing_files": missing,
        "split_sizes": split_counts,
        "provenance": provenance,
        "file_hash_leakage": {**file_l, "cross_split_overlap_n": file_l["cross_split_overlap_n"]},
        "pixel_hash_leakage": pix_l,
        "near_duplicate_ahash": near,
        "source_overlap_n": n_overlap(src_ov) if src_ov else None,
        "subjects": subj,
        "leakage_flags": leak_status,
        "exact_hash_status": exact,
    }


def inspect_xgboost_vqc():
    from ml.training.train_xgboost import generate_multimodal_dataset, train_xgboost
    from ml.training import train_vqc as tv
    from ml.classifiers.vqc_classifier import VQCClassifier

    src_xgb = inspect.getsource(train_xgboost)
    src_vqc = inspect.getsource(tv.train_vqc)
    src_vqc_cls = inspect.getsource(VQCClassifier.train)

    X, y = generate_multimodal_dataset(num_samples=200, seed=42)
    X_train, X_val, X_test = X[:140], X[140:170], X[170:]
    row_hashes = [sha256_bytes(np.asarray(row, dtype=np.float32).tobytes()) for row in X]
    split_rows = (
        [{"sample_id": f"mm_{i:03d}", "split": "train", "pixel_sha256": row_hashes[i], "file_sha256": row_hashes[i], "subject_id": "", "source_id": "generate_multimodal_dataset", "ahash": None} for i in range(140)]
        + [{"sample_id": f"mm_{i:03d}", "split": "val", "pixel_sha256": row_hashes[i], "file_sha256": row_hashes[i], "subject_id": "", "source_id": "generate_multimodal_dataset", "ahash": None} for i in range(140, 170)]
        + [{"sample_id": f"mm_{i:03d}", "split": "test", "pixel_sha256": row_hashes[i], "file_sha256": row_hashes[i], "subject_id": "", "source_id": "generate_multimodal_dataset", "ahash": None} for i in range(170, 200)]
    )
    vec_leak = hash_leakage(split_rows, "file_sha256")
    unique_vecs = len(set(row_hashes))

    scaler_path = os.path.join(ROOT, VQC_DIR, "scaler.pkl")
    pca_path = os.path.join(ROOT, VQC_DIR, "pca.pkl")
    vqc_scaler = vqc_pca = None
    if os.path.isfile(scaler_path):
        with open(scaler_path, "rb") as handle:
            vqc_scaler = pickle.load(handle)
    if os.path.isfile(pca_path):
        with open(pca_path, "rb") as handle:
            vqc_pca = pickle.load(handle)

    scaler_n = None
    pca_n = None
    if vqc_scaler is not None:
        scaler_n = int(getattr(vqc_scaler, "n_samples_seen_", -1))
    if vqc_pca is not None:
        pca_n = int(getattr(vqc_pca, "n_samples_seen_", getattr(vqc_pca, "n_samples_", -1)))

    return {
        "dataset_name": "synthetic_multimodal_fusion",
        "origin": "ml/training/train_xgboost.py::generate_multimodal_dataset",
        "provenance": {
            "class": "SYNTHETIC",
            "synthetic_or_real": "synthetic",
            "simulated_or_real": "simulated",
            "np_random": True,
            "not_real_world": True,
            "subject_ids": False,
            "genuinely_paired_clinical_samples": 0,
        },
        "n": 200,
        "split_sizes": {"train": 140, "val": 30, "test": 30},
        "split_method": "index_slice seed=42  [:140] / [140:170] / [170:]",
        "unique_feature_vectors": unique_vecs,
        "duplicate_vectors": 200 - unique_vecs,
        "vector_hash_leakage": vec_leak,
        "subjects": subject_status(split_rows, None),
        "xgboost": {
            "scaler_fitted_on_train_only": "scaler.fit_transform(X_train)" in src_xgb and "fit_transform(X)" not in src_xgb.replace("fit_transform(X_train)", ""),
            "pca_fitted_on_train_only": "pca.fit(X_train_scaled)" in src_xgb,
            "pca_used_for_booster": False,
            "booster_trained_on_raw_unscaled_X_train": True,
            "scaler_saved_to_disk": False,
            "feature_selection_present": False,
            "hyperparameter_search_on_test": False,
            "hyperparameters": "hardcoded n_estimators=60 max_depth=3 learning_rate=0.08; no GridSearch/RandomizedSearch",
            "val_used_for_early_stopping_or_tuning": False,
            "val_merged_into_test_eval": False,
            "test_eval_uses_X_test_only": "xgb.predict(X_test)" in src_xgb,
            "inference_uses_scaler": False,
            "notes": [
                "StandardScaler is fitted on train then discarded; the booster is fit on raw X_train.",
                "PCA is fitted on train-scaled features and written to metadata variance only; not used at train or infer.",
                "Validation split is allocated but not used for early stopping or hyperparameter search.",
                "Held-out metrics use X[170:] only.",
            ],
        },
        "vqc": {
            "scaler_fitted_on_train_only": "self.scaler.fit_transform(X)" in src_vqc_cls and "vqc.train(X_train, y_train" in src_vqc,
            "pca_fitted_on_train_only": "self.pca.fit_transform(X_scaled)" in src_vqc_cls and "vqc.train(X_train, y_train" in src_vqc,
            "on_disk_scaler_n_samples_seen": scaler_n,
            "on_disk_pca_n_samples": pca_n,
            "on_disk_matches_train_n_140": scaler_n == 140 and pca_n == 140,
            "feature_selection_present": False,
            "optimization_on_train_only": "vqc.train(X_train, y_train" in src_vqc and "opt.step_and_cost" in src_vqc_cls,
            "val_assigned_in_trainer": False,
            "val_merged_into_test_eval": False,
            "test_eval_uses_X_test_only": "for row in X_test" in src_vqc,
            "hyperparameter_search_on_test": False,
            "quantum_backend": "PennyLane default.qubit — classical SIMULATED circuit, not a QPU",
            "notes": [
                "train_vqc.py never materializes X_val; samples 140:170 are unused, not merged into test.",
                "VQCClassifier.train fits scaler+PCA on the array it is given (X_train, n=140).",
                "On-disk scaler.pkl / pca.pkl n_samples_seen should equal 140.",
            ],
        },
        "records": split_rows,
    }


def inspect_sensor():
    from ml.training import train_sensor_model as tsm
    src = inspect.getsource(tsm.train_sensor_model)
    manifest = load_csv(os.path.join(ROOT, "data", "datasets", "manifests", "sensor_manifest.csv"))
    rows = []
    for rec in manifest:
        vec = np.array([
            rec["peak_g_force"], rec["peak_accel_ms2"], rec["peak_jerk_gs"],
            rec["accel_variance"], rec["gyro_variance"], rec["sma"],
            rec["stabilization_seconds"], rec["impact_flag"],
        ], dtype=np.float32)
        rows.append({
            "sample_id": rec["sample_id"],
            "split": rec["split"],
            "subject_id": rec["subject_id"],
            "source_id": rec.get("source_dataset") or "",
            "file_sha256": sha256_bytes(vec.tobytes()),
            "pixel_sha256": sha256_bytes(vec.tobytes()),
            "ahash": None,
            "canonical_label": rec.get("canonical_label"),
        })
    raw_dir = os.path.join(ROOT, "data", "datasets", "raw", "sisfall_uci_har")
    return {
        "dataset_name": "synthetic_50hz_motion_windows",
        "manifest": "data/datasets/manifests/sensor_manifest.csv",
        "origin": "ml/data/prepare_sensor_dataset.py::prepare_sensor_dataset (np.random windows)",
        "sisfall_uci_har_raw_dir_exists": os.path.isdir(raw_dir),
        "source_dataset_column_value": "SisFall / UCI HAR Motion Telemetry",
        "source_column_is_schema_label_not_download": True,
        "provenance": {
            "class": "SYNTHETIC",
            "also": "SIMULATED",
            "synthetic_or_real": "synthetic",
            "simulated_or_real": "simulated",
            "np_random": True,
            "not_downloaded_sisfall_or_uci_har": True,
            "not_real_world": True,
        },
        "n": len(rows),
        "split_sizes": dict(Counter(r["split"] for r in rows)),
        "vector_hash_leakage": hash_leakage(rows, "file_sha256"),
        "subjects": subject_status(rows, "SYNTHETIC_GENERATOR_LABEL"),
        "source_overlap_n": n_overlap(cross_split(split_sets(rows, "source_id"))),
        "scaler": {
            "fitted_on_train_only": "scaler.fit_transform(X_train)" in src,
            "val_and_test_transform_only": "scaler.transform(X_val)" in src and "scaler.transform(X_test)" in src,
            "feature_selection_present": False,
            "hyperparameter_search_on_test": False,
            "val_merged_into_test_eval": False,
            "saved_scaler": "ml/models/sensor_scaler.pkl",
        },
        "records": rows,
    }


def main():
    os.chdir(ROOT)
    yolo_meta = read_json(YOLO_METADATA)
    eff_meta = read_json(EFFNET_METADATA)
    unet_meta = read_json(UNET_METADATA)
    xgb_meta = read_json(XGB_METADATA)
    vqc_meta = read_json(os.path.join(VQC_DIR, "vqc_metadata.json"))
    sensor_meta = read_json(SENSOR_METADATA)

    print("Auditing YOLO yolo_retrain_v2 ...")
    yolo_rows = image_rows_from_manifest(
        os.path.join(ROOT, "data", "datasets", "yolo_retrain_v2", "manifest.csv"),
        image_col="dest_image",
        sample_col="sample_id",
        source_col="source_dataset",
    )
    yolo_ds = summarize_image_dataset(
        "yolo_retrain_v2",
        yolo_rows,
        {
            "class": "SYNTHETIC",
            "demo_files": ["blank_skin.jpg (val empty-label)", "dummy_test.jpg (test empty-label)"],
            "synthetic_or_real": "synthetic",
            "simulated_or_real": "synthetic_drawings",
            "origin": "yolo_processed cut/bruise drawings from data/datasets/raw/synthetic_wound plus two existing negative files",
            "production_for": "YOLO11 Detection v1.4.0",
        },
        None,
    )

    print("Auditing public_wound_dataset (EfficientNet + U-Net production train set) ...")
    pub_rows = image_rows_from_manifest(
        os.path.join(ROOT, "data", "datasets", "public_wound_dataset", "manifest.csv"),
        image_col="image_path",
        source_col="source",
        root=os.path.join(ROOT, "data", "datasets", "public_wound_dataset"),
    )
    pub_ds = summarize_image_dataset(
        "public_wound_dataset",
        pub_rows,
        {
            "class": "SYNTHETIC",
            "claimed_in_csv_source": "Kaggle/Roboflow/WOUNDSEG Public Wound Taxonomy",
            "actual_origin": "ml/training/download_public_datasets.py::generate_expanded_wound_dataset PIL drawings",
            "not_downloaded_kaggle": True,
            "synthetic_or_real": "synthetic",
            "simulated_or_real": "synthetic_drawings",
            "do_not_label_PUBLIC": "CSV source/license text is a taxonomy label, not a download receipt.",
            "production_for": ["EfficientNetV2 Classification v1.3.0", "ResNet34-UNet Segmentation v1.3.0"],
        },
        "SYNTHETIC_GENERATOR_LABEL",
    )

    print("Auditing injury_dataset ...")
    inj_rows = image_rows_from_manifest(
        os.path.join(ROOT, "data", "datasets", "injury_dataset", "manifest.csv"),
        image_col="image_path",
        source_col="source",
        root=os.path.join(ROOT, "data", "datasets", "injury_dataset"),
    )
    inj_ds = summarize_image_dataset(
        "injury_dataset",
        inj_rows,
        {
            "class": "SYNTHETIC",
            "origin": "ml/training/prepare_dataset.py synthetic drawings",
            "synthetic_or_real": "synthetic",
            "used_by": "unet_processed unique-hash pool (not production U-Net SHA)",
        },
        "SYNTHETIC_GENERATOR_LABEL",
    )

    print("Auditing efficientnet_processed (candidate set, not production) ...")
    eff_rows = image_rows_from_manifest(
        os.path.join(ROOT, "data", "datasets", "efficientnet_processed", "manifest.csv"),
        image_col="image_path",
        source_col="source_dataset",
    )
    eff_ds = summarize_image_dataset(
        "efficientnet_processed",
        eff_rows,
        {
            "class": "SYNTHETIC",
            "role": "candidate training set; production EfficientNet was trained on public_wound_dataset",
            "synthetic_or_real": "synthetic",
        },
        None,
    )

    print("Auditing unet_processed (candidate set, not production) ...")
    unet_rows = image_rows_from_manifest(
        os.path.join(ROOT, "data", "datasets", "unet_processed", "manifest.csv"),
        image_col="image_path",
        source_col="source_dataset",
    )
    unet_ds = summarize_image_dataset(
        "unet_processed",
        unet_rows,
        {
            "class": "SYNTHETIC",
            "role": "candidate training set; production U-Net was trained on public_wound_dataset",
            "synthetic_or_real": "synthetic",
        },
        None,
    )

    print("Auditing XGBoost/VQC synthetic matrix ...")
    mm = inspect_xgboost_vqc()
    mm_rows = mm.pop("records")

    print("Auditing sensor ...")
    sensor = inspect_sensor()
    sensor_rows = sensor.pop("records")

    demo = {
        "dataset_name": "demo_and_eval_only_files",
        "provenance": {"class": "DEMO"},
        "files": [
            {"path": "data/sample/image/football_injury.jpg", "role": "DEMO photograph used in UI/OOD probes", "exists": os.path.isfile(os.path.join(ROOT, "data/sample/image/football_injury.jpg"))},
            {"path": "data/datasets/yolo_injury/blank_skin.jpg", "role": "existing no-injury file; YOLO val empty-label; EfficientNet/U-Net eval-only", "exists": os.path.isfile(os.path.join(ROOT, "data/datasets/yolo_injury/blank_skin.jpg"))},
            {"path": "data/datasets/yolo_injury/dummy_test.jpg", "role": "existing no-injury file; YOLO test empty-label; EfficientNet/U-Net eval-only", "exists": os.path.isfile(os.path.join(ROOT, "data/datasets/yolo_injury/dummy_test.jpg"))},
        ],
    }

    prior = {
        "data/datasets/DATASET_PROVENANCE_AUDIT.json": (
            "Incorrectly labels PIL drawings as MedWound/Kaggle public downloads and "
            "np.random sensor windows as SisFall & UCI HAR. Those claims are rejected."
        ),
        "data/datasets/LEAKAGE_AUDIT.json": (
            "Claimed ZERO_SUBJECT_LEAKAGE and 0 SHA-256 duplicates. Generator subject IDs "
            "are not patients. public_wound_dataset has exact pixel-hash groups across splits."
        ),
    }

    def conclusion(ds, extra=None):
        flags = ds.get("leakage_flags") or []
        exact = ds.get("exact_hash_status") or ds.get("vector_hash_leakage", {}).get("leakage")
        subj = (ds.get("subjects") or {}).get("subject_leakage_status")
        return extra or {
            "exact_hash": "LEAKAGE" if (ds.get("pixel_hash_leakage") or {}).get("leakage") or (ds.get("file_hash_leakage") or {}).get("leakage") or (ds.get("vector_hash_leakage") or {}).get("leakage") else "NO_EXACT_HASH_CROSS_SPLIT",
            "near_duplicate": "NEAR_DUP_CROSS_SPLIT" if (ds.get("near_duplicate_ahash") or {}).get("pair_count", 0) > 0 else "NO_AHASH_CROSS_SPLIT_OR_NOT_IMAGE",
            "subject": subj,
            "flags": flags,
        }

    models = [
        {
            "model": "YOLO11 Detection",
            "production_dataset": "yolo_retrain_v2",
            "metadata_dataset_name": (yolo_meta.get("metrics") or {}).get("dataset_name"),
            "data_provenance": "SYNTHETIC",
            "split_sizes": yolo_ds["split_sizes"],
            "subject_counts": yolo_ds["subjects"]["unique_subject_ids"],
            "duplicate_overlap": yolo_ds["pixel_hash_leakage"]["cross_split_overlap_n"],
            "subject_overlap": yolo_ds["subjects"]["subject_overlap_n"],
            "near_duplicate_cross_split_pairs": yolo_ds["near_duplicate_ahash"]["pair_count"],
            "leakage_status": conclusion(yolo_ds),
            "final_conclusion": "SYNTHETIC drawings. No subject_id. Exact-hash splits copied from yolo_processed. SUBJECT_LEAKAGE_NOT_VERIFIABLE. Near-duplicate canvases remain a generalization risk.",
        },
        {
            "model": "EfficientNetV2 Classification",
            "production_dataset": "public_wound_dataset",
            "metadata_dataset_name": (eff_meta.get("metrics") or {}).get("dataset_name"),
            "candidate_dataset_not_loaded": "efficientnet_processed",
            "data_provenance": "SYNTHETIC",
            "split_sizes": pub_ds["split_sizes"],
            "subject_counts": pub_ds["subjects"]["unique_subject_ids"],
            "duplicate_overlap": pub_ds["pixel_hash_leakage"]["cross_split_overlap_n"],
            "subject_overlap": pub_ds["subjects"]["subject_overlap_n"],
            "near_duplicate_cross_split_pairs": pub_ds["near_duplicate_ahash"]["pair_count"],
            "leakage_status": conclusion(pub_ds),
            "final_conclusion": "Production weights trained on PIL drawings with generator subject_ids. Pixel-hash and near-dup leakage across splits. Do not call this PUBLIC or REAL. SUBJECT_LEAKAGE_NOT_VERIFIABLE.",
        },
        {
            "model": "ResNet34-UNet Segmentation",
            "production_dataset": "public_wound_dataset",
            "metadata_dataset_name": (unet_meta.get("metrics") or {}).get("dataset_name"),
            "candidate_dataset_not_loaded": "unet_processed",
            "data_provenance": "SYNTHETIC",
            "split_sizes": pub_ds["split_sizes"],
            "subject_counts": pub_ds["subjects"]["unique_subject_ids"],
            "duplicate_overlap": pub_ds["pixel_hash_leakage"]["cross_split_overlap_n"],
            "subject_overlap": pub_ds["subjects"]["subject_overlap_n"],
            "near_duplicate_cross_split_pairs": pub_ds["near_duplicate_ahash"]["pair_count"],
            "leakage_status": conclusion(pub_ds),
            "final_conclusion": "Same production train set as EfficientNet. Unique-hash unet_processed is a later candidate, not the live SHA. SUBJECT_LEAKAGE_NOT_VERIFIABLE.",
        },
        {
            "model": "XGBoost Multimodal",
            "production_dataset": "synthetic_multimodal_fusion",
            "metadata_dataset_name": xgb_meta.get("data_provenance"),
            "data_provenance": "SYNTHETIC",
            "split_sizes": mm["split_sizes"],
            "subject_counts": 0,
            "duplicate_overlap": mm["vector_hash_leakage"]["cross_split_overlap_n"],
            "subject_overlap": mm["subjects"]["subject_overlap_n"],
            "leakage_status": {
                "exact_hash": "LEAKAGE" if mm["vector_hash_leakage"]["leakage"] else "NO_EXACT_VECTOR_CROSS_SPLIT",
                "subject": "SUBJECT_LEAKAGE_NOT_VERIFIABLE",
                "scaler_pca": mm["xgboost"],
            },
            "final_conclusion": "np.random 23-d vectors. Not real-world. No subject_id. Scaler/PCA fitted on train then unused by the booster. Test slice [170:] not merged with val. SUBJECT_LEAKAGE_NOT_VERIFIABLE.",
        },
        {
            "model": "Experimental 4-Qubit VQC",
            "production_dataset": "synthetic_multimodal_fusion (same generator/split as XGBoost)",
            "metadata_dataset_name": (vqc_meta.get("dataset_provenance") or vqc_meta.get("data_provenance_class")),
            "data_provenance": "SYNTHETIC",
            "quantum_backend": "SIMULATED",
            "split_sizes": mm["split_sizes"],
            "subject_counts": 0,
            "duplicate_overlap": mm["vector_hash_leakage"]["cross_split_overlap_n"],
            "subject_overlap": mm["subjects"]["subject_overlap_n"],
            "leakage_status": {
                "exact_hash": "LEAKAGE" if mm["vector_hash_leakage"]["leakage"] else "NO_EXACT_VECTOR_CROSS_SPLIT",
                "subject": "SUBJECT_LEAKAGE_NOT_VERIFIABLE",
                "scaler_pca": mm["vqc"],
            },
            "final_conclusion": "Same SYNTHETIC matrix as XGBoost. Scaler+PCA+Adam on train n=140 only. Val unused, not in test. default.qubit is simulated. SUBJECT_LEAKAGE_NOT_VERIFIABLE.",
        },
        {
            "model": "Sensor Motion Event Classifier",
            "production_dataset": "synthetic_50hz_motion_windows",
            "metadata_dataset_name": (sensor_meta.get("metrics") or {}).get("dataset_name"),
            "data_provenance": "SYNTHETIC",
            "also": "SIMULATED",
            "split_sizes": sensor["split_sizes"],
            "subject_counts": sensor["subjects"]["unique_subject_ids"],
            "duplicate_overlap": sensor["vector_hash_leakage"]["cross_split_overlap_n"],
            "subject_overlap": sensor["subjects"]["subject_overlap_n"],
            "leakage_status": {
                "exact_hash": "LEAKAGE" if sensor["vector_hash_leakage"]["leakage"] else "NO_EXACT_VECTOR_CROSS_SPLIT",
                "subject": sensor["subjects"]["subject_leakage_status"],
                "scaler": sensor["scaler"],
            },
            "final_conclusion": "np.random 50 Hz windows with synthetic subj_001… IDs. Not SisFall/UCI HAR files (raw dir missing). Generator IDs may be disjoint; real-patient leakage is SUBJECT_LEAKAGE_NOT_VERIFIABLE. Scaler fit on train only.",
        },
    ]

    manifest = {
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": ROOT.replace("\\", "/"),
        "method": {
            "file_hash": "SHA-256 of file bytes",
            "pixel_hash": "SHA-256 of RGB pixels plus WxH",
            "near_duplicate": f"8x8 average hash Hamming <= {AHASH_NEAR}",
            "tabular_hash": "SHA-256 of float32 feature bytes",
            "subject_rule": "Missing or synthetic generator IDs => SUBJECT_LEAKAGE_NOT_VERIFIABLE. Never claim ZERO_SUBJECT_LEAKAGE for real patients.",
            "np_random_is_not_real_world": True,
        },
        "prior_documents_rejected": prior,
        "labels_used": ["REAL", "SYNTHETIC", "SIMULATED", "PUBLIC", "DEMO"],
        "no_dataset_labeled_REAL": True,
        "no_dataset_labeled_PUBLIC_as_downloaded": True,
        "models": models,
        "datasets": {
            "yolo_retrain_v2": yolo_ds,
            "public_wound_dataset": pub_ds,
            "injury_dataset": inj_ds,
            "efficientnet_processed": eff_ds,
            "unet_processed": unet_ds,
            "synthetic_multimodal_fusion": mm,
            "synthetic_50hz_motion_windows": sensor,
            "demo_and_eval_only_files": demo,
        },
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    rec_fields = ["dataset", "sample_id", "split", "subject_id", "source_id", "path", "file_sha256", "pixel_sha256"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rec_fields, extrasaction="ignore")
        writer.writeheader()
        for name, rows in (
            ("yolo_retrain_v2", yolo_rows),
            ("public_wound_dataset", pub_rows),
            ("injury_dataset", inj_rows),
            ("efficientnet_processed", eff_rows),
            ("unet_processed", unet_rows),
            ("synthetic_multimodal_fusion", mm_rows),
            ("synthetic_50hz_motion_windows", sensor_rows),
        ):
            for row in rows:
                writer.writerow({
                    "dataset": name,
                    "sample_id": row.get("sample_id"),
                    "split": row.get("split"),
                    "subject_id": row.get("subject_id") or "",
                    "source_id": row.get("source_id") or "",
                    "path": row.get("path") or "",
                    "file_sha256": row.get("file_sha256") or "",
                    "pixel_sha256": row.get("pixel_sha256") or "",
                })

    print(json.dumps({
        "wrote": _rel(OUT_JSON),
        "records": _rel(OUT_CSV),
        "yolo_pixel_cross": yolo_ds["pixel_hash_leakage"]["cross_split_overlap_n"],
        "yolo_near": yolo_ds["near_duplicate_ahash"]["pair_count"],
        "public_pixel_cross": pub_ds["pixel_hash_leakage"]["cross_split_overlap_n"],
        "public_near": pub_ds["near_duplicate_ahash"]["pair_count"],
        "public_unique_pixel": pub_ds["pixel_hash_leakage"]["unique_hashes"],
        "eff_processed_pixel_cross": eff_ds["pixel_hash_leakage"]["cross_split_overlap_n"],
        "unet_processed_pixel_cross": unet_ds["pixel_hash_leakage"]["cross_split_overlap_n"],
        "mm_vec_cross": mm["vector_hash_leakage"]["cross_split_overlap_n"],
        "mm_unique": mm["unique_feature_vectors"],
        "vqc_scaler_n": mm["vqc"]["on_disk_scaler_n_samples_seen"],
        "sensor_subj": sensor["subjects"]["subject_overlap_n"],
        "sensor_vec_cross": sensor["vector_hash_leakage"]["cross_split_overlap_n"],
    }, indent=2))


if __name__ == "__main__":
    main()
