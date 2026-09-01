"""Build real sensor motion manifest from SisFall + UCI HAR.

Activity-name mapping (not peak-g heuristics):
  SisFall F*          -> fall
  SisFall D18, D19    -> impact  (stumble / jump)
  SisFall other D*    -> normal_activity
  UCI HAR (all)       -> normal_activity

Subject-level train/val/test splits. UCI subsampled so impact is not drowned.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.dataset_registry import generate_hierarchical_splits
from ml.data.prepare_sensor_dataset import extract_kinetic_sensor_features

ROOT = Path(__file__).resolve().parents[2]
SISFALL_ROOT = ROOT / "data" / "raw" / "sisfall" / "SisFall_dataset" / "SisFall_dataset"
HAR_ROOT = ROOT / "data" / "raw" / "uci" / "har_smartphones" / "UCI HAR Dataset"
MANIFEST_PATH = ROOT / "data" / "datasets" / "manifests" / "sensor_real_manifest.csv"
PROCESSED_CSV = ROOT / "data" / "datasets" / "processed" / "sensor_motion" / "sensor_real_features.csv"
REPORT_PATH = ROOT / "data" / "manifests" / "sensor_real_prepare_report.json"

FEATURE_COLS = [
    "peak_g_force",
    "peak_accel_ms2",
    "peak_jerk_gs",
    "accel_variance",
    "gyro_variance",
    "sma",
    "stabilization_seconds",
    "impact_flag",
]

# SisFall ADXL345 ±16 g, 13-bit → g = raw / 256; ITG3200 → deg/s = raw / 14.375
ADXL_LSB_PER_G = 256.0
ITG_LSB_PER_DPS = 14.375
G = 9.80665
SISFALL_FS = 200.0
TARGET_FS = 50.0
WINDOW_SEC = 2.5
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)  # 125

IMPACT_CODES = {"D18", "D19"}  # stumble while walking, gently jump

UCI_ACTIVITIES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_sisfall_file(path: Path) -> np.ndarray:
    """Return (N, 9) float array of raw SisFall columns."""
    rows = []
    text = path.read_text(encoding="latin-1", errors="ignore")
    for line in text.splitlines():
        line = line.strip().rstrip(";").strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        try:
            rows.append([float(parts[i]) for i in range(9)])
        except ValueError:
            continue
    if not rows:
        return np.zeros((0, 9), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _sisfall_to_si(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ADXL345 + ITG3200 raw ints to accel m/s^2 and gyro rad/s."""
    ax = raw[:, 0] / ADXL_LSB_PER_G * G
    ay = raw[:, 1] / ADXL_LSB_PER_G * G
    az = raw[:, 2] / ADXL_LSB_PER_G * G
    gx = np.deg2rad(raw[:, 3] / ITG_LSB_PER_DPS)
    gy = np.deg2rad(raw[:, 4] / ITG_LSB_PER_DPS)
    gz = np.deg2rad(raw[:, 5] / ITG_LSB_PER_DPS)
    accel = np.stack([ax, ay, az], axis=1)
    gyro = np.stack([gx, gy, gz], axis=1)
    return accel, gyro


def _downsample(accel: np.ndarray, gyro: np.ndarray, factor: int = 4) -> tuple[np.ndarray, np.ndarray]:
    return accel[::factor], gyro[::factor]


def _window_around_peak(accel: np.ndarray, gyro: np.ndarray, n: int = WINDOW_SAMPLES) -> tuple[np.ndarray, np.ndarray] | None:
    if len(accel) < max(8, n // 4):
        return None
    mag = np.linalg.norm(accel, axis=1)
    peak = int(np.argmax(mag))
    half = n // 2
    start = max(0, peak - half)
    end = start + n
    if end > len(accel):
        end = len(accel)
        start = max(0, end - n)
    if end - start < n:
        # pad edges with edge values
        pad = n - (end - start)
        a = np.pad(accel[start:end], ((0, pad), (0, 0)), mode="edge")
        g = np.pad(gyro[start:end], ((0, pad), (0, 0)), mode="edge")
        return a, g
    return accel[start:end], gyro[start:end]


def _middle_window(accel: np.ndarray, gyro: np.ndarray, n: int = WINDOW_SAMPLES) -> tuple[np.ndarray, np.ndarray] | None:
    if len(accel) < max(8, n // 4):
        return None
    if len(accel) <= n:
        pad = n - len(accel)
        a = np.pad(accel, ((0, pad), (0, 0)), mode="edge")
        g = np.pad(gyro, ((0, pad), (0, 0)), mode="edge")
        return a, g
    start = (len(accel) - n) // 2
    return accel[start : start + n], gyro[start : start + n]


def _label_sisfall(code: str) -> tuple[str, str]:
    if code.startswith("F"):
        return "fall", code
    if code in IMPACT_CODES:
        return "impact", code
    if code.startswith("D"):
        return "normal_activity", code
    return "normal_activity", code


def _collect_sisfall() -> list[dict]:
    records = []
    if not SISFALL_ROOT.is_dir():
        return records
    for path in sorted(SISFALL_ROOT.rglob("*.txt")):
        if path.name.lower().startswith("readme"):
            continue
        m = re.match(r"^(D\d+|F\d+)_((?:SA|SE)\d+)_R(\d+)\.txt$", path.name, re.I)
        if not m:
            continue
        code, subject, trial = m.group(1).upper(), m.group(2).upper(), m.group(3)
        canon, orig = _label_sisfall(code)
        raw = _parse_sisfall_file(path)
        if len(raw) < 16:
            continue
        accel, gyro = _sisfall_to_si(raw)
        accel, gyro = _downsample(accel, gyro, factor=int(SISFALL_FS // TARGET_FS))
        if canon == "fall" or canon == "impact":
            win = _window_around_peak(accel, gyro)
        else:
            win = _middle_window(accel, gyro)
        if win is None:
            continue
        a, g = win
        feats = extract_kinetic_sensor_features(
            a[:, 0], a[:, 1], a[:, 2], g[:, 0], g[:, 1], g[:, 2], sampling_rate_hz=TARGET_FS
        )
        records.append(
            {
                "sample_id": f"sisfall_{path.stem}",
                "subject_id": f"sisfall_{subject}",
                "source": "SisFall",
                "activity": orig,
                "original_label": orig,
                "canonical_label": canon,
                "sampling_rate_hz": TARGET_FS,
                "window_seconds": WINDOW_SEC,
                **dict(zip(FEATURE_COLS, feats)),
            }
        )
    return records


def _collect_uci(max_samples: int = 2000, seed: int = 42) -> list[dict]:
    records = []
    train_dir = HAR_ROOT / "train"
    test_dir = HAR_ROOT / "test"
    if not train_dir.is_dir():
        return records
    rng = np.random.RandomState(seed)
    for split_name, base in (("train", train_dir), ("test", test_dir)):
        if not base.is_dir():
            continue
        y_path = base / ("y_train.txt" if split_name == "train" else "y_test.txt")
        s_path = base / ("subject_train.txt" if split_name == "train" else "subject_test.txt")
        labels = np.loadtxt(y_path, dtype=int)
        subjects = np.loadtxt(s_path, dtype=int)
        # Prefer total_acc (includes gravity) in g units
        acc_x = np.loadtxt(base / "Inertial Signals" / f"total_acc_x_{split_name}.txt")
        acc_y = np.loadtxt(base / "Inertial Signals" / f"total_acc_y_{split_name}.txt")
        acc_z = np.loadtxt(base / "Inertial Signals" / f"total_acc_z_{split_name}.txt")
        gyro_x = np.loadtxt(base / "Inertial Signals" / f"body_gyro_x_{split_name}.txt")
        gyro_y = np.loadtxt(base / "Inertial Signals" / f"body_gyro_y_{split_name}.txt")
        gyro_z = np.loadtxt(base / "Inertial Signals" / f"body_gyro_z_{split_name}.txt")
        for i in range(acc_x.shape[0]):
            act = UCI_ACTIVITIES.get(int(labels[i]), "WALKING")
            # UCI accel in g → m/s^2; gyro already rad/s
            ax = acc_x[i] * G
            ay = acc_y[i] * G
            az = acc_z[i] * G
            # UCI windows are 128 samples @ 50 Hz (~2.56 s); trim/pad to 125
            n = WINDOW_SAMPLES
            if len(ax) >= n:
                ax, ay, az = ax[:n], ay[:n], az[:n]
                gx, gy, gz = gyro_x[i][:n], gyro_y[i][:n], gyro_z[i][:n]
            else:
                pad = n - len(ax)
                ax = np.pad(ax, (0, pad), mode="edge")
                ay = np.pad(ay, (0, pad), mode="edge")
                az = np.pad(az, (0, pad), mode="edge")
                gx = np.pad(gyro_x[i], (0, pad), mode="edge")
                gy = np.pad(gyro_y[i], (0, pad), mode="edge")
                gz = np.pad(gyro_z[i], (0, pad), mode="edge")
            feats = extract_kinetic_sensor_features(ax, ay, az, gx, gy, gz, sampling_rate_hz=TARGET_FS)
            records.append(
                {
                    "sample_id": f"uci_{split_name}_{i:05d}",
                    "subject_id": f"uci_subj_{int(subjects[i]):02d}",
                    "source": "UCI_HAR",
                    "activity": act,
                    "original_label": act,
                    "canonical_label": "normal_activity",
                    "sampling_rate_hz": TARGET_FS,
                    "window_seconds": WINDOW_SEC,
                    **dict(zip(FEATURE_COLS, feats)),
                }
            )
    if len(records) > max_samples:
        idx = rng.choice(len(records), size=max_samples, replace=False)
        records = [records[i] for i in sorted(idx)]
    return records


def prepare_sensor_real(uci_max: int = 2000, seed: int = 42) -> pd.DataFrame:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sisfall = _collect_sisfall()
    uci = _collect_uci(max_samples=uci_max, seed=seed)
    records = sisfall + uci
    if not records:
        raise RuntimeError("No real sensor samples found. Run acquire_sisfall.py and ensure UCI HAR is present.")

    df = pd.DataFrame(records)
    df_split = generate_hierarchical_splits(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed)
    df_split.to_csv(MANIFEST_PATH, index=False)
    df_split.to_csv(PROCESSED_CSV, index=False)

    train_subs = set(df_split.loc[df_split["split"] == "train", "subject_id"])
    test_subs = set(df_split.loc[df_split["split"] == "test", "subject_id"])
    overlap = train_subs & test_subs

    report = {
        "created_utc": _utc(),
        "status": "OK",
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
        "rows": int(len(df_split)),
        "by_source": df_split["source"].value_counts().to_dict(),
        "by_label": df_split["canonical_label"].value_counts().to_dict(),
        "by_split": df_split["split"].value_counts().to_dict(),
        "subjects": int(df_split["subject_id"].nunique()),
        "subject_leakage_train_test": int(len(overlap)),
        "label_mapping": {
            "SisFall_F_star": "fall",
            "SisFall_D18_D19": "impact",
            "SisFall_other_D": "normal_activity",
            "UCI_HAR_all": "normal_activity",
        },
        "uci_max_subsample": uci_max,
        "data_provenance_class": "REAL",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return df_split


if __name__ == "__main__":
    prepare_sensor_real()
