"""Extract UCI HAR normal-activity features for sensor baseline. Fall/impact remain unmapped."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HAR_ROOT = ROOT / "data" / "raw" / "uci" / "har_smartphones"
OUT = ROOT / "data" / "processed" / "sensor"
MANIFEST = ROOT / "data" / "manifests" / "sensor_uci_har_manifest.csv"

ACTIVITIES = {
    "WALKING": "normal_activity",
    "WALKING_UPSTAIRS": "normal_activity",
    "WALKING_DOWNSTAIRS": "normal_activity",
    "SITTING": "normal_activity",
    "STANDING": "normal_activity",
    "LAYING": "normal_activity",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _window_features(acc: np.ndarray, gyro: np.ndarray, fs: float = 50.0) -> list[float]:
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
    gx, gy, gz = gyro[:, 0], gyro[:, 1], gyro[:, 2]
    g = np.sqrt(ax**2 + ay**2 + az**2) / 9.80665
    peak_g = float(np.max(g))
    peak_accel = float(np.max(g * 9.80665))
    dt = 1.0 / fs
    jerk = np.diff(g) / dt if len(g) > 1 else np.array([0.0])
    peak_jerk = float(np.max(np.abs(jerk)))
    accel_var = float(np.var(g))
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    gyro_var = float(np.var(gyro_mag))
    sma = float(np.sum(np.abs(ax) + np.abs(ay) + np.abs(az)) / len(ax))
    peak_idx = int(np.argmax(g))
    post = g[peak_idx:]
    stabilization_sec = float(np.sum(post < 1.2) * dt)
    impact_flag = 1.0 if peak_g > 3.0 else 0.0
    return [peak_g, peak_accel, peak_jerk, accel_var, gyro_var, sma, stabilization_sec, impact_flag]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_dir = HAR_ROOT / "UCI HAR Dataset" / "train"
    test_dir = HAR_ROOT / "UCI HAR Dataset" / "test"
    if not train_dir.is_dir():
        # alternate extract layout
        candidates = list(HAR_ROOT.rglob("Inertial Signals"))
        if not candidates:
            report = {"status": "BLOCKED", "reason": "UCI HAR layout not found", "har_root": str(HAR_ROOT)}
            (ROOT / "data" / "manifests" / "sensor_uci_har_report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report))
            return

    records = []
    for split, base in (("train", train_dir), ("test", test_dir)):
        if not base.is_dir():
            continue
        labels = np.loadtxt(base / "y_train.txt" if split == "train" else base / "y_test.txt", dtype=int)
        subjects = np.loadtxt(base / "subject_train.txt" if split == "train" else base / "subject_test.txt", dtype=int)
        acc_x = np.loadtxt(base / "Inertial Signals" / f"body_acc_x_{split}.txt")
        acc_y = np.loadtxt(base / "Inertial Signals" / f"body_acc_y_{split}.txt").reshape(acc_x.shape)
        acc_z = np.loadtxt(base / "Inertial Signals" / f"body_acc_z_{split}.txt").reshape(acc_x.shape)
        gyro_x = np.loadtxt(base / "Inertial Signals" / f"body_gyro_x_{split}.txt").reshape(acc_x.shape)
        gyro_y = np.loadtxt(base / "Inertial Signals" / f"body_gyro_y_{split}.txt").reshape(acc_x.shape)
        gyro_z = np.loadtxt(base / "Inertial Signals" / f"body_gyro_z_{split}.txt").reshape(acc_x.shape)
        act_names = list(ACTIVITIES.keys())
        for i in range(acc_x.shape[0]):
            act = act_names[int(labels[i]) - 1]
            canonical = ACTIVITIES[act]
            feats = _window_features(
                np.stack([acc_x[i], acc_y[i], acc_z[i]], axis=1),
                np.stack([ gyro_x[i], gyro_y[i], gyro_z[i]], axis=1),
            )
            records.append(
                {
                    "subject_id": f"uci_subj_{int(subjects[i]):02d}",
                    "split": "train" if split == "train" else "test",
                    "source": "UCI_HAR",
                    "activity": act,
                    "canonical_label": canonical,
                    **dict(zip(
                        ["peak_g_force", "peak_accel_ms2", "peak_jerk_gs", "accel_variance", "gyro_variance", "sma", "stabilization_seconds", "impact_flag"],
                        feats,
                    )),
                }
            )

    df = pd.DataFrame(records)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)
    report = {
        "created_utc": _utc(),
        "status": "OK",
        "rows": len(df),
        "label_mapping_limitation": "UCI HAR provides normal ADL only. Fall/impact labels require KFall/MobiFall — not downloaded.",
        "usable_for": "normal_activity baseline features",
        "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
    }
    (ROOT / "data" / "manifests" / "sensor_uci_har_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
