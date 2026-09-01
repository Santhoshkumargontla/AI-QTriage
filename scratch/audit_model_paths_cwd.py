"""Probe model path resolution from project root, backend/, and a foreign cwd."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.models.canonical_paths import (  # noqa: E402
    EFFNET_CANONICAL,
    UNET_CANONICAL,
    VQC_DIR,
    XGB_CANONICAL,
    YOLO_CANONICAL,
    abs_path,
    resolve_existing,
)


def probe_cwd(label: str, cwd: str) -> dict:
    os.chdir(cwd)
    rels = {
        "YOLO": YOLO_CANONICAL,
        "EfficientNet": EFFNET_CANONICAL,
        "U-Net": UNET_CANONICAL,
        "XGBoost": XGB_CANONICAL,
        "VQC": os.path.join(VQC_DIR, "vqc_weights.npz"),
        "eval": os.path.join("data", "results", "evaluation_results.json"),
        "xgb_meta": os.path.join("ml", "models", "xgboost_metadata.json"),
        "registry": os.path.join("ml", "models", "model_registry.json"),
    }
    rows = {}
    for name, rel in rels.items():
        rows[name] = {
            "declared": rel.replace("\\", "/"),
            "os.path.exists(rel)": os.path.exists(rel),
            "os.path.exists(abs_path)": os.path.exists(abs_path(rel)),
            "resolve_existing": resolve_existing(rel).replace("\\", "/"),
            "resolve_exists": os.path.exists(resolve_existing(rel)),
        }
    load = {}
    try:
        from ml.vision.yolo_wrapper import YOLO11Detector
        det = YOLO11Detector()
        load["yolo"] = {
            "status": det.status,
            "loaded": det.model is not None,
            "model_path": str(det.model_path).replace("\\", "/"),
            "sha256": det.artifact_sha256,
            "classes": list(det.class_list),
        }
    except Exception as exc:  # noqa: BLE001
        load["yolo"] = {"error": repr(exc)}
    try:
        from ml.classifiers.xgboost_classifier import XGBoostClassifier, load_xgboost_metadata
        clf = XGBoostClassifier(XGB_CANONICAL)
        meta = load_xgboost_metadata()
        load["xgb"] = {
            "status": clf.status,
            "loaded": clf.is_trained,
            "model_path": str(clf.model_path).replace("\\", "/"),
            "sha256": clf.artifact_sha256,
            "metadata_keys": sorted(meta.keys())[:12],
            "metadata_empty": not bool(meta),
        }
    except Exception as exc:  # noqa: BLE001
        load["xgb"] = {"error": repr(exc)}
    try:
        from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
        clf = EfficientNetV2Classifier()
        load["effnet"] = {
            "loaded": clf.is_loaded,
            "model_path": str(clf.model_path).replace("\\", "/") if clf.model_path else None,
        }
    except Exception as exc:  # noqa: BLE001
        load["effnet"] = {"error": repr(exc)}
    try:
        from ml.vision.unet_wrapper import UNetSegmenter
        seg = UNetSegmenter()
        load["unet"] = {
            "loaded": seg.is_loaded,
            "model_path": str(seg.model_path).replace("\\", "/") if seg.model_path else None,
        }
    except Exception as exc:  # noqa: BLE001
        load["unet"] = {"error": repr(exc)}
    try:
        from ml.classifiers.vqc_classifier import VQCClassifier
        vqc = VQCClassifier(VQC_DIR)
        load["vqc"] = {
            "status": vqc.status,
            "loaded": vqc.is_trained,
            "model_dir": str(vqc.model_dir).replace("\\", "/") if vqc.model_dir else None,
        }
    except Exception as exc:  # noqa: BLE001
        load["vqc"] = {"error": repr(exc)}
    try:
        from ml.classifiers.sensor_classifier import SensorClassifier
        sc = SensorClassifier()
        load["sensor"] = {
            "loaded": sc.is_trained,
            "model_path": str(sc.model_path).replace("\\", "/"),
            "error": sc.load_error,
        }
    except Exception as exc:  # noqa: BLE001
        load["sensor"] = {"error": repr(exc)}

    return {
        "label": label,
        "cwd": os.getcwd(),
        "paths": rows,
        "loads": load,
    }


def main() -> None:
    foreign = os.path.join(os.environ.get("TEMP", "C:\\Windows\\Temp"), "aiqtriage_cwd_probe")
    os.makedirs(foreign, exist_ok=True)
    backend = os.path.join(ROOT, "backend")
    results = [
        probe_cwd("project_root", ROOT),
        probe_cwd("backend_dir", backend),
        probe_cwd("foreign_temp", foreign),
    ]
    dest = os.path.join(ROOT, "scratch", "audit_model_paths_cwd.json")
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"wrote {dest}")
    for block in results:
        print(f"\n=== {block['label']} cwd={block['cwd']} ===")
        for name, row in block["paths"].items():
            print(
                f"  {name}: exists(rel)={row['os.path.exists(rel)']} "
                f"exists(abs)={row['os.path.exists(abs_path)']} "
                f"resolve_ok={row['resolve_exists']}"
            )
        for name, row in block["loads"].items():
            print(f"  LOAD {name}: {row}")


if __name__ == "__main__":
    main()
