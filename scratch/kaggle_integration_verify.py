"""Independent verification for Kaggle-era model integration. No mocked metrics."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

REPORT = ROOT / "scratch" / "kaggle_integration_verify_report.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    from ml.models.canonical_paths import (
        EFFNET_CANONICAL,
        YOLO_CANONICAL,
        UNET_CANONICAL,
        MANIFEST_PATH,
        REGISTRY_PATH,
        abs_path,
        exists,
    )
    from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
    from ml.vision.yolo_wrapper import YOLO11Detector

    out: dict = {"workspace": str(ROOT)}

    # --- Phase 0/1: SHA chain ---
    eff_path = Path(abs_path(EFFNET_CANONICAL))
    yolo_path = Path(abs_path(YOLO_CANONICAL))
    unet_path = Path(abs_path(UNET_CANONICAL))
    eff_sha = sha256_file(eff_path) if eff_path.exists() else None
    clf = EfficientNetV2Classifier()
    from ml.models.canonical_paths import sha256_file as cp_sha256

    sidecar = Path(abs_path("ml/models/vision/efficientnetv2_injury_best_classes.json"))
    sidecar_classes = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else []
    wrapper_sha = cp_sha256(EFFNET_CANONICAL) if exists(EFFNET_CANONICAL) else None
    meta_path = ROOT / "ml" / "models" / "vision" / "efficientnetv2_metadata.json"
    meta_eff = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    manifest = json.loads(Path(abs_path(MANIFEST_PATH)).read_text(encoding="utf-8"))
    registry = json.loads(Path(abs_path(REGISTRY_PATH)).read_text(encoding="utf-8"))
    eff_reg = registry.get("EfficientNetV2 Classification") or {}
    man_eff = next((m for m in manifest.get("models", []) if "EfficientNet" in m.get("model_name", "")), {})

    out["efficientnet"] = {
        "disk_sha256": eff_sha,
        "claimed_sha256": "95cf385d85419a63eeef8e46de9c9ef7e7487ef0f326fb60ca376b8bae0eec9f",
        "sha_match_claim": eff_sha == "95cf385d85419a63eeef8e46de9c9ef7e7487ef0f326fb60ca376b8bae0eec9f",
        "wrapper_sha256": wrapper_sha,
        "registry_sha256": eff_reg.get("artifact_sha256"),
        "manifest_sha256": man_eff.get("sha256"),
        "chain_agrees": eff_sha == wrapper_sha == eff_reg.get("artifact_sha256"),
        "manifest_stale": man_eff.get("sha256") != eff_sha,
        "version": meta_eff.get("version"),
        "sidecar_classes": sidecar_classes,
        "wrapper_classes": clf.classes,
        "n_classes": len(sidecar_classes),
    }

    # --- Phase 4: recompute test metrics from manifest ---
    import pandas as pd

    manifest_csv = ROOT / "data" / "datasets" / "efficientnet_kaggle_v1" / "manifest.csv"
    metrics = {"recomputed": None, "error": None}
    if manifest_csv.exists() and eff_path.exists():
        try:
            df = pd.read_csv(manifest_csv)
            test_df = df[df["split"] == "test"].copy()
            classes = sidecar_classes
            class_to_idx = {c: i for i, c in enumerate(classes)}
            path_col = "image_path" if "image_path" in df.columns else "path"
            label_col = "class" if "class" in df.columns else "label"

            y_true, y_pred = [], []
            for _, row in test_df.iterrows():
                img_path = ROOT / str(row[path_col])
                if not img_path.exists():
                    continue
                label = str(row[label_col]).lower()
                if label not in class_to_idx:
                    continue
                img = np.array(Image.open(img_path).convert("RGB"))
                raw = clf.predict_raw(img)
                winner = str(raw.get("winner", "")).lower().replace(" ", "_")
                if winner.startswith("ood"):
                    winner = "ood_reject"
                if winner not in class_to_idx:
                    probs = raw.get("probs") or {}
                    if probs:
                        winner = max(probs, key=probs.get).lower()
                        if winner.startswith("ood"):
                            winner = "ood_reject"
                if winner not in class_to_idx:
                    continue
                y_true.append(class_to_idx[label])
                y_pred.append(class_to_idx[winner])

            from sklearn.metrics import (
                accuracy_score,
                classification_report,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
            )

            labels = list(range(len(classes)))
            report = classification_report(
                y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
            metrics["recomputed"] = {
                "n": len(y_true),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "per_class": {c: report.get(c, {}) for c in classes},
                "confusion_matrix": cm,
            }
            meta_path = ROOT / "ml" / "models" / "vision" / "efficientnetv2_metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            claimed = meta.get("metrics", {}).get("test", {})
            metrics["claimed"] = {
                "n": claimed.get("n"),
                "macro_f1": claimed.get("macro_f1"),
                "accuracy": claimed.get("accuracy"),
            }
            metrics["delta"] = {
                "n_diff": metrics["recomputed"]["n"] - (claimed.get("n") or 0),
                "macro_f1_diff": metrics["recomputed"]["macro_f1"] - (claimed.get("macro_f1") or 0),
            }
        except Exception as exc:
            metrics["error"] = str(exc)
    out["efficientnet_metrics"] = metrics

    # --- Phase 5: OOD probes ---
    from PIL import Image  # noqa: F811 — used in metrics block

    probes = {}
    for name, color in [
        ("black", (0, 0, 0)),
        ("white", (255, 255, 255)),
        ("gray", (128, 128, 128)),
        ("blank_skin", (220, 190, 170)),
    ]:
        img = np.array(Image.new("RGB", (224, 224), color))
        parsed = clf.predict(img)
        raw = clf.predict_raw(img)
        probes[name] = {
            "gated_winner": parsed.get("__winner") or parsed.get("winner"),
            "raw_winner": raw.get("winner"),
            "is_confident": parsed.get("__is_confident"),
            "model_status": parsed.get("__status"),
            "raw_top_prob": raw.get("max_prob"),
        }
    out["ood_probes"] = probes

    # --- Phase 8: YOLO burn candidate ---
    cand = ROOT / "ml" / "models" / "yolo_skin_kaggle_v1" / "run_v1" / "weights" / "best.pt"
    yolo_info = YOLO11Detector().get_info()
    burn_gate = {"candidate_exists": cand.exists(), "canonical_sha": sha256_file(yolo_path) if yolo_path.exists() else None}
    if cand.exists():
        burn_gate["candidate_sha256"] = sha256_file(cand)
        burn_gate["promoted"] = burn_gate["candidate_sha256"] == burn_gate["canonical_sha"]
    train_report = ROOT / "ml" / "models" / "yolo_skin_kaggle_v1" / "TRAIN_REPORT.json"
    if train_report.exists():
        tr = json.loads(train_report.read_text(encoding="utf-8"))
        burn_gate["burn_metrics"] = (tr.get("promotion_gate") or {}).get("burn") or tr.get("burn_test_metrics")
    out["yolo_skin"] = {"runtime": yolo_info, "burn_candidate": burn_gate}

    # --- Phase 9: Fracture isolation ---
    frac = ROOT / "ml" / "models" / "vision" / "yolo11_fracture_xray_best.pt"
    out["fracture_yolo"] = {
        "exists": frac.exists(),
        "sha256_prefix": sha256_file(frac)[:16] if frac.exists() else None,
        "loaded_by_skin_yolo": "fracture" in str(yolo_path).lower(),
    }
    if frac.exists():
        tr = ROOT / "ml" / "models" / "yolo_fracture_xray_v1" / "TRAIN_REPORT.json"
        if tr.exists():
            out["fracture_yolo"]["train_report"] = json.loads(tr.read_text(encoding="utf-8")).get("test_metrics")

    # --- Security: token pattern scan (no printing secrets) ---
    patterns = [r"hf_[A-Za-z0-9]{20,}", r"KGAT_[a-z0-9]{32}"]
    import re

    hits = []
    for p in ROOT.rglob("*"):
        if p.is_dir() or ".git" in p.parts or "node_modules" in p.parts or "venv" in p.parts:
            continue
        if p.suffix in {".pt", ".pth", ".jpg", ".png", ".jpeg", ".webp", ".npz", ".pkl"}:
            continue
        if p.name in {".env", "access_token", "kaggle.json"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for pat in patterns:
            if re.search(pat, text):
                hits.append(str(p.relative_to(ROOT)))
                break
    out["security_scan"] = {"credential_pattern_hits_in_repo": hits[:20], "hit_count": len(hits)}

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
