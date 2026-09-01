"""Train YOLO expanded skin candidate (6 classes). Promote only if gates pass."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.models.canonical_paths import YOLO_CANONICAL, YOLO_PRETRAINED_INIT, abs_path, sha256_file

DATA_YAML = ROOT / "data" / "datasets" / "yolo_expanded_v1" / "data.yaml"
PREPARE = ROOT / "data" / "datasets" / "yolo_expanded_v1" / "PREPARE_REPORT.json"
OUT_DIR = ROOT / "ml" / "models" / "yolo_expanded_skin_candidate"
FORENSIC = ROOT / "data" / "uploads" / "3f629ca8-dd98-427d-a708-f976e2042555.jpeg"
SEED = 42
EPOCHS_DEFAULT = 8
BATCH = 8
IMGSZ = 416
KEEP = 0.25
HAND_GT = [550.0, 200.0, 750.0, 480.0]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (aa + bb - inter + 1e-9))


def _predict(model, path: Path, conf: float) -> list[dict]:
    if not path.is_file():
        return []
    res = model.predict(str(path), conf=conf, verbose=False)
    out = []
    for r in res:
        if r.boxes is None:
            continue
        names = r.names
        for box in r.boxes:
            xyxy = [round(float(x), 2) for x in box.xyxy[0].tolist()]
            cid = int(box.cls[0].item())
            out.append(
                {
                    "class": names.get(cid, str(cid)),
                    "confidence": float(box.conf[0].item()),
                    "bounding_box": xyxy,
                }
            )
    return out


def _hand_eval(model, conf: float) -> dict:
    dets = _predict(model, FORENSIC, conf)
    best_iou = 0.0
    for d in dets:
        best_iou = max(best_iou, _iou(d["bounding_box"], HAND_GT))
    cuts = [d for d in dets if d["class"] == "cut"]
    return {
        "n": len(dets),
        "n_cut": len(cuts),
        "covers_injury": best_iou >= 0.2,
        "cut_covers": any(_iou(d["bounding_box"], HAND_GT) >= 0.2 for d in cuts),
        "best_iou": round(best_iou, 4),
        "classes": sorted({d["class"] for d in dets}),
    }


def _neg_fps(model, conf: float) -> dict:
    neg_dir = ROOT / "data" / "datasets" / "yolo_expanded_v1" / "_ood_gen"
    paths = [
        neg_dir / "black.png",
        neg_dir / "white.png",
        neg_dir / "gray.png",
        ROOT / "data" / "datasets" / "yolo_injury" / "blank_skin.jpg",
    ]
    out = {}
    for p in paths:
        dets = _predict(model, p, conf)
        out[p.name] = {"n": len(dets), "dets": dets[:3]}
    return out


def _val_metrics(model, split: str) -> dict:
    metrics = model.val(data=str(DATA_YAML), split=split, workers=0, verbose=False)
    box = getattr(metrics, "box", None)
    result = {
        "mAP50": float(getattr(box, "map50", 0.0) or 0.0) if box is not None else None,
        "mAP50-95": float(getattr(box, "map", 0.0) or 0.0) if box is not None else None,
        "precision": float(getattr(box, "mp", 0.0) or 0.0) if box is not None else None,
        "recall": float(getattr(box, "mr", 0.0) or 0.0) if box is not None else None,
    }
    names = getattr(metrics, "names", None) or {}
    if isinstance(names, dict):
        name_list = [names[i] for i in sorted(names)]
    else:
        name_list = list(names)
    per = {}
    if box is not None and hasattr(box, "ap50") and box.ap50 is not None:
        for i, ap in enumerate(box.ap50):
            label = name_list[i] if i < len(name_list) else str(i)
            per[label] = float(ap)
    result["per_class_ap50"] = per
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    if not DATA_YAML.is_file() or not PREPARE.is_file():
        raise SystemExit("Run prepare_yolo_expanded_v1.py first")
    prepare = json.loads(PREPARE.read_text(encoding="utf-8"))
    if not prepare.get("leakage_free"):
        raise SystemExit("Leakage detected — refuse training")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    init = abs_path(YOLO_PRETRAINED_INIT) if Path(abs_path(YOLO_PRETRAINED_INIT)).is_file() else "yolo11n.pt"
    print("INIT", init, flush=True)
    model = YOLO(init)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        project=str(OUT_DIR),
        name="run_v1",
        exist_ok=True,
        pretrained=True,
        patience=5,
        device="cpu",
        workers=0,
    )

    best = OUT_DIR / "run_v1" / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit("best.pt missing after train")

    cand = YOLO(str(best))
    cand_test = _val_metrics(cand, "test")
    cand_hand_001 = _hand_eval(cand, 0.01)
    cand_hand_025 = _hand_eval(cand, KEEP)
    cand_neg = _neg_fps(cand, KEEP)

    support = prepare.get("class_support_honest") or {}
    gates = {
        "gate_leakage_free": bool(prepare.get("leakage_free")),
        "gate_all_classes_nonzero_train": all(
            int((support.get(c) or {}).get("train_boxes") or 0) > 0 for c in prepare["names"]
        ),
        "gate_blank_clean": all(
            cand_neg.get(k, {}).get("n", 0) == 0 for k in ("black.png", "white.png", "gray.png", "blank_skin.jpg") if k in cand_neg
        ),
        "gate_hand_localization_ok": bool(
            cand_hand_025.get("covers_injury") or cand_hand_001.get("cut_covers")
        ),
        "gate_sha_distinct": sha256_file(str(best.relative_to(ROOT))) != sha256_file(YOLO_CANONICAL),
        "gate_map50_reasonable": isinstance(cand_test.get("mAP50"), float) and cand_test["mAP50"] >= 0.20,
    }
    all_pass = all(gates.values())
    recommendation = "PROMOTE" if all_pass else "KEEP_BASELINE"
    promoted = False

    report = {
        "created_utc": _utc(),
        "recommendation": recommendation,
        "promoted_to_production": False,
        "promotion_gates": gates,
        "failure_reasons": [k for k, v in gates.items() if not v],
        "epochs": args.epochs,
        "batch": BATCH,
        "seed": SEED,
        "init": str(init),
        "dataset": {
            "names": prepare["names"],
            "image_counts": prepare["image_counts"],
            "box_counts": prepare["box_counts"],
            "class_support_honest": support,
        },
        "production_sha256": sha256_file(YOLO_CANONICAL),
        "candidate_sha256": sha256_file(str(best.relative_to(ROOT))),
        "candidate_path": str(best.relative_to(ROOT)).replace("\\", "/"),
        "candidate_test": cand_test,
        "forensic_hand": {"001": cand_hand_001, "025": cand_hand_025},
        "negatives_025": cand_neg,
        "keep_threshold": KEEP,
        "status": "READY_FOR_PROMOTION" if all_pass else "PROMOTION_REJECTED",
    }

    if args.promote and all_pass:
        backup = Path(abs_path(YOLO_CANONICAL) + ".pre_expanded_v1_backup")
        shutil.copy2(abs_path(YOLO_CANONICAL), backup)
        shutil.copy2(best, abs_path(YOLO_CANONICAL))
        meta_path = ROOT / "ml" / "models" / "vision" / "yolo11_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        meta.update(
            {
                "model_name": "YOLO11 Detection",
                "version": "expanded-skin-v1",
                "status": "INFERENCE_EXECUTES",
                "classes": prepare["names"],
                "untrained_classes": ["fracture", "swelling", "Normal", "OOD_Reject"],
                "classes_with_zero_training_boxes": [
                    c for c, s in support.items() if int(s.get("train_boxes") or 0) <= 0
                ],
                "canonical_path": "ml/models/vision/yolo11_injury_best.pt",
                "artifact_sha256": sha256_file(YOLO_CANONICAL),
                "previous_canonical_sha256": report["production_sha256"],
                "backup_path": str(backup.relative_to(ROOT)).replace("\\", "/"),
                "data_provenance_class": "PUBLIC_REAL_PHOTOS",
                "dataset_provenance": (
                    "Roboflow injury_detection_v2+wound2+aid; shubhambaid burns; "
                    "HF WSEG wound boxes; Stab→laceration; Normal/OOD empty labels"
                ),
                "training_status": "TRAINED_AND_EVALUATED",
                "training_was_real": True,
                "training_command": "backend/venv/Scripts/python.exe ml/training/train_yolo_expanded_v1.py --epochs 8 --promote",
                "evaluation_artifact": "ml/models/yolo_expanded_skin_candidate/TRAIN_EVAL_REPORT.json",
                "metrics": {
                    "dataset_name": "yolo_expanded_v1",
                    "dataset_type": "public_real_photos_merged",
                    "independent_test": cand_test,
                    "mAP50": cand_test.get("mAP50"),
                    "mAP50-95": cand_test.get("mAP50-95"),
                    "precision": cand_test.get("precision"),
                    "recall": cand_test.get("recall"),
                    "promoted": True,
                    "selected_checkpoint": "yolo_expanded_skin_candidate",
                },
                "known_limitations": (
                    "Research only. Laceration has few boxes (Stab remap). "
                    "Fracture is X-ray-only separate model. Normal/OOD are not detection classes. Not clinical."
                ),
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        support_doc = {
            "schema_version": "1.0",
            "promotion_status": "PROMOTED_EXPANDED_SKIN_V1",
            "dataset_provenance": meta["dataset_provenance"],
            "classes": support,
        }
        (ROOT / "ml" / "models" / "vision" / "yolo11_class_support.json").write_text(
            json.dumps(support_doc, indent=2), encoding="utf-8"
        )
        promoted = True
        report["promoted_to_production"] = True
        report["status"] = "PROMOTED"
        report["backup"] = str(backup.relative_to(ROOT)).replace("\\", "/")

    (OUT_DIR / "TRAIN_EVAL_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "PROMOTION.json").write_text(
        json.dumps({"recommendation": recommendation, "promoted": promoted, "gates": gates}, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "recommendation": recommendation,
                "promoted": promoted,
                "gates": gates,
                "cand_mAP50": cand_test.get("mAP50"),
                "hand_025": cand_hand_025,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
