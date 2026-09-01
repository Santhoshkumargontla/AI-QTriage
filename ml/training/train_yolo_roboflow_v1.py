"""Train YOLO Roboflow candidate — never overwrites production without --promote and gates."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from backend.config import settings
from ml.models.canonical_paths import YOLO_CANONICAL, YOLO_PRETRAINED_INIT, abs_path, sha256_file

DATA_YAML = ROOT / "data" / "datasets" / "yolo_roboflow_v1" / "data.yaml"
OUT_DIR = ROOT / "ml" / "models" / "yolo_roboflow_candidate_v1"
FORENSIC = ROOT / "data" / "uploads" / "3f629ca8-dd98-427d-a708-f976e2042555.jpeg"
NEG_IMAGES = [
    ROOT / "data" / "datasets" / "yolo_injury" / "blank_skin.jpg",
    ROOT / "data" / "datasets" / "yolo_injury" / "dummy_test.jpg",
]
SEED = 42
EPOCHS = 8
BATCH = 8
IMGSZ = 416
KEEP = 0.25
# heuristic injury region for forensic hand (mid palm / cut area from prior audit)
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
            out.append({"class": names.get(cid, str(cid)), "confidence": float(box.conf[0].item()), "bounding_box": xyxy})
    return out


def _hand_eval(model, conf: float) -> dict:
    dets = _predict(model, FORENSIC, conf)
    best_iou = 0.0
    best = None
    for d in dets:
        iou = _iou(d["bounding_box"], HAND_GT)
        if iou > best_iou:
            best_iou = iou
            best = d
    cuts = [d for d in dets if d["class"] == "cut"]
    bruises = [d for d in dets if d["class"] == "bruise"]
    return {
        "n": len(dets),
        "detections": dets,
        "best": best,
        "best_iou": round(best_iou, 4),
        "n_cut": len(cuts),
        "n_bruise": len(bruises),
        "covers_injury": best_iou >= 0.2,
        "cut_covers": any(_iou(d["bounding_box"], HAND_GT) >= 0.2 for d in cuts),
    }


def _neg_fps(model, conf: float) -> dict:
    out = {}
    for p in NEG_IMAGES:
        dets = _predict(model, p, conf)
        # also solid probes
        out[p.name] = {"n": len(dets), "detections": dets}
    # solid
    probe = OUT_DIR / "probe"
    probe.mkdir(parents=True, exist_ok=True)
    for name, color in [("black.png", 0), ("white.png", 255), ("gray.png", 128)]:
        img = np.full((256, 256, 3), color, dtype=np.uint8)
        path = probe / name
        cv2.imwrite(str(path), img)
        dets = _predict(model, path, conf)
        out[name] = {"n": len(dets), "detections": dets}
    return out


def _val_metrics(model, split_yaml_key: str = "test") -> dict:
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    # ultralytics val on test
    metrics = model.val(data=str(DATA_YAML), split=split_yaml_key, conf=KEEP, verbose=False)
    names = data.get("names", {})
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    else:
        names = {i: v for i, v in enumerate(names)}
    per = {}
    try:
        mp = metrics.box
        for i, name in names.items():
            per[name] = {
                "precision": float(mp.p[i]) if hasattr(mp, "p") and i < len(mp.p) else None,
                "recall": float(mp.r[i]) if hasattr(mp, "r") and i < len(mp.r) else None,
                "mAP50": float(mp.ap50[i]) if hasattr(mp, "ap50") and i < len(mp.ap50) else None,
                "mAP50-95": float(mp.ap[i]) if hasattr(mp, "ap") and i < len(mp.ap) else None,
            }
        summary = {
            "precision": float(mp.mp),
            "recall": float(mp.mr),
            "mAP50": float(mp.map50),
            "mAP50-95": float(mp.map),
            "per_class": per,
        }
    except Exception as exc:
        summary = {"error": str(exc)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    if not DATA_YAML.is_file():
        raise SystemExit("Dataset missing — run prepare_yolo_roboflow_v1.py first")

    prepare = json.loads((ROOT / "data" / "datasets" / "yolo_roboflow_v1" / "PREPARE_REPORT.json").read_text(encoding="utf-8"))
    if not prepare.get("leakage_free"):
        raise SystemExit("Leakage detected — refuse training")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    # Prefer COCO-pretrained nano weights; do not fine-tune production 3-class onto 3-class with abrasion remapped
    init = abs_path(YOLO_PRETRAINED_INIT) if Path(abs_path(YOLO_PRETRAINED_INIT)).is_file() else "yolo11n.pt"
    print("INIT", init, flush=True)
    model = YOLO(init)
    results = model.train(
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
    last = OUT_DIR / "run_v1" / "weights" / "last.pt"
    if not best.is_file():
        raise SystemExit("best.pt missing after train")

    cand = YOLO(str(best))
    prod = YOLO(abs_path(YOLO_CANONICAL))

    cand_test = _val_metrics(cand, "test")
    # production has different class set — evaluate production on same yaml may misalign classes
    # Still run for honesty, but mark schema mismatch
    try:
        prod_test = _val_metrics(prod, "test")
        prod_schema_note = "production_evaluated_on_roboflow_yaml_class_mismatch_possible"
    except Exception as exc:
        prod_test = {"error": str(exc)}
        prod_schema_note = "production_val_failed"

    cand_hand_001 = _hand_eval(cand, 0.01)
    cand_hand_025 = _hand_eval(cand, KEEP)
    prod_hand_001 = _hand_eval(prod, 0.01)
    prod_hand_025 = _hand_eval(prod, KEEP)

    cand_neg = _neg_fps(cand, KEEP)
    prod_neg = _neg_fps(prod, KEEP)

    cand_neg_n = sum(v["n"] for v in cand_neg.values())
    prod_neg_n = sum(v["n"] for v in prod_neg.values())

    gates = {
        "gate_leakage_free": bool(prepare.get("leakage_free")),
        "gate_all_classes_nonzero_train": all(
            prepare["class_support_honest"][c]["total_boxes"] > 0 for c in prepare["names"]
        ),
        "gate_negatives_no_worse": cand_neg_n <= prod_neg_n,
        "gate_blank_clean": all(cand_neg[k]["n"] == 0 for k in ("black.png", "white.png", "gray.png", "blank_skin.jpg") if k in cand_neg),
        "gate_hand_localization_improved_or_ok": (
            cand_hand_025["covers_injury"] or cand_hand_001["cut_covers"]
        ) and not (
            # worse: prod had wrong bruise; candidate must not fire bruise-only wrist junk at keep
            cand_hand_025["n_bruise"] > 0 and not cand_hand_025["covers_injury"] and cand_hand_025["n_cut"] == 0
        ),
        "gate_sha_distinct": sha256_file(str(best.relative_to(ROOT))) != sha256_file(YOLO_CANONICAL),
        "gate_map50_reasonable": isinstance(cand_test.get("mAP50"), float) and cand_test["mAP50"] >= 0.25,
    }
    # stricter: if candidate at keep has bruise FP on hand without covering injury → fail
    if cand_hand_025["n"] > 0 and not cand_hand_025["covers_injury"]:
        gates["gate_hand_no_wrong_region_at_keep"] = False
    else:
        gates["gate_hand_no_wrong_region_at_keep"] = True

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
        "dataset": prepare,
        "production_sha256": sha256_file(YOLO_CANONICAL),
        "candidate_sha256": sha256_file(str(best.relative_to(ROOT))),
        "candidate_path": str(best.relative_to(ROOT)).replace("\\", "/"),
        "candidate_test": cand_test,
        "production_test_on_same_yaml": prod_test,
        "production_schema_note": prod_schema_note,
        "forensic_hand": {
            "candidate_001": cand_hand_001,
            "candidate_025": cand_hand_025,
            "production_001": prod_hand_001,
            "production_025": prod_hand_025,
            "heuristic_gt": HAND_GT,
        },
        "negatives_025": {"candidate": cand_neg, "production": prod_neg},
        "keep_threshold": KEEP,
        "status": "PROMOTION_REJECTED" if not all_pass else "READY_FOR_PROMOTION",
    }

    if args.promote and all_pass:
        backup = Path(abs_path(YOLO_CANONICAL) + ".pre_roboflow_v1_backup")
        shutil.copy2(abs_path(YOLO_CANONICAL), backup)
        shutil.copy2(best, abs_path(YOLO_CANONICAL))
        # update metadata sidecar minimally
        meta_path = ROOT / "ml" / "models" / "vision" / "yolo11_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(
            {
                "version": "roboflow-v1",
                "classes": prepare["names"],
                "untrained_classes": ["wound", "laceration", "swelling"],
                "classes_with_zero_training_boxes": ["wound"],
                "artifact_sha256": sha256_file(YOLO_CANONICAL),
                "dataset_provenance": "Roboflow injury_detection_v2 + wound2 + aid (CC BY 4.0); cut/bruise/abrasion",
                "data_provenance_class": "PUBLIC_REAL_PHOTOS",
                "known_limitations": "Research only. wound not in trained head. Not clinical.",
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        promoted = True
        report["promoted_to_production"] = True
        report["status"] = "PROMOTED"
        report["backup"] = str(backup.relative_to(ROOT)).replace("\\", "/")

    (OUT_DIR / "TRAIN_EVAL_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "PROMOTION.json").write_text(
        json.dumps({"recommendation": recommendation, "promoted": promoted, "gates": gates}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"recommendation": recommendation, "gates": gates, "cand_mAP50": cand_test.get("mAP50"), "hand_025": cand_hand_025}, indent=2))


if __name__ == "__main__":
    main()
