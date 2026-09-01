"""Independent failure reproduction for AI-QTriage models — no training, no promotion."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "phase_retrain_failure_baseline.json"
os.chdir(ROOT)
os.environ.setdefault("PYTHONPATH", str(ROOT))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def make_probe(name: str, arr: np.ndarray) -> Path:
    d = ROOT / "scratch" / "ood_probes"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.png"
    cv2.imwrite(str(p), arr)
    return p


def main() -> None:
    from ml.models.canonical_paths import (
        EFFNET_CANONICAL,
        UNET_CANONICAL,
        YOLO_CANONICAL,
        abs_path,
        sha256_file,
    )
    from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier, interpret_prediction
    from ml.vision.unet_wrapper import UNetSegmenter, interpret_segmentation
    from ml.vision.yolo_wrapper import YOLO11Detector

    report: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha": {
            "yolo": sha256_file(YOLO_CANONICAL),
            "effnet": sha256_file(EFFNET_CANONICAL),
            "unet": sha256_file(UNET_CANONICAL),
        },
        "probes": {},
        "demo_image": {},
        "forensic_hand": {},
        "xgb_vqc_sensor": {},
    }

    h, w = 256, 256
    probes = {
        "black": np.zeros((h, w, 3), np.uint8),
        "white": np.full((h, w, 3), 255, np.uint8),
        "gray": np.full((h, w, 3), 128, np.uint8),
        "noise": np.random.default_rng(0).integers(0, 255, (h, w, 3), np.uint8),
    }
    # unrelated natural-ish: green field gradient
    yy, xx = np.mgrid[0:h, 0:w]
    unrelated = np.stack(
        [
            (xx * 0.3).astype(np.uint8),
            (80 + yy * 0.4).astype(np.uint8),
            (40 + xx * 0.1).astype(np.uint8),
        ],
        axis=-1,
    )
    probes["unrelated_green"] = unrelated

    yolo = YOLO11Detector()
    eff = EfficientNetV2Classifier()
    unet = UNetSegmenter()

    for name, arr in probes.items():
        path = make_probe(name, arr)
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        probs = eff.predict(rgb)
        parsed = interpret_prediction(probs)
        mask, px, ratio, dbg = unet.segment(rgb)
        seg = interpret_segmentation(mask, px, ratio, dbg)
        dets = yolo.detect(str(path))
        report["probes"][name] = {
            "path": str(path),
            "pixel_mean": float(arr.mean()),
            "pixel_std": float(arr.std()),
            "effnet_raw": {
                "winner": parsed.get("winner"),
                "max": parsed.get("max_prob"),
                "probs": parsed.get("class_probs"),
                "status": parsed.get("status"),
                "reason": parsed.get("reason"),
                "is_confident": parsed.get("is_confident"),
            },
            "unet": {
                "raw_pos_ratio": dbg.get("raw_positive_ratio"),
                "pos_px": px,
                "reliable": seg.get("is_reliable"),
                "withheld": seg.get("mask_withheld"),
                "status": seg.get("status"),
                "reason": dbg.get("reason") or seg.get("display_message"),
            },
            "yolo_dets": dets,
        }

    demo = abs_path("data/sample/image/football_injury.jpg")
    if os.path.isfile(demo):
        dets = yolo.detect(demo)
        bgr = cv2.imread(demo)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        probs = eff.predict(rgb)
        parsed = interpret_prediction(probs)
        mask, px, ratio, dbg = unet.segment(rgb)
        seg = interpret_segmentation(mask, px, ratio, dbg)
        report["demo_image"] = {
            "path": demo,
            "sha256": _sha(Path(demo)),
            "shape": list(bgr.shape),
            "yolo": dets,
            "effnet": {
                "winner": parsed.get("winner"),
                "max": parsed.get("max_prob"),
                "status": parsed.get("status"),
                "model_status_note": "see metadata NOT_TRUSTWORTHY",
            },
            "unet": {
                "raw_pos": dbg.get("raw_positive_ratio"),
                "reliable": seg.get("is_reliable"),
                "withheld": seg.get("mask_withheld"),
            },
        }

    hand = abs_path(
        "data/uploads/ac69884f-7a50-48e5-b7cc-a9bea9b20313.jpeg"
    )
    if os.path.isfile(hand):
        dets = yolo.detect(hand)
        bgr = cv2.imread(hand)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # use top yolo bbox ROI for effnet like analyze
        if dets:
            x1, y1, x2, y2 = [int(v) for v in dets[0]["bounding_box"]]
            roi = rgb[y1:y2, x1:x2]
        else:
            roi = rgb
        probs = eff.predict(roi)
        parsed = interpret_prediction(probs)
        bbox = dets[0]["bounding_box"] if dets else None
        mask, px, ratio, dbg = unet.segment(rgb, bbox)
        seg = interpret_segmentation(mask, px, ratio, dbg)
        report["forensic_hand"] = {
            "path": hand,
            "sha256": _sha(Path(hand)),
            "yolo_top": dets[0] if dets else None,
            "yolo_n": len(dets),
            "effnet": {
                "winner": parsed.get("winner"),
                "max": parsed.get("max_prob"),
                "status": parsed.get("status"),
            },
            "unet_roi": {
                "raw_pos": dbg.get("raw_positive_ratio"),
                "reliable": seg.get("is_reliable"),
                "withheld": seg.get("mask_withheld"),
            },
        }

    # XGB / VQC / sensor provenance via metadata (not inventing metrics)
    for meta_rel, key in [
        ("ml/models/xgboost_metadata.json", "xgboost"),
        ("ml/models/vqc/vqc_metadata.json", "vqc"),
        ("ml/models/sensor_metadata.json", "sensor"),
    ]:
        p = abs_path(meta_rel)
        if os.path.isfile(p):
            report["xgb_vqc_sensor"][key] = json.load(open(p, encoding="utf-8"))

    # Confirm held-out artifact
    held = abs_path("data/results/canonical_held_out_evaluation.json")
    if os.path.isfile(held):
        report["held_out_artifact"] = json.load(open(held, encoding="utf-8"))

    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("WROTE", OUT)
    # compact summary
    for k, v in report["probes"].items():
        e = v["effnet_raw"]
        u = v["unet"]
        print(
            f"{k:16} EFF {e['winner']}@{e['max']:.3f} gate={e['status']} | "
            f"UNET raw_pos={u['raw_pos_ratio']} withheld={u['withheld']} | "
            f"YOLO n={len(v['yolo_dets'])}"
        )
    if report.get("forensic_hand"):
        fh = report["forensic_hand"]
        print("HAND", fh.get("yolo_top"), fh.get("effnet"), fh.get("unet_roi"))


if __name__ == "__main__":
    main()
