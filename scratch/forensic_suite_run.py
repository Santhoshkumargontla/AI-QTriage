"""Independent forensic sweep: artifacts + fresh synthetic test images + model probes."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("PYTHONPATH", str(ROOT))

OUT_DIR = ROOT / "scratch" / "forensic_suite_2026_08_29"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUT_DIR / "test_images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def make_images() -> dict[str, Path]:
    """Create forensic test images NOT taken from training manifests."""
    rng = np.random.default_rng(20260829)
    paths = {}

    # blank / extremes
    for name, arr in {
        "blank_black": np.zeros((320, 320, 3), np.uint8),
        "blank_white": np.full((320, 320, 3), 255, np.uint8),
        "blank_gray": np.full((320, 320, 3), 128, np.uint8),
        "noise": rng.integers(0, 256, (320, 320, 3), dtype=np.uint8),
    }.items():
        p = IMG_DIR / f"{name}.png"
        cv2.imwrite(str(p), arr)
        paths[name] = p

    # normal-ish skin tone (no injury mark)
    skin = np.full((360, 480, 3), (145, 170, 205), np.uint8)  # BGR peachy
    skin = np.clip(skin.astype(np.float32) + rng.normal(0, 6, skin.shape), 0, 255).astype(np.uint8)
    p = IMG_DIR / "normal_skin.png"
    cv2.imwrite(str(p), skin)
    paths["normal_skin"] = p

    # cut-like: red linear gash on skin
    cut = skin.copy()
    cv2.line(cut, (80, 120), (380, 250), (20, 20, 180), 6)
    cv2.line(cut, (82, 122), (378, 248), (40, 40, 220), 2)
    p = IMG_DIR / "synth_cut.png"
    cv2.imwrite(str(p), cut)
    paths["synth_cut"] = p

    # bruise-like: purple blotch
    bruise = skin.copy()
    cv2.ellipse(bruise, (240, 180), (70, 45), 20, 0, 360, (90, 40, 60), -1)
    bruise = cv2.GaussianBlur(bruise, (21, 21), 0)
    # re-blend skin texture edges
    mask = np.zeros(bruise.shape[:2], np.uint8)
    cv2.ellipse(mask, (240, 180), (70, 45), 20, 0, 360, 255, -1)
    base = skin.copy()
    base[mask > 0] = bruise[mask > 0]
    p = IMG_DIR / "synth_bruise.png"
    cv2.imwrite(str(p), base)
    paths["synth_bruise"] = p

    # wound-like: irregular dark red region
    wound = skin.copy()
    pts = np.array([[200, 140], [280, 130], [300, 200], [220, 230], [180, 190]], np.int32)
    cv2.fillPoly(wound, [pts], (15, 15, 120))
    cv2.polylines(wound, [pts], True, (30, 30, 200), 2)
    p = IMG_DIR / "synth_wound.png"
    cv2.imwrite(str(p), wound)
    paths["synth_wound"] = p

    # swelling-like: lighter raised blob (brighter patch)
    swell = skin.copy()
    cv2.circle(swell, (240, 180), 55, (180, 200, 230), -1)
    swell = cv2.GaussianBlur(swell, (31, 31), 0)
    p = IMG_DIR / "synth_swelling.png"
    cv2.imwrite(str(p), swell)
    paths["synth_swelling"] = p

    # unrelated natural-ish gradient scene
    h, w = 300, 400
    yy, xx = np.mgrid[0:h, 0:w]
    unrelated = np.stack(
        [
            (xx * 0.4).astype(np.uint8),
            (60 + yy * 0.5).astype(np.uint8),
            (30 + (xx + yy) * 0.15).astype(np.uint8),
        ],
        axis=-1,
    )
    p = IMG_DIR / "unrelated_scene.png"
    cv2.imwrite(str(p), unrelated)
    paths["unrelated_scene"] = p

    # low quality / blurry
    blur = cv2.GaussianBlur(cut, (51, 51), 0)
    p = IMG_DIR / "low_quality_blur.png"
    cv2.imwrite(str(p), blur)
    paths["low_quality_blur"] = p

    return paths


def probe_models(paths: dict[str, Path]) -> dict:
    from ml.models.canonical_paths import (
        EFFNET_CANONICAL,
        EFFNET_METADATA,
        UNET_CANONICAL,
        UNET_METADATA,
        YOLO_CANONICAL,
        YOLO_METADATA,
        XGB_CANONICAL,
        VQC_WEIGHTS,
        SENSOR_MODEL,
        REGISTRY_PATH,
        exists,
        read_json,
        sha256_file,
    )
    from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier, interpret_prediction
    from ml.vision.unet_wrapper import UNetSegmenter, interpret_segmentation
    from ml.vision.yolo_wrapper import YOLO11Detector

    artifacts = {}
    for name, path, meta in [
        ("yolo", YOLO_CANONICAL, YOLO_METADATA),
        ("effnet", EFFNET_CANONICAL, EFFNET_METADATA),
        ("unet", UNET_CANONICAL, UNET_METADATA),
        ("xgb", XGB_CANONICAL, "ml/models/xgboost_metadata.json"),
        ("vqc", VQC_WEIGHTS, "ml/models/vqc/vqc_metadata.json"),
        ("sensor", SENSOR_MODEL, "ml/models/sensor_metadata.json"),
    ]:
        artifacts[name] = {
            "path": path,
            "exists": exists(path),
            "sha256": sha256_file(path) if exists(path) else None,
            "meta": read_json(meta) if exists(meta) else {},
        }

    yolo = YOLO11Detector()
    eff = EfficientNetV2Classifier()
    unet = UNetSegmenter()

    results = {}
    for name, path in paths.items():
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = yolo.detect(str(path))
        raw = eff.predict_raw(rgb, temperature=1.0)
        gated = interpret_prediction(eff.predict(rgb))
        mask, px, ratio, dbg = unet.segment(rgb)
        seg = interpret_segmentation(mask, px, ratio, dbg)
        results[name] = {
            "file_sha256": _sha_file(path),
            "shape_hwc": list(bgr.shape),
            "pixel_mean": float(bgr.mean()),
            "pixel_std": float(bgr.std()),
            "yolo": dets,
            "effnet_raw": raw,
            "effnet_gated": gated,
            "unet": {
                "raw_pos_ratio": dbg.get("raw_positive_ratio"),
                "pos_px": px,
                "affected_ratio": ratio,
                "reliable": seg.get("is_reliable"),
                "withheld": seg.get("mask_withheld"),
                "status": seg.get("status"),
                "reason": dbg.get("reason") or seg.get("display_message"),
            },
        }
    return {"artifacts": artifacts, "registry_keys": list(read_json(REGISTRY_PATH).keys()), "probes": results}


def main():
    paths = make_images()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": {k: str(v) for k, v in paths.items()},
    }
    report.update(probe_models(paths))
    out = OUT_DIR / "forensic_model_probes.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("WROTE", out)
    # compact table
    for name, row in report["probes"].items():
        y = row["yolo"]
        e = row["effnet_raw"]
        g = row["effnet_gated"]
        u = row["unet"]
        top = y[0] if y else None
        conf = f"{top['confidence']:.3f}" if top else "-"
        print(
            f"{name:20} YOLO={top['finding'] if top else None}@{conf} "
            f"EFF_raw={e.get('winner')}@{float(e.get('max_prob') or 0):.3f} "
            f"gate={g.get('status')}/{g.get('winner')}/{g.get('abstention_class')} "
            f"UNET raw_pos={u.get('raw_pos_ratio')} withheld={u.get('withheld')}"
        )


if __name__ == "__main__":
    main()
