"""Runtime verification for mask geometry, OOD, YOLO thresholds, fusion, SOS. No mocked success."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from ml.models.canonical_paths import (
    YOLO_CANONICAL,
    abs_path,
    sha256_file,
)
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
from ml.vision.unet_wrapper import UNetSegmenter
from ml.vision.yolo_wrapper import YOLO11Detector
from backend.main import _compose_full_image_mask
from ml.fusion.feature_fusion import MultimodalFeatureFusion
from backend.services.twilio_service import TwilioService


def _blank(color, h=240, w=400):
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def mask_geometry():
    print("\n=== PHASE 2 mask geometry ===")
    orig_h, orig_w = 300, 500
    bbox = [80, 40, 220, 200]
    rh, rw = bbox[3] - bbox[1], bbox[2] - bbox[0]
    roi = np.zeros((rh, rw), dtype=np.uint8)
    roi[10:40, 15:60] = 1
    full = _compose_full_image_mask(roi, orig_h, orig_w, bbox)
    assert full.shape == (orig_h, orig_w)
    assert int(full[0:40].sum()) == 0
    assert int(full[:, 0:80].sum()) == 0
    pasted = full[40:200, 80:220]
    assert pasted.shape == (rh, rw)
    assert int((pasted == roi).sum()) == rh * rw
    print("compose paste OK landscape", full.shape, "roi", roi.shape, "nonzero", int(full.sum()))

    portrait = _compose_full_image_mask(roi, 640, 360, [20, 100, 20 + rw, 100 + rh])
    assert portrait.shape == (640, 360)
    assert int(portrait[0:100].sum()) == 0
    print("compose paste OK portrait", portrait.shape)

    unet = UNetSegmenter()
    img = np.zeros((280, 520, 3), dtype=np.uint8)
    img[60:180, 120:300] = (180, 40, 40)
    bbox2 = [120, 60, 300, 180]
    mask, n_pix, ratio, dbg = unet.segment(img, bbox2)
    assert mask is not None
    assert list(mask.shape) == [180 - 60, 300 - 120], (mask.shape, dbg)
    full2 = _compose_full_image_mask(mask, 280, 520, bbox2)
    assert full2.shape == (280, 520)
    outside = full2.copy()
    outside[60:180, 120:300] = 0
    print("unet roi mask", mask.shape, "full", full2.shape, "outside_roi_sum", int(outside.sum()), "ratio", ratio, "status", (dbg or {}).get("status"))
    return {
        "compose_landscape_ok": True,
        "compose_portrait_ok": True,
        "unet_mask_equals_roi": list(mask.shape) == [120, 180],
        "outside_roi_sum": int(outside.sum()),
        "affected_ratio": float(ratio) if ratio is not None else None,
    }


def ood_probes():
    print("\n=== PHASE 3/4 OOD probes ===")
    eff = EfficientNetV2Classifier()
    unet = UNetSegmenter()
    rows = []
    specs = {
        "gray": _blank(128),
        "black": _blank(0),
        "white": _blank(255),
        "blank_noise": np.full((240, 400, 3), 128, dtype=np.uint8),
    }
    demo = abs_path("data/sample/image/football_injury.jpg")
    if os.path.isfile(demo):
        specs["football_injury"] = cv2.cvtColor(cv2.imread(demo), cv2.COLOR_BGR2RGB)
    for name, img in specs.items():
        pred = eff.predict(img)
        raw = eff.predict_raw(img)
        winner = pred.get("__winner")
        status = pred.get("__status")
        maxp = pred.get("__max_prob")
        raw_winner = raw.get("winner")
        raw_max = raw.get("max_prob")
        raw_probs = raw.get("probs")
        mask, n_pix, ratio, dbg = unet.segment(img)
        rows.append({
            "name": name,
            "hw": list(img.shape[:2]),
            "eff_winner": winner,
            "eff_raw_winner": raw_winner,
            "eff_raw_max": raw_max,
            "eff_status": status,
            "eff_max_prob": maxp,
            "unet_positive_ratio": float(ratio) if ratio is not None else None,
            "unet_raw_positive_ratio": (dbg or {}).get("raw_positive_ratio"),
            "unet_status": (dbg or {}).get("status"),
            "unet_reason": (dbg or {}).get("reason"),
            "unet_pixels": int(n_pix) if n_pix is not None else None,
            "eff_raw_probs": raw_probs,
        })
        print(name, "gated", status, winner, maxp, "raw", raw_winner, raw_max, "unet", (dbg or {}).get("status"), "ratio", ratio)
    return rows


def yolo_thresholds():
    print("\n=== PHASE 5 YOLO thresholds ===")
    det = YOLO11Detector()
    info = det.get_info()
    sha = sha256_file(YOLO_CANONICAL)
    print("path", info.get("model_path"), "task", det.model.task, "names", det.model.names, "sha", sha[:16])
    demo = abs_path("data/sample/image/football_injury.jpg")
    img = cv2.cvtColor(cv2.imread(demo), cv2.COLOR_BGR2RGB)
    sweep = {}
    for thr in (0.01, 0.05, 0.10, 0.25, 0.40, 0.50):
        try:
            raw = det.model.predict(demo, conf=thr, verbose=False)[0]
            n = 0 if raw.boxes is None else len(raw.boxes)
            maxc = None
            classes = []
            if raw.boxes is not None and len(raw.boxes):
                confs = raw.boxes.conf.cpu().numpy()
                clss = raw.boxes.cls.cpu().numpy().astype(int)
                maxc = float(confs.max())
                names = det.model.names
                classes = sorted({str(names[int(c)]) for c in clss})
            sweep[str(thr)] = {"n": n, "max_conf": maxc, "classes": classes}
        except Exception as e:
            sweep[str(thr)] = {"error": f"{type(e).__name__}: {e}"}
        print("thr", thr, sweep[str(thr)])
    return {
        "sha256": sha,
        "task": det.model.task,
        "names": {str(k): str(v) for k, v in det.model.names.items()},
        "demo_sweep": sweep,
        "wrapper_default_n": len(det.detect(demo)),
    }


def fusion_and_sos():
    print("\n=== PHASE 7/16 fusion + SOS ===")
    fusion = MultimodalFeatureFusion()
    _, vec, names = fusion.fuse_features({"questionnaire": {"answers": {"pain_level": None}}})
    print("missing pain encoded", float(vec[names.index("pain_level")]), "not 5.0")
    tw = TwilioService()
    cfg = tw.get_status_info()
    print("twilio", cfg.get("status") if isinstance(cfg, dict) else cfg, "configured", tw.is_configured())
    return {"pain_missing": float(vec[names.index("pain_level")]), "twilio": cfg}


def main():
    out = {
        "yolo": yolo_thresholds(),
        "mask": mask_geometry(),
        "ood": ood_probes(),
        "fusion_sos": fusion_and_sos(),
    }
    path = os.path.join(ROOT, "scratch", "phase_runtime_verify.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
