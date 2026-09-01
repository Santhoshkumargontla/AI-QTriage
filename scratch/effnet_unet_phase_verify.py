"""Independent EffNet + U-Net verify-first probe. Read-only."""
from __future__ import annotations

import csv
import json
import os

import cv2
import numpy as np

from ml.explainability.grad_cam import maybe_generate_gradcam
from ml.models.canonical_paths import ROOT, abs_path
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier, interpret_prediction
from ml.vision.unet_wrapper import UNetSegmenter, interpret_segmentation

OUT = abs_path("scratch/effnet_unet_phase_verify.json")


def main():
    clf = EfficientNetV2Classifier()
    unet = UNetSegmenter()

    def eff_probe(name, rgb, provenance):
        raw = clf.predict_raw(rgb)
        gated = clf.predict(rgb)
        parsed = interpret_prediction(gated)
        return {
            "name": name,
            "prov": provenance,
            "shape": list(rgb.shape),
            "mean": round(float(rgb.mean()), 2),
            "std": round(float(rgb.std()), 2),
            "raw_winner": raw["winner"],
            "raw_max": round(float(raw["max_prob"]), 4),
            "raw_probs": {k: round(float(v), 4) for k, v in raw["probs"].items()},
            "raw_margin": round(float(raw["margin"]), 4),
            "raw_entropy": round(float(raw["entropy"]), 4),
            "gate_status": gated.get("__status"),
            "gate_reason": gated.get("__reason"),
            "api_winner": parsed.get("winner"),
            "api_conf": parsed.get("max_prob"),
            "api_confident": parsed.get("is_confident"),
        }

    def unet_probe(name, rgb, provenance, bbox=None):
        raw = unet.segment_raw(rgb, bbox)
        mask, pc, ar, info = unet.segment(rgb, bbox)
        parsed = interpret_segmentation(mask, pc, ar, info)
        m = np.asarray(mask) if mask is not None else np.zeros(rgb.shape[:2], np.uint8)
        nlab = cv2.connectedComponents((m > 0).astype(np.uint8), 8)[0]
        return {
            "name": name,
            "prov": provenance,
            "shape": list(rgb.shape),
            "mean": round(float(rgb.mean()), 2),
            "std": round(float(rgb.std()), 2),
            "raw_min": None if raw.get("min_prob") is None else round(float(raw["min_prob"]), 4),
            "raw_max": None if raw.get("max_prob") is None else round(float(raw["max_prob"]), 4),
            "raw_mean": None if raw.get("mean_prob") is None else round(float(raw["mean_prob"]), 4),
            "raw_pos_ratio": None if raw.get("positive_ratio") is None else round(float(raw["positive_ratio"]), 6),
            "disp_pos_px": int((m > 0).sum()),
            "disp_pos_ratio": round(float((m > 0).mean()), 6),
            "cc": int(max(nlab - 1, 0)),
            "status": info.get("status"),
            "reason": info.get("reason"),
            "withheld": info.get("mask_withheld"),
            "reliable": info.get("is_reliable"),
            "info_raw_pos_ratio": info.get("raw_positive_ratio"),
            "interp_reliable": parsed.get("is_reliable"),
            "interp_withheld": parsed.get("mask_withheld"),
            "interp_px": parsed.get("pixel_count"),
        }

    imgs = []
    for n, v in [("black", 0), ("white", 255), ("gray_128", 128), ("gray_180", 180)]:
        imgs.append((n, np.full((256, 256, 3), v, np.uint8), "blank_control"))
    imgs.append(("uniform_skin", np.full((256, 256, 3), (185, 145, 125), np.uint8), "blank_control"))
    rng = np.random.default_rng(1)
    imgs.append(("noise", rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), "synthetic_ood"))
    blur = cv2.GaussianBlur(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), (31, 31), 0)
    imgs.append(("blur_lowtex", blur, "low_texture"))
    for tag, rel, prov in [
        ("dummy", "data/datasets/yolo_injury/dummy_test.jpg", "synthetic_tiny"),
        ("blank_skin", "data/datasets/yolo_injury/blank_skin.jpg", "synthetic_blank_skin"),
        ("football", "data/sample/image/football_injury.jpg", "demo_unlabeled"),
        ("forensic_ood", "data/debug/yolo/forensic_non_injury.png", "unlabeled_ood"),
    ]:
        p = abs_path(rel)
        if os.path.exists(p):
            bgr = cv2.imread(p)
            if bgr is not None:
                imgs.append((tag, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), prov))

    for man_rel, prefix in [
        ("data/datasets/efficientnet_processed/manifest.csv", "eff"),
        ("data/datasets/unet_public_real/manifest.csv", "unetpub"),
        ("data/datasets/unet_deduped_subject/manifest.csv", "unetd"),
    ]:
        mp = abs_path(man_rel)
        if not os.path.exists(mp):
            continue
        rows = list(csv.DictReader(open(mp, encoding="utf-8")))
        picked = False
        for r in rows:
            if r.get("split") != "test":
                continue
            # Prefer non-empty / injury classes for positive probe
            cls = (r.get("class") or r.get("label") or "").lower()
            if prefix.startswith("unet") and cls in {"empty", "normal", "background"}:
                continue
            ip = (r.get("image_path") or r.get("image") or "").replace("/", os.sep)
            if not os.path.isabs(ip):
                ip = os.path.join(ROOT, ip)
            if not os.path.exists(ip):
                continue
            bgr = cv2.imread(ip)
            if bgr is None:
                continue
            imgs.append((f"{prefix}_{cls or 'pos'}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "dataset_test"))
            picked = True
            break
        if not picked:
            for r in rows:
                ip = (r.get("image_path") or r.get("image") or "").replace("/", os.sep)
                if not os.path.isabs(ip):
                    ip = os.path.join(ROOT, ip)
                if os.path.exists(ip):
                    bgr = cv2.imread(ip)
                    if bgr is not None:
                        imgs.append((f"{prefix}_any", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "dataset_any"))
                        break

    # Geometry: square / portrait / landscape with synthetic ROI
    geometry = []
    for gname, shape in [("square", (320, 320)), ("portrait", (480, 320)), ("landscape", (320, 480))]:
        h, w = shape
        img = np.full((h, w, 3), (190, 160, 140), np.uint8)
        # paint a red injury blob inside ROI
        x1, y1, x2, y2 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
        cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 3, (y2 - y1) // 3), 0, 0, 360, (180, 30, 30), -1)
        bbox = [x1, y1, x2, y2]
        mask_roi, pc, ar, info = unet.segment(img, bbox)
        # paste into full image like backend
        full = np.zeros((h, w), dtype=np.uint8)
        if mask_roi is not None and pc > 0:
            mh, mw = mask_roi.shape[:2]
            rh, rw = y2 - y1, x2 - x1
            if (mh, mw) != (rh, rw):
                mask_roi = cv2.resize(mask_roi, (rw, rh), interpolation=cv2.INTER_NEAREST)
            full[y1:y2, x1:x2] = (mask_roi > 0).astype(np.uint8)
        outside = int(full.sum()) - int(full[y1:y2, x1:x2].sum())
        inside = int(full[y1:y2, x1:x2].sum())
        geometry.append({
            "name": gname,
            "orig": [h, w],
            "bbox": bbox,
            "roi_mask_shape": list(np.asarray(mask_roi).shape) if mask_roi is not None else None,
            "full_mask_shape": list(full.shape),
            "pos_inside_roi": inside,
            "pos_outside_roi": outside,
            "status": info.get("status"),
            "withheld": info.get("mask_withheld"),
            "raw_pos_ratio": info.get("raw_positive_ratio"),
            "reliable": info.get("is_reliable"),
            "display_px": int(pc or 0),
        })

    # Public wound geometry if available
    man = abs_path("data/datasets/unet_public_real/manifest.csv")
    if os.path.exists(man):
        for r in csv.DictReader(open(man, encoding="utf-8")):
            if r.get("split") != "test":
                continue
            ip = (r.get("image_path") or "").replace("/", os.sep)
            if not os.path.isabs(ip):
                ip = os.path.join(ROOT, ip)
            if not os.path.exists(ip):
                continue
            bgr = cv2.imread(ip)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            # full-image segment (no bbox)
            mask, pc, ar, info = unet.segment(rgb)
            m = np.asarray(mask)
            geometry.append({
                "name": "public_test_full",
                "orig": [h, w],
                "bbox": None,
                "roi_mask_shape": list(m.shape),
                "full_mask_shape": list(m.shape),
                "pos_inside_roi": int((m > 0).sum()),
                "pos_outside_roi": 0,
                "status": info.get("status"),
                "withheld": info.get("mask_withheld"),
                "raw_pos_ratio": info.get("raw_positive_ratio"),
                "reliable": info.get("is_reliable"),
                "display_px": int(pc or 0),
            })
            # with a central ROI
            bbox = [w // 5, h // 5, 4 * w // 5, 4 * h // 5]
            mask_roi, pc, ar, info = unet.segment(rgb, bbox)
            full = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            if mask_roi is not None and int(pc or 0) > 0:
                mr = np.asarray(mask_roi)
                rh, rw = y2 - y1, x2 - x1
                if mr.shape[:2] != (rh, rw):
                    mr = cv2.resize(mr, (rw, rh), interpolation=cv2.INTER_NEAREST)
                full[y1:y2, x1:x2] = (mr > 0).astype(np.uint8)
            geometry.append({
                "name": "public_test_roi",
                "orig": [h, w],
                "bbox": bbox,
                "roi_mask_shape": list(np.asarray(mask_roi).shape) if mask_roi is not None else None,
                "full_mask_shape": list(full.shape),
                "pos_inside_roi": int(full[y1:y2, x1:x2].sum()),
                "pos_outside_roi": int(full.sum()) - int(full[y1:y2, x1:x2].sum()),
                "status": info.get("status"),
                "withheld": info.get("mask_withheld"),
                "raw_pos_ratio": info.get("raw_positive_ratio"),
                "reliable": info.get("is_reliable"),
                "display_px": int(pc or 0),
            })
            break

    fp = abs_path("data/sample/image/football_injury.jpg")
    rgb = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    parsed = interpret_prediction(clf.predict(rgb))
    ov, meta = maybe_generate_gradcam(clf, rgb, parsed)

    payload = {
        "effnet": [eff_probe(n, im, p) for n, im, p in imgs],
        "unet": [unet_probe(n, im, p) for n, im, p in imgs],
        "geometry": geometry,
        "gradcam_football": meta,
        "gradcam_overlay": ov is not None,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("WROTE", OUT)
    # compact summary
    print("EFFNET_SUMMARY")
    for row in payload["effnet"]:
        print(f"  {row['name']}: raw={row['raw_winner']}@{row['raw_max']} gate={row['gate_status']} api={row['api_winner']}")
    print("UNET_SUMMARY")
    for row in payload["unet"]:
        print(
            f"  {row['name']}: raw_pos={row['raw_pos_ratio']} raw_mean={row['raw_mean']} "
            f"disp={row['disp_pos_px']} withheld={row['withheld']} status={row['status']}"
        )
    print("GEOMETRY")
    for row in payload["geometry"]:
        print(
            f"  {row['name']}: outside={row['pos_outside_roi']} inside={row['pos_inside_roi']} "
            f"withheld={row['withheld']} status={row['status']}"
        )
    print("GRADCAM", meta.get("explanation_status"), meta.get("withheld_reason"), ov is not None)


if __name__ == "__main__":
    main()
