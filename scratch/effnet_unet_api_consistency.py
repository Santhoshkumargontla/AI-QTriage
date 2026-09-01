"""Clean-state TestClient: direct model vs API vs stored case for EffNet/U-Net."""
from __future__ import annotations

import csv
import json
import os
import sys

import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.main import app
from ml.explainability.grad_cam import maybe_generate_gradcam
from ml.models.canonical_paths import (
    EFFNET_CANONICAL,
    UNET_CANONICAL,
    abs_path,
    sha256_file,
)
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier, interpret_prediction
from ml.vision.unet_wrapper import UNetSegmenter, interpret_segmentation

OUT = abs_path("scratch/effnet_unet_api_consistency.json")


def _vis(case):
    return case.get("visible_injury") or {}


def analyze_image(client, path, label):
    cr = client.post("/api/cases", json={"notes": f"effnet-unet-verify-{label}"})
    assert cr.status_code in (200, 201), cr.text
    cid = cr.json()["case_id"]
    with open(path, "rb") as handle:
        up = client.post(
            f"/api/cases/{cid}/image",
            files={"file": (os.path.basename(path), handle, "image/jpeg")},
        )
    assert up.status_code in (200, 201), up.text
    skip = client.post(f"/api/cases/{cid}/sensor/skip")
    assert skip.status_code in (200, 201), skip.text
    q = client.post(
        f"/api/cases/{cid}/questionnaire",
        json={"answers": {"pain_level": 3, "swelling": "no", "crack_pop": "no"}},
    )
    assert q.status_code in (200, 201), q.text
    a = client.post(f"/api/cases/{cid}/analyze")
    assert a.status_code == 200, a.text[:800]
    g = client.get(f"/api/cases/{cid}")
    assert g.status_code == 200
    return cid, _vis(a.json()), _vis(g.json())


def main():
    client = TestClient(app)
    models = client.get("/api/models").json()
    eff = next(m for m in models if "EfficientNet" in m.get("model_name", ""))
    une = next(
        m for m in models if "U-Net" in m.get("model_name", "") or "UNet" in m.get("model_name", "")
    )

    clf = EfficientNetV2Classifier()
    unet = UNetSegmenter()

    football = abs_path("data/sample/image/football_injury.jpg")
    blank = abs_path("data/datasets/yolo_injury/blank_skin.jpg")
    public_img = None
    man = abs_path("data/datasets/unet_public_real/manifest.csv")
    if os.path.exists(man):
        for row in csv.DictReader(open(man, encoding="utf-8")):
            if row.get("split") != "test":
                continue
            ip = (row.get("image_path") or "").replace("/", os.sep)
            if not os.path.isabs(ip):
                ip = os.path.join(ROOT, ip)
            if os.path.exists(ip):
                public_img = ip
                break

    gray = np.full((224, 224, 3), 180, np.uint8)
    mask_g, pc_g, ar_g, info_g = unet.segment(gray)
    direct_gray = {
        "note": "upload_rejects_pure_uniform_gray_variance_0",
        "effnet_raw_winner": clf.predict_raw(gray).get("winner"),
        "effnet_raw_max": clf.predict_raw(gray).get("max_prob"),
        "effnet_gated": interpret_prediction(clf.predict(gray)),
        "unet_raw_pos": unet.segment_raw(gray).get("positive_ratio"),
        "unet_display": interpret_segmentation(mask_g, pc_g, ar_g, info_g),
    }

    rows = {}
    candidates = [("football", football), ("blank_skin", blank)]
    if public_img:
        candidates.append(("public_wound", public_img))

    for label, path in candidates:
        if not path or not os.path.exists(path):
            continue
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        raw_e = clf.predict_raw(rgb)
        gate_e = interpret_prediction(clf.predict(rgb))
        raw_u = unet.segment_raw(rgb)
        mask, pc, ar, info = unet.segment(rgb)
        parsed_u = interpret_segmentation(mask, pc, ar, info)
        ov, cam = maybe_generate_gradcam(clf, rgb, gate_e)
        cid, api_vis, stored_vis = analyze_image(client, path, label)
        rows[label] = {
            "case_id": cid,
            "path": path,
            "direct_effnet_raw": {"winner": raw_e.get("winner"), "max": raw_e.get("max_prob")},
            "direct_effnet_gated": gate_e,
            "direct_unet_raw_pos": raw_u.get("positive_ratio"),
            "direct_unet_raw_mean": raw_u.get("mean_prob"),
            "direct_unet_display": {
                "reliable": parsed_u.get("is_reliable"),
                "withheld": parsed_u.get("mask_withheld"),
                "status": parsed_u.get("status"),
                "reason": parsed_u.get("reason"),
                "px": parsed_u.get("pixel_count"),
            },
            "gradcam": {
                "overlay": ov is not None,
                "status": cam.get("explanation_status"),
                "reason": cam.get("withheld_reason"),
                "model_status": cam.get("model_status"),
            },
            "api": {
                "classifier_finding": api_vis.get("classifier_finding"),
                "classifier_probability": api_vis.get("classifier_probability"),
                "classifier_status": api_vis.get("classifier_status"),
                "classifier_model_status": api_vis.get("classifier_model_status"),
                "classifier_is_confident": api_vis.get("classifier_is_confident"),
                "classifier_reason": api_vis.get("classifier_reason"),
                "segmentation_reliable": api_vis.get("segmentation_reliable"),
                "segmentation_available": api_vis.get("segmentation_available"),
                "segmentation_trust": api_vis.get("segmentation_trust"),
                "segmentation_message": api_vis.get("segmentation_message"),
                "gradcam_overlay_generated": api_vis.get("gradcam_overlay_generated"),
                "gradcam_explanation_status": api_vis.get("gradcam_explanation_status"),
                "gradcam_withheld_reason": api_vis.get("gradcam_withheld_reason"),
                "gradcam_model_status": api_vis.get("gradcam_model_status"),
                "affected_ratio": api_vis.get("affected_ratio"),
            },
            "stored": {
                "classifier_finding": stored_vis.get("classifier_finding"),
                "classifier_status": stored_vis.get("classifier_status"),
                "classifier_model_status": stored_vis.get("classifier_model_status"),
                "classifier_is_confident": stored_vis.get("classifier_is_confident"),
                "segmentation_reliable": stored_vis.get("segmentation_reliable"),
                "gradcam_overlay_generated": stored_vis.get("gradcam_overlay_generated"),
                "gradcam_explanation_status": stored_vis.get("gradcam_explanation_status"),
            },
            "consistency": {
                "effnet_winner_match": gate_e.get("winner") == api_vis.get("classifier_finding"),
                "effnet_status_match": gate_e.get("status") == api_vis.get("classifier_status"),
                "unet_reliable_match": parsed_u.get("is_reliable")
                == bool(api_vis.get("segmentation_reliable")),
                "api_equals_stored_classifier": api_vis.get("classifier_finding")
                == stored_vis.get("classifier_finding"),
                "api_equals_stored_seg": api_vis.get("segmentation_reliable")
                == stored_vis.get("segmentation_reliable"),
                "gradcam_api_withheld": api_vis.get("gradcam_overlay_generated") is False,
            },
        }

    payload = {
        "effnet_sha": sha256_file(EFFNET_CANONICAL),
        "unet_sha": sha256_file(UNET_CANONICAL),
        "api_effnet": {
            "name": eff.get("model_name"),
            "status": eff.get("status") or eff.get("training_status"),
            "sha": eff.get("artifact_sha256"),
            "path": eff.get("model_path") or eff.get("canonical_path"),
        },
        "api_unet": {
            "name": une.get("model_name"),
            "status": une.get("status") or une.get("training_status"),
            "sha": une.get("artifact_sha256"),
            "path": une.get("model_path") or une.get("canonical_path"),
        },
        "direct_gray_upload_blocked": direct_gray,
        "cases": rows,
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print("WROTE", OUT)
    print(json.dumps({k: v.get("consistency") for k, v in rows.items()}, indent=2))
    for key, value in rows.items():
        print(
            key,
            "clf",
            value["api"]["classifier_finding"],
            value["api"]["classifier_status"],
            value["api"]["classifier_model_status"],
            "seg",
            value["api"]["segmentation_reliable"],
            "cam",
            value["api"]["gradcam_explanation_status"],
        )


if __name__ == "__main__":
    main()
