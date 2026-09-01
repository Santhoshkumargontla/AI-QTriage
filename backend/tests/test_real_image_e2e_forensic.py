"""Regression tests from the 2026-08-29 real-image end-to-end forensic audit.

These tests lock pipeline honesty — not a desired class label for one photo.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import (
    _apply_vision_image_provenance,
    _gradcam_payload,
    app,
)
from ml.models.canonical_paths import YOLO_CANONICAL, abs_path, sha256_file


client = TestClient(app)


def _rgb_png_bytes(size=(320, 240)) -> bytes:
    """Non-uniform pixels so upload quality gates (blur/contrast) accept the file."""
    import random

    img = Image.new("RGB", size)
    pixels = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = (
                40 + (x * 3 + y * 5 + random.randint(0, 40)) % 180,
                30 + (x * 2 + y * 7 + random.randint(0, 40)) % 160,
                20 + (x + y * 4 + random.randint(0, 40)) % 140,
            )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_yolo_class_index_order_matches_checkpoint_names():
    """cut=0, bruise=1, wound=2 — no frontend remapping, no index swap."""
    from ml.vision.yolo_wrapper import YOLO11Detector

    det = YOLO11Detector()
    assert det.model is not None
    names = det.model.names
    ordered = [str(names[i]).lower() for i in sorted(int(k) for k in names.keys())]
    assert ordered == ["cut", "bruise", "abrasion"]
    assert det.class_list == ordered
    assert sha256_file(YOLO_CANONICAL) == (
        "319a2cbc15d6ced2730060ff6e73baf2968271026611124539ce0b06486a1926"
    )


def test_apply_vision_provenance_keeps_upload_when_sensor_is_demo():
    vision = {"yolo_finding": "Bruise"}
    case = {"is_demo": False}
    _apply_vision_image_provenance(vision, case, sensor_is_demo=True)
    assert vision["source_type"] == "uploaded"
    assert vision["data_provenance"] == "user_provided"
    assert "Synthetic demonstration image" not in vision["display_message"]
    assert vision["sensor_data_is_demo"] is True


def test_apply_vision_provenance_demo_case_keeps_synthetic_label():
    vision = {"yolo_finding": "Cut"}
    _apply_vision_image_provenance(vision, {"is_demo": True}, sensor_is_demo=True)
    assert vision["source_type"] == "demo"
    assert vision["data_provenance"] == "synthetic"
    assert "Synthetic demonstration image" in vision["display_message"]


def test_analyze_does_not_overwrite_upload_vision_as_synthetic_demo():
    import backend.main as main

    src = inspect.getsource(main.analyze_case)
    assert "_apply_vision_image_provenance" in src
    # Blind overwrite removed from analyze_case; only the helper may set demo labels.
    assert 'vision_results["source_type"] = "demo"' not in src
    helper = inspect.getsource(main._apply_vision_image_provenance)
    assert 'vision_results["source_type"] = "demo"' in helper
    assert 'vision_results["source_type"] = "uploaded"' in helper


def test_no_filename_or_hash_special_case_for_forensic_case():
    import backend.main as main

    blob = inspect.getsource(main)
    assert "ac69884f-7a50-48e5-b7cc-a9bea9b20313" not in blob
    assert "87a76147983d0cdb3a63c9f3d3988b0e16ba8157085513f63463de26a559b446" not in blob


def test_upload_persists_image_sha256_and_path_identity(tmp_path):
    create = client.post("/api/cases", json={})
    assert create.status_code in (200, 201)
    case_id = create.json()["case_id"]
    payload = _rgb_png_bytes()
    expected = hashlib.sha256(payload).hexdigest()
    up = client.post(
        f"/api/cases/{case_id}/image",
        files={"file": ("hand.png", payload, "image/png")},
    )
    assert up.status_code == 200, up.text
    body = up.json()
    assert body["image_sha256"] == expected
    from backend.database.connection import get_database

    stored = get_database().cases.find_one({"case_id": case_id})
    assert stored["image_sha256"] == expected
    assert os.path.isfile(stored["image_reference"])
    with open(stored["image_reference"], "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == expected


def test_gradcam_payload_withholds_when_classifier_not_trustworthy(monkeypatch):
    import numpy as np
    from ml.explainability import grad_cam as gc

    class _FakeClf:
        pass

    def _fake_maybe(classifier, image_rgb, parsed):
        return None, {
            "overlay_generated": False,
            "source_model": "EfficientNetV2",
            "predicted_class": None,
            "confidence": None,
            "model_status": "NOT_TRUSTWORTHY",
            "explanation_status": "WITHHELD",
            "gradcam_label": "MODEL VISUALIZATION",
            "gradcam_explanation": "NOT CLINICAL EXPLANATION",
            "gradcam_reliability": "NOT_CLINICAL_EXPLANATION",
            "withheld_reason": "classifier_model_not_trustworthy",
        }

    monkeypatch.setattr(gc, "maybe_generate_gradcam", _fake_maybe)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    out = _gradcam_payload(_FakeClf(), rgb, {"status": "VALID"}, "test-case-overlay", 32, 32)
    assert out["gradcam_explanation_status"] == "WITHHELD"
    assert out["gradcam_overlay_generated"] is False
    assert out["overlay_url"] is None


def test_frontend_does_not_hardcode_yolo_class_relabel():
    page = abs_path(os.path.join("frontend", "app", "cases", "[id]", "page.tsx"))
    text = open(page, encoding="utf-8").read()
    # Display uses API yolo_finding; must not remap bruise→cut locally.
    assert "yolo_finding" in text
    assert 'replace("bruise", "cut")' not in text.lower()
    assert "bruiseToCut" not in text
    assert "forceCut" not in text


@pytest.mark.skipif(
    not os.path.isfile(
        abs_path(
            os.path.join(
                "data",
                "uploads",
                "3f629ca8-dd98-427d-a708-f976e2042555.jpeg",
            )
        )
    ),
    reason="forensic hand fixture not present",
)
def test_forensic_hand_cut_kept_as_cut_not_wrist_bruise():
    """Promoted Roboflow-v1 must keep a cut on the injury region at 0.25, not a wrist bruise."""
    from ml.vision.yolo_wrapper import YOLO11Detector, DEFAULT_YOLO_INFER_CONF

    path = abs_path(os.path.join("data", "uploads", "3f629ca8-dd98-427d-a708-f976e2042555.jpeg"))
    if not os.path.isfile(path):
        pytest.skip("forensic hand fixture missing")
    det = YOLO11Detector()
    kept = det.detect(path)
    assert DEFAULT_YOLO_INFER_CONF == 0.25
    assert kept, "expected at least one kept detection"
    top = max(kept, key=lambda d: float(d["confidence"]))
    assert top["finding"].lower() == "cut"
    assert float(top["confidence"]) >= 0.25
    box = top["bounding_box"]
    # Injury region is mid-hand, not the far-left wrist box of the old synthetic model.
    assert box[0] > 400
    assert not any(d["finding"].lower() == "bruise" and d["bounding_box"][0] < 50 for d in kept)
