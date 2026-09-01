"""
AI-QTriage — YOLO11 Reliability & Taxonomy Regression Tests
Verifies model weight selection, class taxonomy, affected area calculation, and negative image handling.
"""

import os
import hashlib
import pytest
import numpy as np
from ml.vision.yolo_wrapper import YOLO11Detector
from backend.services.first_aid_service import StructuredEvidenceBuilder


ACTIVE_YOLO_HASH = "319a2cbc15d6ced2730060ff6e73baf2968271026611124539ce0b06486a1926"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def test_model_weight_loading_not_coco():
    """Verify that trained weights are loaded instead of COCO baseline weights."""
    detector = YOLO11Detector()
    info = detector.get_info()
    assert detector.model is not None, "YOLO11 model failed to load!"
    assert "yolo11n.pt" not in info["model_path"].lower(), "Accidentally loaded untrained COCO base!"

def test_taxonomy_wound_support():
    """Wound is not in model.names (no honest boxes). Abrasion is trained."""
    detector = YOLO11Detector()
    info = detector.get_info()
    supported = info["supported_classes"]
    assert "cut" in supported
    assert "bruise" in supported
    assert "abrasion" in supported
    assert "wound" not in supported
    assert "swelling" not in supported, "'swelling' must NOT be in YOLO object detection classes!"
    assert detector.class_status("wound") == "UNTRAINED_CLASS"
    assert "cut" in (info.get("validated_classes") or [])
    assert "bruise" in (info.get("validated_classes") or [])
    assert "abrasion" in (info.get("validated_classes") or [])


def test_active_yolo_artifact_hash_task_and_class_names():
    """Verify active YOLO artifact identity and detection taxonomy."""
    detector = YOLO11Detector()
    info = detector.get_info()
    assert os.path.exists(info["model_path"])
    assert _sha256(info["model_path"]) == ACTIVE_YOLO_HASH
    assert detector.model.task == "detect"
    assert detector.model.names == {0: "cut", 1: "bruise", 2: "abrasion"}
    assert info["artifact_sha256"] == ACTIVE_YOLO_HASH
    assert info["classes"] == ["cut", "bruise", "abrasion"]
    assert info["task"] == "detect"
    assert float(info.get("infer_conf") or 0) == 0.25


def test_backup_checkpoint_is_not_runtime_path():
    from ml.models.canonical_paths import YOLO_CANONICAL, abs_path
    active = abs_path(YOLO_CANONICAL)
    backup = active + ".pre_retrain_v2_backup"
    assert os.path.exists(backup)
    assert _sha256(backup) == "6cc84115e4cb85c8b82715211c3935200b815b76efbc95f83855c2cc988dce4f"
    assert _sha256(active) == ACTIVE_YOLO_HASH
    assert _sha256(active) != _sha256(backup)
    detector = YOLO11Detector()
    assert os.path.normpath(detector.model_path).endswith(os.path.normpath(YOLO_CANONICAL))
    assert ".pre_retrain_v2_backup" not in str(detector.model_path)


def test_football_filename_does_not_alter_inference_logic():
    """Production detect() must not branch on football_injury filename."""
    import inspect
    from ml.vision import yolo_wrapper
    src = inspect.getsource(yolo_wrapper.YOLO11Detector.detect)
    assert "football" not in src.lower()
    assert "is_demo" not in src.lower()


def test_blank_and_dummy_true_negatives_clean_at_keep_threshold():
    detector = YOLO11Detector()
    for rel in (
        os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg"),
        os.path.join("data", "datasets", "yolo_injury", "dummy_test.jpg"),
    ):
        assert os.path.exists(rel)
        assert detector.detect(rel) == []


def test_forensic_upload_is_unlabeled_ood_not_a_blank_negative():
    """Case upload 3e0dbd17 is an unlabeled 300x167 photo, not a blank/no-injury negative.

    Historical UI showed no detection because scores stayed below the keep-threshold.
    Raw boxes at conf=0.01 are allowed; they must not be surfaced by detect().
    """
    import cv2

    image_path = os.path.join("data", "uploads", "3e0dbd17-7475-487d-9f10-e7f9d6800238.jpg")
    assert os.path.exists(image_path)
    image = cv2.imread(image_path)
    assert image is not None
    height, width = image.shape[:2]
    assert (width, height) == (300, 167)
    assert float(image.std()) > 20.0, "this upload is a real photo, not a uniform negative"

    detector = YOLO11Detector()
    findings = detector.detect(image_path)
    assert findings == [], f"keep-threshold {detector.infer_conf} surfaced {findings}"

    low_results = detector.model(image_path, conf=0.01, verbose=False)
    low_boxes = list(low_results[0].boxes)
    if low_boxes:
        max_conf = max(float(box.conf[0].item()) for box in low_boxes)
        assert max_conf < detector.infer_conf


def test_first_aid_structured_evidence_preserves_yolo_cut():
    """A valid runtime YOLO cut detection must not be discarded downstream."""
    evidence = StructuredEvidenceBuilder.build_evidence(
        visible_injury={
            "yolo_finding_detected": True,
            "yolo_finding": "Cut",
            "yolo_confidence": 0.6034,
            "bounding_box": [495.21, 277.23, 773.57, 408.21],
            "classifier_finding": "Ood_reject",
        }
    )
    assert evidence["yolo"]["finding_detected"] is True
    assert evidence["yolo"]["finding"] == "Cut"
    assert evidence["yolo"]["confidence"] == 0.6034
    assert "cut" in evidence["yolo"]["supported_classes"]
    assert "abrasion" in evidence["yolo"]["supported_classes"]
    assert "wound" not in evidence["yolo"]["supported_classes"]

def test_affected_area_math_calculation():
    """Verify mathematical calculation of affected area ratio from bounding box coordinates."""
    img_w, img_h = 300, 300
    img_area = img_w * img_h
    bbox = [100.0, 100.0, 200.0, 200.0]  # width 100, height 100 => area 10,000
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    expected_ratio = bbox_area / img_area
    assert abs(expected_ratio - (10000.0 / 90000.0)) < 1e-5
    assert round(expected_ratio, 4) == 0.1111

def test_negative_image_zero_false_positive():
    """Verify that a clean synthetic skin image yields zero false-positive YOLO detections."""
    detector = YOLO11Detector()
    blank_img = np.full((300, 300, 3), 200, dtype=np.uint8)
    blank_path = os.path.join("data", "test_suite", "neg_clean_skin_unit_test.jpg")
    os.makedirs(os.path.dirname(blank_path), exist_ok=True)
    import cv2
    cv2.imwrite(blank_path, blank_img)

    findings = detector.detect(blank_path)
    assert len(findings) == 0, f"Clean skin image produced false positive YOLO detection: {findings}"


def test_existing_empty_label_negatives_empty_at_keep_threshold():
    """blank_skin and dummy_test are the labeled negatives used in retrain_v2."""
    detector = YOLO11Detector()
    for rel in (
        os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg"),
        os.path.join("data", "datasets", "yolo_injury", "dummy_test.jpg"),
    ):
        assert os.path.exists(rel)
        findings = detector.detect(rel)
        assert findings == [], f"{rel} produced {findings}"


def test_default_infer_conf_is_research_demo_not_zero_ten(monkeypatch):
    """Default keep-threshold is 0.25 from the sweep, not the old hardcoded 0.10."""
    monkeypatch.delenv("YOLO_CONF_THRESHOLD", raising=False)
    monkeypatch.delenv("YOLO_CONF_THRESHOLD_CUT", raising=False)
    monkeypatch.delenv("YOLO_CONF_THRESHOLD_BRUISE", raising=False)
    monkeypatch.delenv("YOLO_CONF_THRESHOLD_WOUND", raising=False)
    from ml.vision.yolo_wrapper import DEFAULT_YOLO_INFER_CONF, YOLO11Detector, resolve_yolo_infer_conf
    assert DEFAULT_YOLO_INFER_CONF == 0.25
    assert resolve_yolo_infer_conf() == 0.25
    detector = YOLO11Detector()
    info = detector.get_info()
    assert detector.infer_conf == 0.25
    assert info["infer_conf"] == 0.25
    assert info["recommended_conservative_threshold"] == 0.30


def test_env_yolo_conf_threshold_overrides_default(monkeypatch):
    monkeypatch.setenv("YOLO_CONF_THRESHOLD", "0.30")
    from ml.vision.yolo_wrapper import YOLO11Detector, resolve_yolo_infer_conf
    assert resolve_yolo_infer_conf() == 0.30
    detector = YOLO11Detector()
    assert detector.infer_conf == 0.30


def test_class_specific_wound_threshold_filters_below_keep(monkeypatch):
    monkeypatch.setenv("YOLO_CONF_THRESHOLD", "0.25")
    monkeypatch.setenv("YOLO_CONF_THRESHOLD_WOUND", "0.30")
    import torch
    from unittest.mock import MagicMock, patch
    from ml.vision.yolo_wrapper import YOLO11Detector

    with patch("ml.vision.yolo_wrapper.YOLO") as mock_yolo_cls:
        mock_yolo = MagicMock()
        mock_yolo.task = "detect"
        mock_yolo.names = {0: "cut", 1: "bruise", 2: "wound"}
        mock_yolo_cls.return_value = mock_yolo

        box_wound = MagicMock()
        box_wound.cls = torch.tensor([2.0])
        box_wound.conf = torch.tensor([0.28])
        box_wound.xyxy = [torch.tensor([1.0, 1.0, 10.0, 10.0])]

        box_bruise = MagicMock()
        box_bruise.cls = torch.tensor([1.0])
        box_bruise.conf = torch.tensor([0.90])
        box_bruise.xyxy = [torch.tensor([2.0, 2.0, 20.0, 20.0])]

        mock_results = MagicMock()
        mock_results.boxes = [box_wound, box_bruise]
        mock_results.orig_shape = (64, 64)
        mock_yolo.return_value = [mock_results]

        detector = YOLO11Detector()
        detector.model = mock_yolo
        detector._sync_classes_from_model()
        findings = detector.detect("dummy.png")
        names = [f["finding"] for f in findings]
        assert names == ["bruise"]
        assert findings[0]["keep_threshold"] == 0.25

