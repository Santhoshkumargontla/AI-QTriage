"""
AI-QTriage — Real YOLO Upgrade Regression Tests (Phase 8)
Runtime YOLO is a single canonical file. Backups exist on disk but are not loaded
unless an explicit model_path is passed. YOLO_MODEL_VERSION does not change the file.
"""

import os
import pytest
from ml.vision.yolo_wrapper import YOLO11Detector
from ml.models.canonical_paths import YOLO_CANONICAL, YOLO_BACKUP_PATHS, YOLO_SYNTHETIC_BASELINE, exists


def test_synthetic_baseline_preservation():
    """Backup synthetic weights remain on disk. They are not the runtime path."""
    baseline_path = YOLO_SYNTHETIC_BASELINE
    assert exists(baseline_path), "Synthetic baseline weights yolo11n_best.pt missing!"
    assert os.path.normpath(baseline_path) != os.path.normpath(YOLO_CANONICAL)


def test_yolo_environment_does_not_switch_runtime_path(monkeypatch):
    """YOLO_MODEL_VERSION must not silently load a different checkpoint."""
    monkeypatch.setenv("YOLO_MODEL_VERSION", "synthetic_baseline")
    detector = YOLO11Detector()
    info = detector.get_info()
    assert os.path.normpath(info["model_path"]) == os.path.normpath(YOLO_CANONICAL)
    assert "swelling" not in info["supported_classes"]


def test_yolo_environment_real_flag_still_uses_canonical(monkeypatch):
    monkeypatch.setenv("YOLO_MODEL_VERSION", "real_data_experimental")
    detector = YOLO11Detector()
    info = detector.get_info()
    assert os.path.normpath(info["model_path"]) == os.path.normpath(YOLO_CANONICAL)
    assert "wound" not in info["supported_classes"]
    assert "abrasion" in info["supported_classes"]


def test_semantic_separation_swelling_not_in_yolo():
    """Swelling is UNTRAINED_CLASS even on backup checkpoints loaded explicitly."""
    det_runtime = YOLO11Detector()
    assert "swelling" not in det_runtime.supported_classes
    assert det_runtime.class_status("swelling") == "UNTRAINED_CLASS"

    det_synth = YOLO11Detector(model_path=YOLO_SYNTHETIC_BASELINE)
    assert "swelling" not in det_synth.supported_classes


def test_no_invented_detections_on_blank_image():
    """Inference on a blank image produces 0 false positive detections."""
    detector = YOLO11Detector()
    blank_path = os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")
    if os.path.exists(blank_path) and detector.model is not None:
        findings = detector.detect(blank_path)
        assert len(findings) == 0, "Blank control image produced false positive YOLO detection!"


def test_backup_paths_are_not_canonical():
    for path in YOLO_BACKUP_PATHS:
        assert os.path.normpath(path) != os.path.normpath(YOLO_CANONICAL)
