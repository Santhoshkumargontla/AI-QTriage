"""EfficientNet kaggle-v1: OOD must not confidently predict injury classes.

Application quality gates still withhold blank inputs. Raw probes must show
abstention (normal / ood_reject), not injury-class collapse on blanks.
"""
import json
import os
import tempfile

import cv2
import numpy as np
import pytest

from ml.models.canonical_paths import (
    EFFNET_CANONICAL,
    EFFNET_METADATA,
    ROOT,
    abs_path,
    exists,
    read_json,
    sha256_file,
)
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier, interpret_prediction
from ml.vision.input_quality import (
    STATUS_LOW_QUALITY,
    STATUS_OOD,
    STATUS_UNAVAILABLE,
    STATUS_VALID,
)

ACTIVE_EFFNET_HASH = "95cf385d85419a63eeef8e46de9c9ef7e7487ef0f326fb60ca376b8bae0eec9f"
ACTIVE_CLASSES = [
    "abrasion",
    "bruise",
    "burn",
    "cut",
    "laceration",
    "wound",
    "normal",
    "ood_reject",
]
INJURY = {"abrasion", "bruise", "burn", "cut", "laceration", "wound"}
ABSTAIN = {"normal", "ood_reject", "ood_reject"}


def _clf():
    return EfficientNetV2Classifier()


def _norm_winner(text: str) -> str:
    return str(text or "").lower().replace(" ", "_")


def test_canonical_checkpoint_sha_and_sidecar():
    disk = sha256_file(EFFNET_CANONICAL)
    assert disk == ACTIVE_EFFNET_HASH
    meta = read_json(EFFNET_METADATA)
    assert meta.get("artifact_sha256") == disk
    assert "READY_FOR_RESEARCH_DEMO" in str(meta.get("training_status") or meta.get("status"))
    sidecar = abs_path("ml/models/vision/efficientnetv2_injury_best_classes.json")
    assert exists(sidecar)
    names = json.loads(open(sidecar, encoding="utf-8").read())
    assert names == ACTIVE_CLASSES
    clf = _clf()
    assert clf.is_loaded
    assert clf.classes == ACTIVE_CLASSES
    assert os.path.normpath(clf.model_path) == os.path.normpath(abs_path(EFFNET_CANONICAL))


def test_registry_and_manifest_sha_match_disk():
    disk = sha256_file(EFFNET_CANONICAL)
    registry = read_json("ml/models/model_registry.json")
    entry = registry["EfficientNetV2 Classification"]
    assert entry["artifact_sha256"] == disk
    assert "READY_FOR_RESEARCH_DEMO" in str(entry.get("status"))
    manifest = read_json("ml/models/canonical_manifest.json")
    models = manifest.get("models") or manifest
    hit = None
    if isinstance(models, dict):
        for key, value in models.items():
            if "efficientnet" in str(key).lower() or "efficientnet" in str((value or {}).get("canonical_path", "")).lower():
                hit = value
                break
    elif isinstance(models, list):
        for value in models:
            if "efficientnet" in str((value or {}).get("canonical_path", "")).lower():
                hit = value
                break
    assert hit is not None
    assert (hit.get("sha256") or hit.get("artifact_sha256")) == disk


def test_runtime_path_independent_of_cwd():
    previous = os.getcwd()
    foreign = tempfile.mkdtemp(prefix="effnet_cwd_")
    try:
        for cwd in (os.path.join(ROOT, "backend"), foreign):
            os.chdir(cwd)
            clf = EfficientNetV2Classifier()
            assert clf.is_loaded
            assert sha256_file(clf.model_path) == ACTIVE_EFFNET_HASH
            assert clf.classes == ACTIVE_CLASSES
    finally:
        os.chdir(previous)


def test_blank_gray_is_low_quality_not_injury():
    clf = _clf()
    img = np.full((224, 224, 3), 180, dtype=np.uint8)
    raw = clf.predict_raw(img)
    gated = clf.predict(img)
    parsed = interpret_prediction(gated)
    assert raw["winner"] is not None
    assert _norm_winner(raw["winner"]) not in INJURY
    assert gated["__status"] == STATUS_LOW_QUALITY
    assert parsed["winner"] is None
    assert parsed["is_confident"] is False


@pytest.mark.parametrize(
    "factory",
    [
        lambda: np.zeros((200, 200, 3), dtype=np.uint8),
        lambda: np.full((200, 200, 3), 255, dtype=np.uint8),
        lambda: np.full((200, 200, 3), 64, dtype=np.uint8),
        lambda: np.full((200, 200, 3), 128, dtype=np.uint8),
        lambda: np.full((200, 200, 3), 220, dtype=np.uint8),
        lambda: np.clip(
            np.full((200, 200, 3), 180) + np.random.default_rng(0).normal(0, 5, (200, 200, 3)),
            0,
            255,
        ).astype(np.uint8),
        lambda: np.full((200, 200, 3), (185, 145, 125), dtype=np.uint8),
    ],
)
def test_black_white_gray_levels_raw_abstain_app_withheld(factory):
    clf = _clf()
    img = factory()
    raw = clf.predict_raw(img)
    gated = clf.predict(img)
    parsed = interpret_prediction(gated)
    assert float(raw["max_prob"]) >= 0.40
    assert _norm_winner(raw["winner"]) not in INJURY
    assert gated["__is_confident"] is False
    assert gated["__status"] in {STATUS_LOW_QUALITY, STATUS_OOD, STATUS_VALID}
    assert parsed["is_confident"] is False
    assert parsed["winner"] is None


def test_unrelated_blue_is_ood_or_abstention():
    clf = _clf()
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :, 2] = 200
    img[::6, :, 1] = 180
    out = clf.predict(img)
    parsed = interpret_prediction(out)
    assert out["__is_confident"] is False
    assert parsed["winner"] is None
    raw = clf.predict_raw(img)
    assert _norm_winner(raw["winner"]) not in INJURY


def test_synthetic_cut_template_injury_or_abstention_honest():
    """Drawing-trained cut may fire; abstention is also acceptable. Never claim clinical."""
    clf = _clf()
    img = np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8)
    cv2.line(img, (70, 40), (150, 180), (190, 20, 20), 7)
    out = clf.predict(img)
    parsed = interpret_prediction(out)
    meta = read_json(EFFNET_METADATA)
    assert "READY_FOR_RESEARCH_DEMO" in str(meta.get("training_status") or meta.get("status"))
    if parsed["is_confident"]:
        assert _norm_winner(parsed["winner"]) in INJURY
    else:
        assert parsed["winner"] is None


def test_abstention_normal_is_not_routed_as_injury():
    from backend.main import _routing_finding, _routing_confidence

    blank = {
        "yolo_finding_detected": False,
        "classifier_finding": None,
        "classifier_probability": None,
        "classifier_status": STATUS_LOW_QUALITY,
        "classifier_model_status": "READY_FOR_RESEARCH_DEMO",
        "classifier_is_confident": False,
    }
    assert _routing_finding(blank) == ""
    assert _routing_confidence(blank) == 0.0

    abstain = {
        "yolo_finding_detected": False,
        "classifier_finding": "Normal",
        "classifier_probability": 0.99,
        "classifier_status": STATUS_VALID,
        "classifier_model_status": "READY_FOR_RESEARCH_DEMO",
        "classifier_is_confident": False,
        "classifier_abstention_class": "Normal",
    }
    assert _routing_finding(abstain) == ""
    assert _routing_confidence(abstain) == 0.0


def test_api_models_exposes_effnet_and_unet_sha():
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    models = client.get("/api/models").json()
    by_name = {m["model_name"]: m for m in models}
    assert by_name["EfficientNetV2"]["artifact_sha256"] == ACTIVE_EFFNET_HASH
    assert "READY_FOR_RESEARCH_DEMO" in str(by_name["EfficientNetV2"]["status"])
    assert by_name["EfficientNetV2"]["canonical_path"].replace("\\", "/") == "ml/models/vision/efficientnetv2_injury_best.pt"
    unet = by_name["U-Net"]
    assert unet["artifact_sha256"] == sha256_file("ml/models/vision/unet_injury_best.pt")
    assert unet.get("canonical_path")
    assert unet.get("weights_loaded") is True


def test_missing_weights_status_unavailable(tmp_path):
    clf = EfficientNetV2Classifier(model_path=str(tmp_path / "missing.pt"))
    assert clf.is_loaded is False
    out = clf.predict(np.full((64, 64, 3), 180, dtype=np.uint8))
    assert out["__status"] == STATUS_UNAVAILABLE
    assert out["__winner"] is None
