"""Grad-CAM must not run or display when the classifier is not VALID or not trustworthy."""
import numpy as np
import cv2

from ml.explainability.grad_cam import (
    EXPLANATION_GENERATED,
    EXPLANATION_UNAVAILABLE,
    EXPLANATION_WITHHELD,
    GRADCAM_LABEL,
    GRADCAM_NOT_CLINICAL,
    classifier_allows_gradcam,
    maybe_generate_gradcam,
)
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier, interpret_prediction
from ml.vision.input_quality import (
    STATUS_LOW_QUALITY,
    STATUS_OOD,
    STATUS_UNAVAILABLE,
    STATUS_UNTRUSTWORTHY,
    STATUS_VALID,
)


def _clf():
    return EfficientNetV2Classifier()


def _cut_image():
    img = np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8)
    cv2.line(img, (70, 40), (150, 180), (190, 20, 20), 7)
    return img


def test_ready_research_demo_allows_gradcam_for_injury_finding():
    """reject-v2 is READY_FOR_RESEARCH_DEMO — Grad-CAM may run for injury findings only."""
    clf = _clf()
    img = _cut_image()
    parsed = interpret_prediction(clf.predict(img))
    if not parsed.get("is_confident"):
        # Drawing may abstain; Grad-CAM must stay withheld.
        overlay, meta = maybe_generate_gradcam(clf, img, parsed)
        assert overlay is None
        assert meta["explanation_status"] == EXPLANATION_WITHHELD
        return
    assert parsed["status"] == STATUS_VALID
    assert classifier_allows_gradcam(parsed) is True
    overlay, meta = maybe_generate_gradcam(clf, img, parsed)
    assert overlay is not None
    assert meta["explanation_status"] == EXPLANATION_GENERATED
    assert meta["overlay_generated"] is True
    assert meta["gradcam_explanation"] == GRADCAM_NOT_CLINICAL


def test_explicit_not_trustworthy_override_still_withholds_gradcam():
    clf = _clf()
    img = _cut_image()
    parsed = interpret_prediction(clf.predict(img))
    forced = {**parsed, "status": STATUS_VALID, "is_confident": True, "winner": "Cut", "model_training_status": "NOT_TRUSTWORTHY"}
    assert classifier_allows_gradcam(forced) is False
    overlay, meta = maybe_generate_gradcam(clf, img, forced)
    assert overlay is None
    assert meta["explanation_status"] == EXPLANATION_WITHHELD
    assert meta["model_status"] == "NOT_TRUSTWORTHY"
    assert meta["withheld_reason"] == "classifier_model_not_trustworthy"


def test_valid_injury_generates_when_training_status_allows():
    """Mechanism check: Grad-CAM still works if training status is explicitly allowed."""
    clf = _clf()
    img = _cut_image()
    base = interpret_prediction(clf.predict(img))
    parsed = {
        **base,
        "status": STATUS_VALID,
        "is_confident": True,
        "winner": base.get("winner") or "Cut",
        "max_prob": base.get("max_prob") or 0.9,
        "model_training_status": "READY_FOR_RESEARCH_DEMO",
    }
    assert classifier_allows_gradcam(parsed) is True
    overlay, meta = maybe_generate_gradcam(clf, img, parsed)
    assert overlay is not None
    assert overlay.shape[:2] == img.shape[:2]
    assert meta["explanation_status"] == EXPLANATION_GENERATED
    assert meta["overlay_generated"] is True
    assert meta["source_model"] == "EfficientNetV2"
    assert meta["gradcam_label"] == GRADCAM_LABEL
    assert meta["gradcam_explanation"] == GRADCAM_NOT_CLINICAL

def test_blank_image_does_not_generate_gradcam():
    clf = _clf()
    img = np.full((128, 128, 3), 180, dtype=np.uint8)
    parsed = interpret_prediction(clf.predict(img))
    assert parsed["status"] == STATUS_LOW_QUALITY
    overlay, meta = maybe_generate_gradcam(clf, img, parsed)
    assert overlay is None
    assert meta["overlay_generated"] is False
    assert meta["explanation_status"] == EXPLANATION_WITHHELD
    assert meta["predicted_class"] is None
    assert meta["confidence"] is None
    assert meta["gradcam_explanation"] == GRADCAM_NOT_CLINICAL


def test_invalid_and_untrustworthy_inputs_withheld():
    clf = _clf()
    tiny = np.zeros((8, 8, 3), dtype=np.uint8)
    parsed = interpret_prediction(clf.predict(tiny))
    overlay, meta = maybe_generate_gradcam(clf, tiny, parsed)
    assert overlay is None
    assert meta["explanation_status"] == EXPLANATION_WITHHELD
    assert meta["model_status"] in {STATUS_LOW_QUALITY, STATUS_OOD, "NOT_TRUSTWORTHY"}

    ood = np.full((160, 160, 3), (20, 60, 200), dtype=np.uint8)
    ood[::6, :, 1] = 180
    parsed = interpret_prediction(clf.predict(ood))
    overlay, meta = maybe_generate_gradcam(clf, ood, parsed)
    assert overlay is None
    assert meta["explanation_status"] == EXPLANATION_WITHHELD

    fake = {"status": STATUS_UNTRUSTWORTHY, "is_confident": True, "winner": "Cut", "max_prob": 0.99}
    assert classifier_allows_gradcam(fake) is False
    overlay, meta = maybe_generate_gradcam(clf, _cut_image(), fake)
    assert overlay is None
    assert meta["explanation_status"] == EXPLANATION_WITHHELD


def test_unavailable_model_status():
    import tempfile, os
    clf = EfficientNetV2Classifier(model_path=os.path.join(tempfile.gettempdir(), "missing_effnet.pt"))
    assert clf.is_loaded is False
    img = _cut_image()
    parsed = interpret_prediction(clf.predict(img))
    assert parsed["status"] == STATUS_UNAVAILABLE
    overlay, meta = maybe_generate_gradcam(clf, img, parsed)
    assert overlay is None
    assert meta["explanation_status"] == EXPLANATION_UNAVAILABLE
    assert meta["model_status"] == STATUS_UNAVAILABLE
    assert meta["overlay_generated"] is False
    assert meta["gradcam_label"] == GRADCAM_LABEL
    assert meta["gradcam_explanation"] == GRADCAM_NOT_CLINICAL
