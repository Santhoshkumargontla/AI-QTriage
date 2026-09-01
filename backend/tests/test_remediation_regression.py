"""Regression tests for forensic remediation: no fake fallbacks, canonical YOLO, 23-d XGB."""
import os
import inspect
import hashlib
import numpy as np
import pytest

from ml.vision.yolo_wrapper import YOLO11Detector
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.classifiers.vqc_classifier import VQCClassifier
from ml.classifiers.sensor_classifier import SensorClassifier
from ml.models.canonical_paths import YOLO_CANONICAL, XGB_CANONICAL, exists, sha256_file
from backend.services.twilio_service import TwilioService


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_canonical_yolo_path_and_detect_task():
    det = YOLO11Detector()
    info = det.get_info()
    assert det.model is not None
    assert os.path.normpath(info["model_path"]) == os.path.normpath(YOLO_CANONICAL)
    assert det.model.task == "detect"
    assert set(info["supported_classes"]) == {str(v).lower() for v in det.model.names.values()}
    assert "swelling" not in info["supported_classes"]
    assert "abrasion" not in info["supported_classes"] or "abrasion" in set(det.model.names.values())


def test_yolo_classes_equal_model_names():
    det = YOLO11Detector()
    named = {str(v).lower() for v in det.model.names.values()}
    assert det.supported_classes == named


def test_xgboost_exactly_23_features():
    clf = XGBoostClassifier(XGB_CANONICAL)
    assert clf.is_trained
    assert int(clf.model.n_features_in_) == 23
    assert len(clf.feature_names) == 23
    vec = np.zeros(23, dtype=np.float32)
    idx, probs = clf.predict(vec)
    assert idx in (0, 1, 2)
    assert len(probs) == 3
    with pytest.raises(ValueError):
        clf.predict(np.zeros(22, dtype=np.float32))


def test_analyze_source_has_no_random_autotrain():
    import backend.main as main
    src = inspect.getsource(main.analyze_case)
    assert "np.random.randn" not in src
    assert "X_dummy" not in src
    assert "generate_multimodal_dataset" not in src
    assert "MODEL_ARTIFACT_MISSING" in inspect.getsource(main.require_model_artifacts)
    assert "XGB_CANONICAL" in src
    require_src = inspect.getsource(main.require_model_artifacts)
    assert "vqc_weights" not in require_src


def test_vqc_predict_source_has_no_hardcoded_fallback():
    src = inspect.getsource(VQCClassifier.predict)
    assert "[0.15, 0.70, 0.15]" not in src
    assert "except Exception:" not in src


def test_sensor_classifier_missing_features_are_explicit():
    clf = SensorClassifier()
    out = clf.predict_from_summary({})
    assert out["status"] in ("FEATURE_MISSING", "MODEL_UNAVAILABLE")
    assert out["predicted_motion_class"] is None
    assert out.get("probabilities") in (None, {})


def test_twilio_canonical_env_names():
    src = inspect.getsource(TwilioService.reload_config)
    assert "TWILIO_FROM_NUMBER" in src
    assert "TWILIO_TO_NUMBER" in src
    svc = TwilioService()
    configured, msg = svc.is_configured()
    if not configured:
        assert "disabled" in msg.lower() or "missing" in msg.lower() or "not" in msg.lower()


def test_unet_blank_image_withheld():
    from ml.vision.unet_wrapper import UNetSegmenter
    from ml.vision.input_quality import STATUS_LOW_QUALITY
    import numpy as np
    seg = UNetSegmenter()
    gray = np.full((64, 64, 3), 180, dtype=np.uint8)
    mask, count, ratio, info = seg.segment(gray)
    assert int(mask.sum()) == 0
    assert count == 0
    assert info["status"] == STATUS_LOW_QUALITY
    assert info["mask_withheld"] is True
    assert info["is_reliable"] is False
    # Raw collapse must be recorded, not zeroed away.
    assert info["raw_positive_ratio"] is not None



def test_efficientnet_and_unet_trainers_actually_optimize():
    from ml.training import train_efficientnet as te
    from ml.training import train_unet as tu
    assert "loss.backward()" in inspect.getsource(te.train_efficientnet)
    assert "opt.step()" in inspect.getsource(te.train_efficientnet)
    assert "loss.backward()" in inspect.getsource(tu.train_unet)
    assert "opt.step()" in inspect.getsource(tu.train_unet)
    assert "Train split has no empty-mask negatives." not in inspect.getsource(tu)
    from ml.training import prepare_unet_processed_dataset as prep_u
    assert "_make_empty" not in inspect.getsource(prep_u)


def test_registry_canonical_yolo_sha_matches_disk():
    import json
    from ml.models.canonical_paths import REGISTRY_PATH
    assert exists(YOLO_CANONICAL)
    disk = sha256_file(YOLO_CANONICAL)
    if exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            reg = json.load(f)
        yolo = reg.get("YOLO11 Detection") or {}
        if yolo.get("artifact_path"):
            assert os.path.normpath(yolo["artifact_path"]) == os.path.normpath(YOLO_CANONICAL)
        if yolo.get("artifact_sha256"):
            assert yolo["artifact_sha256"] == disk


def test_missing_yolo_file_is_model_artifact_missing(tmp_path):
    missing = str(tmp_path / "does_not_exist.pt")
    det = YOLO11Detector(model_path=missing)
    assert det.model is None
    assert det.status == "MODEL_ARTIFACT_MISSING"
    with pytest.raises(RuntimeError, match="MODEL_ARTIFACT_MISSING"):
        det.detect("unused.jpg")


def test_yolo_classes_are_only_model_names():
    det = YOLO11Detector()
    info = det.get_info()
    named = [str(det.model.names[k]).lower() for k in sorted(det.model.names.keys())]
    assert info["classes"] == named
    assert info["supported_classes"] == named
    assert det.class_status("abrasion") == "abrasion"
    assert det.class_status("wound") == "UNTRAINED_CLASS"
    assert det.class_status("laceration") == "UNTRAINED_CLASS"
    assert det.class_status("swelling") == "UNTRAINED_CLASS"


def test_train_yolo_does_not_silently_remap_classes():
    import ml.training.train_yolo as train_yolo
    src = inspect.getsource(train_yolo)
    assert "class_remap" not in src
    assert "{0: 0, 1: 1, 2: 2, 3: 2}" not in src
    yaml_path = train_yolo.prepare_training_dataset()
    assert os.path.normpath(yaml_path) == os.path.normpath(
        os.path.join("data", "datasets", "yolo_processed", "data.yaml")
    )


def test_processed_yolo_dataset_honest_mapping_no_overlap():
    from collections import defaultdict
    root = os.path.join("data", "datasets", "yolo_processed")
    names = {0: "cut", 1: "bruise", 2: "wound"}
    pixels = defaultdict(set)
    class_ids = set()
    n_images = 0
    for split in ("train", "val", "test"):
        img_dir = os.path.join(root, "images", split)
        if not os.path.isdir(img_dir):
            continue
        for name in os.listdir(img_dir):
            img = os.path.join(img_dir, name)
            if not os.path.isfile(img):
                continue
            n_images += 1
            digest = _sha256(img)
            pixels[split].add(digest)
            stem = os.path.splitext(name)[0]
            lbl = os.path.join(root, "labels", split, stem + ".txt")
            assert os.path.exists(lbl), stem
            for line in open(lbl, encoding="utf-8"):
                parts = line.split()
                if parts:
                    cid = int(float(parts[0]))
                    class_ids.add(cid)
                    assert cid in names
                    assert names[cid] != "abrasion"
                    assert names[cid] != "laceration"
                    assert names[cid] != "swelling"
    assert n_images >= 1
    assert 2 not in class_ids
    assert not (pixels["train"] & pixels["val"])
    assert not (pixels["train"] & pixels["test"])
    assert not (pixels["val"] & pixels["test"])


def test_analyze_is_demo_not_filename_based():
    import backend.main as main
    src = inspect.getsource(main.analyze_case)
    assert 'football_injury.jpg' not in src


def test_classifier_yolo_coverage_is_semantic_not_yolo_fired():
    from backend.main import _classifier_yolo_coverage
    names = ["cut", "bruise", "wound"]
    withheld = {"winner": "Swelling", "is_confident": False}
    assert "NOT APPLICABLE" in _classifier_yolo_coverage(withheld, names)
    mismatch = {"winner": "Swelling", "is_confident": True}
    assert _classifier_yolo_coverage(mismatch, names) == "NOT AVAILABLE"
    match = {"winner": "Wound", "is_confident": True}
    assert _classifier_yolo_coverage(match, names) == "AVAILABLE"


def test_compose_full_image_mask_pastes_roi_not_stretch():
    import numpy as np
    from backend.main import _compose_full_image_mask
    orig_h, orig_w = 400, 700
    bbox = [100, 50, 300, 220]
    roi = np.ones((170, 200), dtype=np.uint8)
    full = _compose_full_image_mask(roi, orig_h, orig_w, bbox)
    assert full.shape == (400, 700)
    assert int(full[50:220, 100:300].sum()) == 170 * 200
    assert int(full[0:50].sum()) == 0
    assert int(full[:, 0:100].sum()) == 0


def test_first_aid_yolo_supported_classes_match_model_names():
    from backend.services.first_aid_service import StructuredEvidenceBuilder, first_aid_service
    ev = StructuredEvidenceBuilder.build_evidence(visible_injury={"yolo_finding_detected": False})
    assert ev["yolo"]["supported_classes"] == ["cut", "bruise", "abrasion"]
    assert "wound" not in ev["yolo"]["supported_classes"]
    assert "laceration" not in ev["yolo"]["supported_classes"]
    res = first_aid_service.generate_first_aid_guidance(
        visible_injury={"yolo_finding_detected": False, "yolo_finding": None}
    )
    yolo_line = [ev for ev in res["evidence_summary"] if "YOLO11 object detection" in ev][0]
    assert "wound" not in yolo_line
    assert "laceration" not in yolo_line
    assert "cut, bruise, abrasion" in yolo_line


def test_routing_helpers_tolerate_none_confidence():
    from backend.main import _routing_confidence, _routing_finding
    miss = {
        "yolo_finding_detected": False,
        "yolo_finding": None,
        "confidence": None,
        "classifier_finding": "Swelling",
        "classifier_probability": 0.91,
        "finding": "Swelling",
    }
    assert _routing_finding(miss) == "Swelling"
    assert _routing_confidence(miss) == 0.91
    untrusted = {**miss, "classifier_model_status": "NOT_TRUSTWORTHY"}
    assert _routing_finding(untrusted) == ""
    assert _routing_confidence(untrusted) == 0.0
    withheld = {**miss, "classifier_is_confident": False}
    assert _routing_finding(withheld) == ""
    assert _routing_confidence(withheld) == 0.0
    hit = {"yolo_finding_detected": True, "yolo_finding": "cut", "yolo_confidence": 0.42, "classifier_finding": "Swelling"}
    assert _routing_finding(hit) == "cut"
    assert _routing_confidence(hit) == 0.42
    normal = {
        "yolo_finding_detected": False,
        "classifier_finding": "Normal",
        "classifier_is_confident": True,
        "classifier_model_status": "READY_FOR_RESEARCH_DEMO",
    }
    assert _routing_finding(normal) == ""


def test_comparison_payload_prefers_canonical_nested_comparison():
    import json
    from backend.main import _comparison_api_payload, get_model_comparisons
    from ml.models.canonical_paths import EVAL_HELD_OUT, abs_path
    from fastapi.testclient import TestClient
    from backend.main import app

    stored = json.load(open(abs_path(EVAL_HELD_OUT), encoding="utf-8"))
    payload = _comparison_api_payload(stored)
    assert payload["status"] == "evaluated"
    assert payload["canonical_artifact"] == "data/results/canonical_held_out_evaluation.json"
    n = int(payload["sample_count"])
    xgb_cm = payload["classical_xgb"]["confusion_matrix"]
    vqc_cm = payload["quantum_vqc"]["confusion_matrix"]
    xgb_diag = sum(xgb_cm[i][i] for i in range(len(xgb_cm)))
    vqc_diag = sum(vqc_cm[i][i] for i in range(len(vqc_cm)))
    assert payload["classical_xgb"]["xgb_correct"] == f"{xgb_diag} / {n}"
    assert payload["quantum_vqc"]["vqc_correct"] == f"{vqc_diag} / {n}"
    assert abs(payload["classical_xgb"]["accuracy"] - xgb_diag / n) < 1e-6
    assert abs(payload["quantum_vqc"]["accuracy"] - vqc_diag / n) < 1e-6

    # Live endpoint must prefer the canonical held-out artifact (behavior, not source-text).
    client = TestClient(app)
    live = client.get("/api/evaluation/comparison").json()
    assert live["status"] == "evaluated"
    assert live["canonical_artifact"] == "data/results/canonical_held_out_evaluation.json"
    assert live["classical_xgb"]["xgb_correct"] == "25 / 30"
    assert live["quantum_vqc"]["vqc_correct"] == "16 / 30"
    assert get_model_comparisons is not None



