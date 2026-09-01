"""YOLO retrain_v2 promotion gates. Decision logic only — no weight overwrite."""
from ml.training.train_yolo_retrain_v2 import decide, kept_detections
from ml.vision.yolo_wrapper import DEFAULT_YOLO_INFER_CONF


def _metrics(cut_recall, bruise_recall):
    return {
        "per_class": {
            "cut": {"recall": cut_recall},
            "bruise": {"recall": bruise_recall},
        }
    }


def _log():
    return {
        "epochs_logged": 20,
        "first_epoch_box_loss": 1.0,
        "last_epoch_box_loss": 0.6,
    }


def _row(image, threshold, n_detections):
    return {"image": image, "threshold": threshold, "n_detections": n_detections}


def _sweeps(blank_c=0, dummy_c=0, blank_b=0, dummy_b=0, keep=DEFAULT_YOLO_INFER_CONF):
    return {
        "baseline_demo_negatives": [
            _row("data/datasets/yolo_injury/blank_skin.jpg", keep, blank_b),
            _row("data/datasets/yolo_injury/dummy_test.jpg", keep, dummy_b),
            _row("data/datasets/yolo_injury/blank_skin.jpg", 0.01, 0),
        ],
        "candidate_demo_negatives": [
            _row("data/datasets/yolo_injury/blank_skin.jpg", keep, blank_c),
            _row("data/datasets/yolo_injury/dummy_test.jpg", keep, dummy_c),
        ],
    }


def test_kept_detections_reads_application_threshold_only():
    rows = [
        _row("blank_skin.jpg", 0.01, 3),
        _row("blank_skin.jpg", DEFAULT_YOLO_INFER_CONF, 0),
        _row("dummy_test.jpg", DEFAULT_YOLO_INFER_CONF, 1),
    ]
    assert kept_detections(rows, "blank_skin") == 0
    assert kept_detections(rows, "dummy_test") == 1
    assert kept_detections(rows, "missing") is None


def test_promote_when_cut_improves_and_true_negatives_do_not_regress():
    decision = decide(
        _metrics(0.0, 0.99),
        _metrics(0.875, 1.0),
        _log(),
        "baseline-sha",
        "candidate-sha",
        sweeps=_sweeps(),
    )
    assert decision["promote"] is True
    assert decision["negative_gate"]["keep_threshold"] == DEFAULT_YOLO_INFER_CONF
    assert decision["negative_gate"]["images"]["blank_skin"]["candidate_n_detections"] == 0


def test_reject_when_true_negative_gains_boxes_at_keep_threshold():
    decision = decide(
        _metrics(0.0, 0.99),
        _metrics(0.875, 1.0),
        _log(),
        "baseline-sha",
        "candidate-sha",
        sweeps=_sweeps(blank_c=1, blank_b=0),
    )
    assert decision["promote"] is False
    assert any("NEGATIVE_BLANK_SKIN_FP_REGRESSED" in reason for reason in decision["reasons"])


def test_reject_when_negative_sweep_missing():
    decision = decide(
        _metrics(0.0, 0.99),
        _metrics(0.875, 1.0),
        _log(),
        "baseline-sha",
        "candidate-sha",
        sweeps={"baseline_demo_negatives": [], "candidate_demo_negatives": []},
    )
    assert decision["promote"] is False
    assert any("SWEEP_MISSING" in reason for reason in decision["reasons"])


def test_reject_when_candidate_sha_equals_production():
    decision = decide(
        _metrics(0.0, 0.99),
        _metrics(0.875, 1.0),
        _log(),
        "same",
        "same",
        sweeps=_sweeps(),
    )
    assert decision["promote"] is False
    assert "CANDIDATE_SHA_EQUALS_PRODUCTION" in decision["reasons"]


def test_api_registry_yolo_sha_matches_canonical_disk():
    import os
    from fastapi.testclient import TestClient
    from backend.main import app
    from ml.models.canonical_paths import YOLO_CANONICAL, sha256_file
    from ml.vision.yolo_wrapper import YOLO11Detector

    disk = sha256_file(YOLO_CANONICAL)
    wrapper = YOLO11Detector()
    client = TestClient(app)
    response = client.get("/api/models/registry")
    assert response.status_code == 200
    yolo = response.json()["YOLO11 Detection"]
    assert yolo["artifact_sha256"] == disk
    assert wrapper.artifact_sha256 == disk
    assert yolo["canonical_path"].replace("\\", "/") == "ml/models/vision/yolo11_injury_best.pt"

    forensic = os.path.join("data", "uploads", "3e0dbd17-7475-487d-9f10-e7f9d6800238.jpg")
    findings = wrapper.detect(forensic)
    assert findings == []

    import inspect
    from backend import main as backend_main
    source = inspect.getsource(backend_main.analyze_case)
    assert 'confidence = best_det["confidence"]' in source
    assert "detections = yolo_det.detect(img_ref)" in source
