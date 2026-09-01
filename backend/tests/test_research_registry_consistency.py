"""Research Mode / registry display must match canonical artifacts — no invented N/A."""
from fastapi.testclient import TestClient

from backend.main import app, _enrich_registry_entry, _comparison_api_payload
from ml.models.canonical_paths import (
    EFFNET_CANONICAL,
    EVAL_HELD_OUT,
    UNET_CANONICAL,
    VQC_WEIGHTS,
    XGB_CANONICAL,
    YOLO_CANONICAL,
    read_json,
    sha256_file,
)


def test_registry_api_exposes_held_out_metrics_and_split_labels():
    client = TestClient(app)
    registry = client.get("/api/models/registry").json()
    assert "XGBoost Multimodal" in registry
    xgb = registry["XGBoost Multimodal"]
    assert xgb["artifact_sha256"] == sha256_file(XGB_CANONICAL)
    assert xgb["display_held_out_metric"] == "25 / 30"
    assert "train" in xgb["display_sample_count"]
    assert "test" in xgb["display_sample_count"]

    vqc = registry["Experimental 4-Qubit VQC"]
    assert vqc["artifact_sha256"] == sha256_file(VQC_WEIGHTS)
    assert vqc["display_held_out_metric"] == "16 / 30"
    assert vqc["metrics"].get("correct_predictions") == "16 / 30"

    yolo = registry["YOLO11 Detection"]
    assert yolo["artifact_sha256"] == sha256_file(YOLO_CANONICAL)
    assert "mAP50" in yolo["display_held_out_metric"]
    assert "0.5382" in yolo["display_held_out_metric"]
    assert "0.8358" not in yolo["display_held_out_metric"]

    unet = registry["ResNet34-UNet Segmentation"]
    assert unet["artifact_sha256"] == sha256_file(UNET_CANONICAL)
    assert "Dice" in unet["display_held_out_metric"]
    assert "val" in unet["display_sample_count"] and "test" in unet["display_sample_count"]

    eff = registry["EfficientNetV2 Classification"]
    assert eff["artifact_sha256"] == sha256_file(EFFNET_CANONICAL)
    status_blob = f"{eff.get('display_held_out_metric')} {eff.get('status')} {eff.get('training_status')}"
    assert (
        "READY_FOR_RESEARCH_DEMO" in status_blob
        or "NOT_TRUSTWORTHY" in status_blob
        or "reject" in status_blob.lower()
    )

def test_comparison_matches_canonical_held_out_and_selective_not_fabricated():
    held = read_json(EVAL_HELD_OUT)
    payload = _comparison_api_payload(held)
    assert payload["sample_count"] == 30
    assert payload["classical_xgb"]["xgb_correct"] == "25 / 30"
    assert abs(payload["classical_xgb"]["accuracy"] - 0.833333) < 1e-6
    assert payload["quantum_vqc"]["vqc_correct"] == "16 / 30"
    assert abs(payload["quantum_vqc"]["accuracy"] - 0.533333) < 1e-6
    sel = payload["selective_classification"]
    assert sel.get("status") == "not_available" or sel.get("coverage") is None
    assert sel.get("coverage") != "0.0%"


def test_enrich_sensor_metric_from_correct_predictions():
    entry = {
        "model_name": "Sensor Motion Event Classifier",
        "status": "TRAINED",
        "sample_count": 138,
        "metrics": {
            "train_samples": 138,
            "val_samples": 26,
            "test_samples": 36,
            "correct_predictions": "28 / 36",
            "accuracy": 0.777778,
        },
    }
    enriched = _enrich_registry_entry(entry)
    assert enriched["display_held_out_metric"] == "28 / 36"
    assert "train 138" in enriched["display_sample_count"]
    assert "test 36" in enriched["display_sample_count"]


def test_enrich_unet_uses_nested_split_n():
    entry = {
        "model_name": "ResNet34-UNet Segmentation",
        "status": "READY_FOR_RESEARCH_DEMO",
        "metrics": {
            "val": {"n": 69, "mean_dice": 0.66},
            "test": {"n": 69, "mean_dice": 0.641808},
            "ood_collapse": 0,
        },
    }
    enriched = _enrich_registry_entry(entry)
    assert "val 69" in enriched["display_sample_count"]
    assert "test 69" in enriched["display_sample_count"]
    assert "Dice(test): 0.6418" == enriched["display_held_out_metric"]


def test_api_models_versions_align_with_registry_for_vision():
    client = TestClient(app)
    models = {m["model_name"]: m for m in client.get("/api/models").json()}
    registry = client.get("/api/models/registry").json()
    assert models["EfficientNetV2"]["artifact_sha256"] == registry["EfficientNetV2 Classification"]["artifact_sha256"]
    assert models["U-Net"]["artifact_sha256"] == registry["ResNet34-UNet Segmentation"]["artifact_sha256"]
    assert models["EfficientNetV2"]["model_version"] == registry["EfficientNetV2 Classification"]["version"]
    assert models["U-Net"]["model_version"] == registry["ResNet34-UNet Segmentation"]["version"]
    assert models["VQC"]["artifact_sha256"] == sha256_file(VQC_WEIGHTS)
