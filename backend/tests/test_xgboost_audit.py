"""XGBoost forensic audit: 23-feature schema, zeros, missing artifact, provenance. No retrain."""
import inspect
import json
import os

import numpy as np
import pytest

from ml.classifiers.xgboost_classifier import (
    MODEL_ARTIFACT_MISSING,
    XGBoostClassifier,
    load_xgboost_metadata,
    provenance_class_from_metadata,
)
from ml.fusion.feature_fusion import FEATURE_NAMES, N_FEATURES, MultimodalFeatureFusion
from ml.models.canonical_paths import XGB_CANONICAL, exists, sha256_file
from ml.training import train_xgboost as train_xgb_mod


EXPECTED_ORDER = [
    "vision_present",
    "prob_cut", "prob_bruise", "prob_swelling", "prob_other",
    "affected_ratio",
    "questionnaire_present",
    "pain_level",
    "mech_fall", "mech_impact", "mech_sports", "mech_sharp", "mech_other",
    "direct_impact", "visible_bleeding", "movement_limitation", "weight_bearing", "crack_pop",
    "sensor_present",
    "peak_g_force", "delta_v", "stabilization_time", "lux_drop",
]


def test_schema_is_exactly_23_and_order_matches():
    assert N_FEATURES == 23
    assert FEATURE_NAMES == EXPECTED_ORDER
    fusion = MultimodalFeatureFusion()
    clf = XGBoostClassifier(XGB_CANONICAL)
    assert fusion.feature_names == EXPECTED_ORDER
    assert clf.feature_names == EXPECTED_ORDER
    assert int(clf.model.n_features_in_ or clf.model.get_booster().num_features()) == 23
    _, vector, names = fusion.fuse_features({})
    assert vector.shape == (23,)
    assert names == EXPECTED_ORDER


def test_missing_modality_option_a_zeros():
    fusion = MultimodalFeatureFusion()
    _, vector, names = fusion.fuse_features({})
    assert vector.tolist() == [0.0] * 23
    assert vector[names.index("vision_present")] == 0.0
    assert vector[names.index("questionnaire_present")] == 0.0
    assert vector[names.index("sensor_present")] == 0.0


def test_explicit_zeros_are_not_replaced():
    fusion = MultimodalFeatureFusion()
    _, vector, names = fusion.fuse_features({
        "vision_analysis": {
            "classification": {"Cut": 0.0, "Bruise": 0.0, "Swelling": 0.0, "Other": 0.0},
            "segmentation": {"affected_ratio": 0.0},
        },
        "questionnaire": {"answers": {"pain_level": 0, "injury_mechanism": "sports"}},
        "sensor_summary": {
            "peak_g_force": 0.0,
            "pre_impact_delta_v": 0.0,
            "post_impact_stabilization_seconds": 0.0,
            "optical_lux_drop": False,
        },
    })
    assert vector[names.index("prob_cut")] == 0.0
    assert vector[names.index("affected_ratio")] == 0.0
    assert vector[names.index("pain_level")] == 0.0
    assert vector[names.index("peak_g_force")] == 0.0
    assert vector[names.index("delta_v")] == 0.0
    assert vector[names.index("stabilization_time")] == 0.0
    assert vector[names.index("vision_present")] == 1.0
    assert vector[names.index("questionnaire_present")] == 1.0
    assert vector[names.index("sensor_present")] == 1.0


def test_missing_fields_are_not_imputed_as_five_or_one():
    fusion = MultimodalFeatureFusion()
    _, vector, names = fusion.fuse_features({
        "questionnaire": {"answers": {"location": "ankle"}},
        "sensor_summary": {"optical_lux_drop": True},
        "vision_analysis": {
            "classification": {"Cut": None, "Bruise": None, "Swelling": None, "Other": None},
            "segmentation": {"affected_ratio": None},
        },
    })
    assert vector[names.index("pain_level")] == 0.0
    assert vector[names.index("peak_g_force")] == 0.0
    assert vector[names.index("prob_cut")] == 0.0
    assert vector[names.index("affected_ratio")] == 0.0
    assert 5.0 not in vector.tolist()
    assert 1.0 == vector[names.index("questionnaire_present")]
    assert 1.0 == vector[names.index("sensor_present")]
    assert 1.0 == vector[names.index("vision_present")]
    assert 1.0 == vector[names.index("lux_drop")]


def test_missing_artifact_raises_model_artifact_missing(tmp_path):
    missing = str(tmp_path / "no_such_xgboost.json")
    with pytest.raises(RuntimeError) as exc:
        XGBoostClassifier(missing)
    assert MODEL_ARTIFACT_MISSING in str(exc.value)
    assert missing in str(exc.value)


def test_uninitialized_predict_does_not_autotrain():
    clf = XGBoostClassifier()
    assert clf.is_trained is False
    with pytest.raises(RuntimeError, match="not trained yet"):
        clf.predict(np.zeros(23, dtype=np.float32))


def test_analyze_and_train_script_do_not_runtime_synthesize():
    import backend.main as main
    analyze_src = inspect.getsource(main.analyze_case)
    require_src = inspect.getsource(main.require_model_artifacts)
    train_src = inspect.getsource(train_xgb_mod.train_xgboost)
    assert "generate_multimodal_dataset" not in analyze_src
    assert "np.random.randn" not in analyze_src
    assert "X_dummy" not in analyze_src
    assert MODEL_ARTIFACT_MISSING in require_src
    assert "XGB_CANONICAL" in train_src
    assert "save_model(XGB_CANONICAL)" in train_src


def test_data_provenance_is_synthetic_not_real_or_mixed():
    meta = load_xgboost_metadata()
    provenance = provenance_class_from_metadata(meta)
    assert provenance == "SYNTHETIC"
    assert provenance in ("REAL", "SYNTHETIC", "MIXED")
    assert int(meta["metrics"]["genuinely_paired_clinical_samples"]) == 0
    assert int(meta["metrics"]["synthetic_multimodal_fusion_samples"]) == 200
    clf = XGBoostClassifier(XGB_CANONICAL)
    info = clf.get_info()
    assert info["data_provenance"] == "SYNTHETIC"
    assert info["n_features"] == 23


def test_canonical_artifact_exists_and_matches_registry_hash():
    assert exists(XGB_CANONICAL)
    registry_path = os.path.join("ml", "models", "model_registry.json")
    with open(registry_path, encoding="utf-8") as handle:
        registry = json.load(handle)
    entry = registry["XGBoost Multimodal"]
    digest = sha256_file(XGB_CANONICAL)
    assert digest == entry["artifact_sha256"]
    assert entry.get("data_provenance_class") == "SYNTHETIC"
    clf = XGBoostClassifier(XGB_CANONICAL)
    idx, probs = clf.predict(np.zeros(23, dtype=np.float32))
    assert idx in (0, 1, 2)
    assert len(probs) == 3
    with pytest.raises(ValueError):
        clf.predict(np.zeros(22, dtype=np.float32))


def test_health_payload_uses_metadata_not_file_existence_alias():
    from backend.main import get_models
    models = get_models()
    xgb = next(m for m in models if m["model_name"] == "XGBoost")
    assert xgb["schema_features"] == 23
    assert xgb["data_provenance"] == "SYNTHETIC"
    assert xgb["status"] != "TRAINED_AND_EVALUATED"
    if exists(XGB_CANONICAL):
        assert xgb["status"] == "TRAINED"
        assert xgb["weights_loaded"] is True
    else:
        assert xgb["status"] == MODEL_ARTIFACT_MISSING
