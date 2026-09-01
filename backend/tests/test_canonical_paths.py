"""Canonical model path / hash / registry alignment. No invented metrics."""
import json
import os
import tempfile

from ml.classifiers.xgboost_classifier import XGBoostClassifier, load_xgboost_metadata
from ml.models.canonical_paths import (
    EFFNET_CANONICAL,
    EFFNET_METADATA,
    EVAL_HELD_OUT,
    EVAL_YOLO_THRESHOLD,
    MANIFEST_PATH,
    REGISTRY_PATH,
    ROOT,
    RUNTIME_MODELS,
    UNET_CANONICAL,
    UNET_METADATA,
    VQC_METADATA,
    VQC_WEIGHTS,
    XGB_CANONICAL,
    XGB_METADATA,
    YOLO_CANONICAL,
    YOLO_METADATA,
    YOLO_RETRAIN_V2_BEST,
    YOLO_SYNTHETIC_BASELINE,
    abs_path,
    exists,
    posix,
    read_json,
    resolve_existing,
    sha256_file,
)
from ml.vision.yolo_wrapper import YOLO11Detector


def test_project_root_constant_is_workspace():
    assert os.path.isdir(os.path.join(ROOT, "ml", "models"))
    assert os.path.isfile(abs_path(YOLO_CANONICAL))


def test_resolve_existing_prefers_project_root_over_cwd_shadow(tmp_path):
    """A same-relative-path file under cwd must not shadow the canonical artifact."""
    shadow_dir = tmp_path / "ml" / "models"
    shadow_dir.mkdir(parents=True)
    shadow = shadow_dir / "xgboost_best.json"
    shadow.write_text("not-the-canonical-booster", encoding="utf-8")
    previous = os.getcwd()
    try:
        os.chdir(tmp_path)
        located = resolve_existing(XGB_CANONICAL)
        assert os.path.normpath(located) == os.path.normpath(abs_path(XGB_CANONICAL))
        shadow_abs = os.path.abspath(str(shadow))
        assert os.path.normpath(located) != os.path.normpath(shadow_abs)
        assert sha256_file(XGB_CANONICAL) != sha256_file(str(shadow))
        assert exists(XGB_CANONICAL)
    finally:
        os.chdir(previous)


def test_runtime_registry_manifest_share_one_sha():
    registry = read_json(REGISTRY_PATH)
    manifest = read_json(MANIFEST_PATH)
    assert registry
    assert manifest.get("models")
    by_name = {row["model_name"]: row for row in manifest["models"]}
    for spec in RUNTIME_MODELS:
        name = spec["model_name"]
        path = spec["canonical_path"]
        assert exists(path), f"missing canonical artifact {path}"
        disk = sha256_file(path)
        reg = registry[name]
        man = by_name[name]
        assert posix(reg["canonical_path"]) == posix(path)
        assert posix(reg["artifact_path"]) == posix(path)
        assert reg["artifact_sha256"] == disk
        assert man["sha256"] == disk
        assert posix(man["canonical_path"]) == posix(path)
        assert posix(man["evaluation_artifact"]) == posix(spec["evaluation_artifact"])
        assert exists(spec["evaluation_artifact"]), f"missing eval artifact for {name}"
        assert man["used_by_runtime"] is True


def test_wrappers_load_canonical_sha():
    yolo = YOLO11Detector()
    assert yolo.model is not None
    assert yolo.artifact_sha256 == sha256_file(YOLO_CANONICAL)
    assert os.path.normpath(abs_path(yolo.get_info()["canonical_path"])) == os.path.normpath(abs_path(YOLO_CANONICAL))

    xgb = XGBoostClassifier(XGB_CANONICAL)
    assert xgb.is_trained
    assert xgb.artifact_sha256 == sha256_file(XGB_CANONICAL)
    meta = load_xgboost_metadata()
    assert meta.get("canonical_path") in (None, posix(XGB_CANONICAL), XGB_CANONICAL.replace("\\", "/"))
    assert meta.get("artifact_sha256") == sha256_file(XGB_CANONICAL)


def test_loads_from_backend_and_foreign_cwd():
    previous = os.getcwd()
    backend = os.path.join(ROOT, "backend")
    foreign = tempfile.mkdtemp(prefix="aiq_cwd_")
    try:
        for cwd in (backend, foreign):
            os.chdir(cwd)
            assert exists(YOLO_CANONICAL)
            assert exists(EFFNET_CANONICAL)
            assert exists(UNET_CANONICAL)
            assert exists(XGB_CANONICAL)
            assert exists(EVAL_HELD_OUT)
            xgb = XGBoostClassifier(XGB_CANONICAL)
            assert xgb.is_trained
            meta = load_xgboost_metadata()
            assert meta.get("status") == "TRAINED"
            yolo = YOLO11Detector()
            assert yolo.model is not None
            assert yolo.artifact_sha256 == sha256_file(YOLO_CANONICAL)
    finally:
        os.chdir(previous)


def test_eval_and_metadata_bind_canonical_sha():
    assert read_json(YOLO_METADATA).get("artifact_sha256") == sha256_file(YOLO_CANONICAL)
    assert read_json(EFFNET_METADATA).get("artifact_sha256") == sha256_file(EFFNET_CANONICAL)
    assert read_json(UNET_METADATA).get("artifact_sha256") == sha256_file(UNET_CANONICAL)
    assert read_json(XGB_METADATA).get("artifact_sha256") == sha256_file(XGB_CANONICAL)
    assert read_json(VQC_METADATA).get("artifact_sha256") == sha256_file(VQC_WEIGHTS)
    held = read_json(EVAL_HELD_OUT)
    assert held.get("evaluation", {}).get("metrics_comparison", {}).get("xgb_accuracy") == 0.833333
    assert held.get("comparison", {}).get("vqc", {}).get("accuracy") == 0.533333
    # Historical threshold-sweep file describes the pre-roboflow checkpoint, not the live one.
    yolo_eval = read_json(EVAL_YOLO_THRESHOLD)
    live = sha256_file(YOLO_CANONICAL)
    assert yolo_eval.get("sha256") in {
        live,
        "4d6e72f5f671fd60065ffc13cbb14efaac1da04ad062e116c118f2a688202879",
    }


def test_yolo_training_run_copy_is_not_runtime():
    assert os.path.isfile(abs_path(YOLO_RETRAIN_V2_BEST))
    # After Roboflow-v1 promotion, the v2 training-run copy is a historical artifact, not a byte duplicate.
    assert sha256_file(YOLO_RETRAIN_V2_BEST) != sha256_file(YOLO_CANONICAL)
    yolo = YOLO11Detector()
    assert yolo.artifact_sha256 == sha256_file(YOLO_CANONICAL)
    assert "yolo_retrain_v2" not in posix(yolo.get_info()["canonical_path"])
    assert posix(yolo.get_info()["canonical_path"]) == posix(YOLO_CANONICAL)


def test_explicit_relative_vision_paths_resolve_from_backend_cwd():
    previous = os.getcwd()
    try:
        os.chdir(os.path.join(ROOT, "backend"))
        assert not os.path.exists(EFFNET_CANONICAL)
        assert not os.path.exists(UNET_CANONICAL)
        from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
        from ml.vision.unet_wrapper import UNetSegmenter
        eff = EfficientNetV2Classifier(model_path=EFFNET_CANONICAL)
        unet = UNetSegmenter(model_path=UNET_CANONICAL)
        assert eff.is_loaded
        assert unet.is_loaded
        assert os.path.normpath(eff.model_path) == os.path.normpath(abs_path(EFFNET_CANONICAL))
        assert os.path.normpath(unet.model_path) == os.path.normpath(abs_path(UNET_CANONICAL))
    finally:
        os.chdir(previous)


def test_synthetic_baseline_is_not_runtime():
    assert exists(YOLO_SYNTHETIC_BASELINE)
    assert sha256_file(YOLO_SYNTHETIC_BASELINE) != sha256_file(YOLO_CANONICAL)


def test_pre_retrain_v2_backup_is_preserved_and_not_runtime():
    backup = abs_path(YOLO_CANONICAL + ".pre_retrain_v2_backup")
    assert os.path.isfile(backup)
    backup_sha = sha256_file(backup)
    canon_sha = sha256_file(YOLO_CANONICAL)
    assert backup_sha != canon_sha
    assert backup_sha == "6cc84115e4cb85c8b82715211c3935200b815b76efbc95f83855c2cc988dce4f"
    yolo = YOLO11Detector()
    assert yolo.artifact_sha256 == canon_sha
    assert yolo.artifact_sha256 != backup_sha


def test_historical_retrained_yolo_is_not_runtime():
    hist = abs_path(os.path.join("ml", "models", "vision", "yolo11_injury_real_retrained.pt"))
    assert os.path.isfile(hist)
    hist_sha = sha256_file(hist)
    canon_sha = sha256_file(YOLO_CANONICAL)
    assert hist_sha != canon_sha
    yolo = YOLO11Detector()
    assert yolo.model is not None
    assert yolo.artifact_sha256 == canon_sha
    assert yolo.artifact_sha256 != hist_sha
    info = yolo.get_info()
    assert os.path.normpath(abs_path(info["canonical_path"])) == os.path.normpath(abs_path(YOLO_CANONICAL))
