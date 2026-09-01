"""VQC pipeline: matching circuit, no fabricated outputs, MODEL_UNAVAILABLE isolation."""
import inspect
import os

import numpy as np
import pytest

from ml.classifiers.vqc_classifier import (
    EXPERIMENTAL_ONLY,
    MODEL_UNAVAILABLE,
    VQCClassifier,
)
from ml.models.canonical_paths import VQC_DIR


def test_train_and_predict_use_the_same_circuit():
    src_train = inspect.getsource(VQCClassifier.train)
    src_predict = inspect.getsource(VQCClassifier.predict)
    src_forward = inspect.getsource(VQCClassifier._forward)
    src_circuit = inspect.getsource(VQCClassifier._vqc_circuit)
    assert "self._forward" in src_train
    assert "self._forward" in src_predict
    assert "self.circuit" in src_forward
    assert "AngleEmbedding" in src_circuit
    assert "StronglyEntanglingLayers" in src_circuit
    clf = VQCClassifier()
    spec = clf.circuit_spec()
    assert spec["circuits_match"] is True
    assert spec["embedding"] == "AngleEmbedding"
    assert spec["ansatz"] == "StronglyEntanglingLayers"
    assert spec["num_qubits"] == 4


def test_predict_has_no_hardcoded_or_fabricated_fallback():
    src = inspect.getsource(VQCClassifier.predict)
    assert "[0.15, 0.70, 0.15]" not in src
    assert "[0.33, 0.33, 0.33]" not in src
    assert "except Exception:" not in src
    clf = VQCClassifier()
    with pytest.raises(RuntimeError, match=MODEL_UNAVAILABLE):
        clf.predict(np.zeros(23, dtype=np.float32))


def test_missing_artifact_is_model_unavailable(tmp_path):
    with pytest.raises((FileNotFoundError, RuntimeError)) as exc:
        VQCClassifier(str(tmp_path / "no_vqc"))
    assert MODEL_UNAVAILABLE in str(exc.value)


def test_incomplete_weights_are_not_zero_filled(tmp_path):
    import pickle

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(np.random.randn(10, 23))
    pca = PCA(n_components=4).fit(scaler.transform(np.random.randn(10, 23)))
    with open(tmp_path / "scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)
    with open(tmp_path / "pca.pkl", "wb") as handle:
        pickle.dump(pca, handle)
    np.savez(tmp_path / "vqc_weights.npz", q_weights=np.zeros((2, 4, 3)))
    with pytest.raises(RuntimeError, match=MODEL_UNAVAILABLE):
        VQCClassifier(str(tmp_path))


def test_training_loss_records_actual_optimization():
    rng = np.random.RandomState(0)
    X = rng.randn(12, 23).astype(np.float32)
    y = rng.randint(0, 3, 12).astype(np.int32)
    clf = VQCClassifier()
    clf.train(X, y, epochs=3, lr=0.2)
    assert clf.optimization is not None
    assert clf.optimization["optimizer"] == "PennyLane AdamOptimizer"
    assert len(clf.loss_history) == 3
    idx, probs = clf.predict(X[0])
    assert idx in (0, 1, 2)
    assert len(probs) == 3
    assert pytest.approx(sum(probs), abs=1e-4) == 1.0
    assert all(np.isfinite(probs))


def test_analyze_does_not_require_vqc_and_isolates_failure():
    import backend.main as main

    require_src = inspect.getsource(main.require_model_artifacts)
    analyze_src = inspect.getsource(main.analyze_case)
    assert "vqc_weights" not in require_src
    assert "VQC_EXCLUDED_FROM_DECISION" in analyze_src
    assert "used_in_main_decision" in analyze_src
    assert MODEL_UNAVAILABLE in analyze_src
    assert EXPERIMENTAL_ONLY in analyze_src


def test_loaded_artifact_is_experimental_only():
    if not os.path.exists(os.path.join(VQC_DIR, "vqc_weights.npz")):
        pytest.skip("VQC artifact not on disk")
    clf = VQCClassifier(VQC_DIR)
    assert clf.status == EXPERIMENTAL_ONLY
    assert clf.get_info()["used_in_main_decision"] is False
    assert clf.get_info()["data_provenance"] == "SYNTHETIC"
