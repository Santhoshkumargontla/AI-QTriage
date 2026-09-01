import os
import pytest
import numpy as np
from ml.classifiers.vqc_classifier import VQCClassifier

@pytest.fixture
def temp_weights_dir(tmpdir):
    return str(tmpdir)

def test_vqc_fit_predict_serialize(temp_weights_dir):
    """Test that VQC classifier handles scaler/PCA, executes training, predicts and serializes."""
    # 1. Generate small dummy training set
    np.random.seed(42)
    X_train = np.random.randn(15, 23).astype(np.float32)
    y_train = np.random.randint(0, 3, 15).astype(np.int32)
    
    # 2. Instantiate and train VQC
    vqc = VQCClassifier()
    assert vqc.is_trained is False
    
    # Train for a few epochs to verify parameter updates without taking too long
    vqc.train(X_train, y_train, epochs=3, lr=0.1)
    assert vqc.is_trained is True
    
    # 3. Predict on a dummy sample
    dummy_sample = np.random.randn(23).astype(np.float32)
    pred_idx, scores = vqc.predict(dummy_sample)
    
    assert pred_idx in [0, 1, 2]
    assert len(scores) == 3
    # Experimental VQC output scores should represent normalized softmax outputs summing to 1.0
    assert pytest.approx(sum(scores), abs=1e-4) == 1.0

    # 4. Save model to temp folder and verify file footprints
    vqc.save_model(temp_weights_dir)
    assert os.path.exists(os.path.join(temp_weights_dir, "scaler.pkl"))
    assert os.path.exists(os.path.join(temp_weights_dir, "pca.pkl"))
    assert os.path.exists(os.path.join(temp_weights_dir, "vqc_weights.npz"))
    
    # 5. Reload in a new instance and check matching predictions
    reloaded_vqc = VQCClassifier(model_dir=temp_weights_dir)
    assert reloaded_vqc.is_trained is True
    
    reloaded_pred, reloaded_scores = reloaded_vqc.predict(dummy_sample)
    assert reloaded_pred == pred_idx
    assert reloaded_scores == scores
