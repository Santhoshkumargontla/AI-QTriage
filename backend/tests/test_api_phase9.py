import os
import pytest
import numpy as np
from ml.classifiers.xgboost_classifier import XGBoostClassifier

@pytest.fixture
def temp_weights_dir(tmpdir):
    return str(tmpdir)

def test_xgboost_fit_predict_explain(temp_weights_dir):
    """Test that XGBoost classifier trains, predicts, serializes, and runs SHAP attributions."""
    # 1. Generate dummy training dataset
    np.random.seed(42)
    X_train = np.random.randn(100, 23).astype(np.float32)
    y_train = np.random.randint(0, 3, 100).astype(np.int32)
    
    # 2. Instantiate and fit model
    classifier = XGBoostClassifier()
    assert classifier.is_trained is False
    
    classifier.train(X_train, y_train)
    assert classifier.is_trained is True
    
    # 3. Predict on test sample
    dummy_sample = np.random.randn(23).astype(np.float32)
    pred_idx, probs = classifier.predict(dummy_sample)
    
    assert pred_idx in [0, 1, 2]
    assert len(probs) == 3
    assert pytest.approx(sum(probs), abs=1e-4) == 1.0

    # 4. Save and reload model weights
    save_path = os.path.join(temp_weights_dir, "xgboost_best.json")
    classifier.save_model(save_path)
    assert os.path.exists(save_path)
    
    reloaded_classifier = XGBoostClassifier(model_path=save_path)
    assert reloaded_classifier.is_trained is True
    
    reloaded_pred, reloaded_probs = reloaded_classifier.predict(dummy_sample)
    assert reloaded_pred == pred_idx
    assert reloaded_probs == probs

    # 5. Local SHAP Explanation Attributions
    local_exps = reloaded_classifier.explain_prediction(dummy_sample, pred_idx)
    assert len(local_exps) == 23
    assert "feature" in local_exps[0]
    assert "shap_value" in local_exps[0]
    assert "Feature contribution analysis" in local_exps[0]["description"]
    
    # Verify sorted descending by absolute value
    abs_values = [abs(x["shap_value"]) for x in local_exps]
    assert abs_values == sorted(abs_values, reverse=True)

    # 6. Global Feature Importances
    global_imps = reloaded_classifier.get_global_importance()
    assert len(global_imps) == 23
    assert "feature" in global_imps[0]
    assert "importance" in global_imps[0]
    assert isinstance(global_imps[0]["importance"], float)
    
    # Verify sorted descending by importance
    importances = [x["importance"] for x in global_imps]
    assert importances == sorted(importances, reverse=True)
