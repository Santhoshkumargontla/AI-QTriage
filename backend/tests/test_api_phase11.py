import os
import pytest
import numpy as np
import pandas as pd
import torch
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.classifiers.vqc_classifier import VQCClassifier
from ml.evaluation.comparison_metrics import ClassicalQuantumEvaluator, compute_ece, compute_brier_score
from backend.database.connection import get_database

@pytest.fixture
def temp_weights_dir(tmpdir):
    return str(tmpdir)

def test_ece_and_brier_calculators():
    """Verify mathematical correctness of ECE and Brier calculations."""
    # 1. Brier score for 2 samples, 3 classes
    # True class labels: sample 0 is class 1 (MODERATE), sample 1 is class 2 (HIGH)
    y_true = np.array([1, 2])
    probs = np.array([
        [0.1, 0.8, 0.1],  # True is 1. error = (0.1)**2 + (0.8 - 1)**2 + (0.1)**2 = 0.01 + 0.04 + 0.01 = 0.06
        [0.2, 0.1, 0.7]   # True is 2. error = (0.2)**2 + (0.1)**2 + (0.7 - 1)**2 = 0.04 + 0.01 + 0.09 = 0.14
    ]) # Total error = (0.06 + 0.14) / 2 = 0.10
    
    brier = compute_brier_score(probs, y_true)
    assert pytest.approx(brier, abs=1e-5) == 0.10

    # 2. Expected Calibration Error (ECE) for 1 bin
    # If all items fall into a single bin, ECE = |average accuracy - average confidence|
    # Samples: 2. Preds: 1 (conf 0.8, true 1, correct), 2 (conf 0.7, true 2, correct)
    # Average confidence = 0.75. Average accuracy = 1.0.
    # ECE = |1.0 - 0.75| = 0.25
    ece = compute_ece(probs, y_true, num_bins=1)
    assert pytest.approx(ece, abs=1e-5) == 0.25

def test_classical_quantum_comparison_metrics(temp_weights_dir):
    """Test that ClassicalQuantumEvaluator runs predictions and commits comparative data to MongoDB."""
    manifest_path = os.path.join(temp_weights_dir, "manifest.csv")
    
    # 1. Create a mock manifest with two test entries
    df = pd.DataFrame([
        {
            "sample_id": "c_test_01",
            "subject_id": "patient_01",
            "image_path": "dummy.png",
            "class": "swelling",
            "mask_path": "dummy_mask.png",
            "source": "Kaggle",
            "license": "CC-BY",
            "split": "test"
        },
        {
            "sample_id": "c_test_02",
            "subject_id": "patient_02",
            "image_path": "dummy.png",
            "class": "cut",
            "mask_path": "dummy_mask.png",
            "source": "Kaggle",
            "license": "CC-BY",
            "split": "test"
        }
    ])
    df.to_csv(manifest_path, index=False)
    
    # Generate dummy files to satisfy path validations
    with open(os.path.join(temp_weights_dir, "dummy.png"), "w") as f:
        f.write("mock image")
    with open(os.path.join(temp_weights_dir, "dummy_mask.png"), "w") as f:
        f.write("mock mask")

    # 2. Generate and train mock classifiers
    np.random.seed(42)
    X_train = np.random.randn(20, 23).astype(np.float32)
    y_train = np.random.randint(0, 3, 20).astype(np.int32)
    
    # Train and save XGBoost model
    xgb_path = os.path.join(temp_weights_dir, "xgb_temp.json")
    xgb_clf = XGBoostClassifier()
    xgb_clf.train(X_train, y_train)
    xgb_clf.save_model(xgb_path)
    
    # Train and save VQC model
    vqc_clf = VQCClassifier()
    vqc_clf.train(X_train, y_train, epochs=2)
    vqc_clf.save_model(temp_weights_dir)

    # 3. Execute comparison metrics pipeline
    evaluator = ClassicalQuantumEvaluator(dataset_dir=temp_weights_dir, manifest_path=manifest_path)
    comparison = evaluator.run_comparison(xgb_weights_path=xgb_path, vqc_model_dir=temp_weights_dir)
    
    # Validate result keys
    assert "classical_xgb" in comparison
    assert "quantum_vqc" in comparison
    assert "mcc" in comparison["classical_xgb"]
    assert "ece" in comparison["quantum_vqc"]
    assert "Experimental VQC outputs" in comparison["interpretation"]
    
    # 4. Verify persisted in MongoDB comparisons collection
    db = get_database()
    saved = db.model_comparisons.find_one()
    assert saved is not None
    assert saved["classical_xgb"]["mcc"] == comparison["classical_xgb"]["mcc"]
    assert saved["quantum_vqc"]["brier_score"] == comparison["quantum_vqc"]["brier_score"]
