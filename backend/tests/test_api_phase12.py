import os
import pytest
import numpy as np
import pandas as pd
from ml.classifiers.xgboost_classifier import XGBoostClassifier
from ml.classifiers.vqc_classifier import VQCClassifier
from ml.evaluation.research_benchmarks import ResearchBenchmarkRunner
from backend.database.connection import get_database

@pytest.fixture
def temp_weights_dir(tmpdir):
    return str(tmpdir)

def test_research_benchmarks_pipeline(temp_weights_dir):
    """Test that ResearchBenchmarkRunner executes ablation, perturbation, and coverage metrics."""
    manifest_path = os.path.join(temp_weights_dir, "manifest.csv")
    
    # 1. Create a mock manifest with two test entries
    df = pd.DataFrame([
        {
            "sample_id": "b_test_01",
            "subject_id": "patient_01",
            "image_path": "dummy.png",
            "class": "swelling",
            "mask_path": "dummy_mask.png",
            "source": "Kaggle",
            "license": "CC-BY",
            "split": "test"
        },
        {
            "sample_id": "b_test_02",
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

    # 3. Execute comparative benchmark pipeline
    runner = ResearchBenchmarkRunner(dataset_dir=temp_weights_dir, manifest_path=manifest_path)
    report = runner.run_all_benchmarks(xgb_weights_path=xgb_path, vqc_model_dir=temp_weights_dir)
    
    # Validate result structures
    assert "ablation_study" in report
    assert "robustness_perturbations" in report
    assert "uncertainty_coverage" in report
    
    # Ablation checks
    assert len(report["ablation_study"]) == 4 # full, no_vision, no_questionnaire, no_sensor
    assert report["ablation_study"][0]["configuration"] == "full"
    assert "xgb_mcc" in report["ablation_study"][0]
    assert "vqc_accuracy" in report["ablation_study"][1]
    
    # Perturbations checks
    assert len(report["robustness_perturbations"]) == 5 # 5 noise levels
    assert report["robustness_perturbations"][0]["noise_level"] == 0.0
    assert "xgb_f1" in report["robustness_perturbations"][3]
    
    # Coverage checks
    assert "classical_xgb" in report["uncertainty_coverage"]
    assert "quantum_vqc" in report["uncertainty_coverage"]
    assert "coverage_ratio" in report["uncertainty_coverage"]["classical_xgb"]
    assert "mean_entropy" in report["uncertainty_coverage"]["quantum_vqc"]
    
    # 4. Verify persisted in MongoDB ablation_studies collection
    db = get_database()
    saved = db.ablation_studies.find_one()
    assert saved is not None
    assert len(saved["ablation_study"]) == 4
    assert saved["uncertainty_coverage"]["threshold"] == 0.70
