import os
import pytest
import numpy as np
import torch
import pandas as pd
import cv2
from unittest.mock import MagicMock, patch
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
from ml.vision.unet_wrapper import UNetSegmenter
from ml.vision.yolo_wrapper import YOLO11Detector
from ml.explainability.grad_cam import GradCAMExplain
from ml.evaluation.cv_evaluator import CVEvaluator, compute_segmentation_metrics
from backend.database.connection import get_database

@pytest.fixture
def temp_weights_dir(tmpdir):
    return str(tmpdir)

def test_grad_cam_hook_and_overlay(temp_weights_dir):
    """Verify Grad-CAM registers hooks and generates colored attention overlays."""
    weights_path = os.path.join(temp_weights_dir, "effnet_temp.pt")
    
    # Initialize classifier and save temp state dict
    classifier_init = EfficientNetV2Classifier()
    torch.save(classifier_init.model.state_dict(), weights_path)
    
    # Load and setup Grad-CAM
    classifier = EfficientNetV2Classifier(model_path=weights_path)
    gcam = GradCAMExplain(classifier)
    
    # Generate dummy image
    dummy_img = np.random.randint(100, 200, (300, 400, 3), dtype=np.uint8)
    
    # Run heat mapping
    heatmap, color_heatmap, overlay = gcam.generate_heatmap(dummy_img, target_class_idx=0)
    
    # Assert correct dimensions (should match original input dimensions)
    assert heatmap.shape == (300, 400)
    assert color_heatmap.shape == (300, 400, 3)
    assert overlay.shape == (300, 400, 3)
    
    # Verify values are bounded appropriately
    assert heatmap.min() >= 0
    assert heatmap.max() <= 255

def test_cv_evaluation_metrics_pipeline(temp_weights_dir):
    """Test that CVEvaluator processes the test split and saves metrics in MongoDB."""
    manifest_path = os.path.join(temp_weights_dir, "manifest.csv")
    
    # Create test image
    img_path = os.path.join(temp_weights_dir, "img_test.png")
    img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    cv2.imwrite(img_path, img)
    
    # Create test mask
    mask_path = os.path.join(temp_weights_dir, "mask_test.png")
    mask = np.zeros((224, 224), dtype=np.uint8)
    cv2.circle(mask, (112, 112), 40, 255, -1)
    cv2.imwrite(mask_path, mask)
    
    # Create test CSV manifest
    df = pd.DataFrame([{
        "sample_id": "test_id_01",
        "subject_id": "patient_test",
        "image_path": "img_test.png",
        "class": "swelling",
        "mask_path": "mask_test.png",
        "source": "Kaggle",
        "license": "CC-BY",
        "split": "test"  # In the test split to be evaluated
    }])
    df.to_csv(manifest_path, index=False)
    
    # Generate temp weight files for the models
    yolo_weights = os.path.join(temp_weights_dir, "yolo_temp.pt")
    unet_weights = os.path.join(temp_weights_dir, "unet_temp.pt")
    effnet_weights = os.path.join(temp_weights_dir, "effnet_temp.pt")
    
    # Mock weights save
    unet_init = UNetSegmenter()
    torch.save(unet_init.model.state_dict(), unet_weights)
    
    effnet_init = EfficientNetV2Classifier()
    torch.save(effnet_init.model.state_dict(), effnet_weights)
    
    # We patch YOLO11 load and detect in the evaluator to avoid file requirements
    with patch("ml.vision.yolo_wrapper.YOLO") as mock_yolo_cls:
        mock_yolo = MagicMock()
        mock_yolo.task = "detect"
        mock_yolo_cls.return_value = mock_yolo
        mock_yolo.names = {0: "cut", 1: "bruise", 2: "swelling", 3: "other"}
        
        # Configure mock YOLO results
        box = MagicMock()
        box.cls = torch.tensor([2.0])  # Swelling
        box.conf = torch.tensor([0.90])
        box.xyxy = [torch.tensor([50.0, 50.0, 150.0, 150.0])]
        
        result = MagicMock()
        result.boxes = [box]
        mock_yolo.return_value = [result]
        
        # Write dummy file for YOLO load mock
        with open(yolo_weights, "w") as f:
            f.write("mock yolo weights")
            
        # Run evaluation
        evaluator = CVEvaluator(dataset_dir=temp_weights_dir, manifest_path=manifest_path)
        metrics = evaluator.evaluate_all(
            yolo_weights=yolo_weights, 
            unet_weights=unet_weights, 
            effnet_weights=effnet_weights
        )
        
        # Check that metrics were generated
        assert "yolo_metrics" in metrics
        assert "unet_metrics" in metrics
        assert "efficientnet_metrics" in metrics
        assert "mAP50" in metrics["yolo_metrics"]
        assert "mean_dice" in metrics["unet_metrics"]
        assert "accuracy" in metrics["efficientnet_metrics"]
        
        # Verify saved in MongoDB
        db = get_database()
        saved = db.model_evaluations.find_one()
        assert saved is not None
        assert saved["yolo_metrics"]["mAP50"] == metrics["yolo_metrics"]["mAP50"]
