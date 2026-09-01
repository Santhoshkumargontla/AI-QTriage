import os
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from ml.vision.yolo_wrapper import YOLO11Detector
from ml.vision.unet_wrapper import UNetSegmenter
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier

@pytest.fixture
def temp_weights_dir(tmpdir):
    """Fixture to manage temporary weight files for testing."""
    return str(tmpdir)

def test_yolo_detection_mocking():
    """Verify YOLO11 wrapper filters classes and confidences using a mock model."""
    with patch("ml.vision.yolo_wrapper.YOLO") as mock_yolo_cls:
        # Configure mock YOLO instance
        mock_yolo = MagicMock()
        mock_yolo.task = "detect"
        mock_yolo.names = {0: "cut", 1: "bruise", 2: "wound"}
        mock_yolo_cls.return_value = mock_yolo
        
        # Configure mock detection boxes
        box_cut = MagicMock()
        box_cut.cls = torch.tensor([0.0])
        box_cut.conf = torch.tensor([0.85])
        box_cut.xyxy = [torch.tensor([10.0, 20.0, 100.0, 110.0])]
        
        box_fracture = MagicMock()
        box_fracture.cls = torch.tensor([3.0])
        box_fracture.conf = torch.tensor([0.95])
        box_fracture.xyxy = [torch.tensor([50.0, 50.0, 200.0, 200.0])]
        
        mock_results = MagicMock()
        mock_results.boxes = [box_cut, box_fracture]
        mock_yolo.return_value = [mock_results]
        
        # Instantiate detector
        detector = YOLO11Detector(conf_threshold=0.50)
        detector.model = mock_yolo
        
        findings = detector.detect("dummy_path.png")

        # Classes come from model.names. Fracture is not in names, so it is dropped.
        assert len(findings) == 1
        assert findings[0]["finding"] == "cut"
        assert findings[0]["confidence"] == 0.85
        assert findings[0]["bounding_box"] == [10.0, 20.0, 100.0, 110.0]
        assert findings[0]["low_confidence"] is False

def test_unet_segmentation_load_and_run(temp_weights_dir):
    """Verify U-Net segmenter loads weights and outputs correct mask dims and ratios."""
    weights_path = os.path.join(temp_weights_dir, "unet_temp.pt")
    
    # Instantiate a segmenter and serialize its state dict to simulate a trained checkpoint
    segmenter_init = UNetSegmenter()
    torch.save(segmenter_init.model.state_dict(), weights_path)
    
    # Reload via wrapper
    segmenter = UNetSegmenter(model_path=weights_path)
    assert segmenter.is_loaded is True
    
    # Run segmentation on dummy image
    dummy_img = np.random.randint(0, 255, (400, 500, 3), dtype=np.uint8)
    mask, pixel_count, ratio, debug_info = segmenter.segment(dummy_img, bbox=[50, 50, 200, 200])
    
    # Assert return formats
    # Size of mask must match bbox dimension: height=150, width=150
    assert mask.shape == (150, 150)
    assert isinstance(pixel_count, int)
    assert 0.0 <= ratio <= 1.0
    assert "raw_output_min" in debug_info
    assert "raw_output_max" in debug_info
    assert "threshold_used" in debug_info
    assert "status" in debug_info
    assert debug_info["status"] in {
        "VALID",
        "LOW_QUALITY_INPUT",
        "UNTRUSTWORTHY_OUTPUT",
        "MODEL_UNAVAILABLE",
    }

def test_efficientnet_classifier_load_and_run(temp_weights_dir):
    """Verify EfficientNetV2 classifier loads weights and outputs normalized softmax distributions."""
    weights_path = os.path.join(temp_weights_dir, "effnet_temp.pt")
    
    # Instantiate classifier and save state dict
    classifier_init = EfficientNetV2Classifier()
    torch.save(classifier_init.model.state_dict(), weights_path)
    
    # Reload via wrapper
    classifier = EfficientNetV2Classifier(model_path=weights_path)
    assert classifier.is_loaded is True
    
    # Run prediction on dummy image
    dummy_img = np.full((300, 300, 3), (185, 145, 125), dtype=np.uint8)
    import cv2
    cv2.line(dummy_img, (40, 40), (250, 250), (190, 20, 20), 8)
    probs = classifier.predict(dummy_img)

    # Filter out __ metadata keys (confidence gate metadata added in v2)
    class_probs = {k: v for k, v in probs.items() if not k.startswith("__")}

    # Assert outputs match the loaded checkpoint class count (reject-v2).
    assert len(class_probs) == len(classifier.classes)
    assert "Cut" in class_probs
    assert "Bruise" in class_probs
    assert "Normal" in class_probs or "OOD_Reject" in class_probs
    assert "Swelling" not in class_probs
    assert "Other" not in class_probs
    # Randomly-initialized head may abstain (Normal/OOD_Reject) or fire injury.
    assert probs["__status"] in {"VALID", "OUT_OF_DISTRIBUTION"}
    if probs["__is_confident"]:
        assert probs["__winner"] in {"Cut", "Bruise"}
    else:
        assert probs.get("__abstention_class") in {None, "Normal", "OOD_Reject"} or probs["__winner"] is None

    # Softmax probabilities should sum to approximately 1.0 when scores are shown
    shown = [v for v in class_probs.values() if v is not None]
    if shown:
        total_prob = sum(shown)
        assert pytest.approx(total_prob, abs=1e-4) == 1.0

    # Verify metadata keys are present
    assert "__is_confident" in probs
    assert "__max_prob" in probs
    assert "__winner" in probs
    assert "__low_confidence" in probs
    assert isinstance(probs["__is_confident"], bool)
    assert 0.0 <= probs["__max_prob"] <= 1.0