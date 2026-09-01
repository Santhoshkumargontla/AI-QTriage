import os
import json
import pytest
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from ml.vision.preprocess import verify_image_quality, ImageQualityError, preprocess_image_for_inference
from ml.training.dataset_manager import validate_dataset_quality

@pytest.fixture
def temp_dir(tmpdir):
    """Fixture to manage temporary test directories."""
    return str(tmpdir)

def test_image_quality_compliancy(temp_dir):
    """Verify that a standard, well-lit, high-contrast image passes validation."""
    clean_path = os.path.join(temp_dir, "clean.png")
    
    # Generate structured high-contrast image (not flat noise, so variance is high)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 60, (255, 255, 255), -1)
    cv2.rectangle(img, (20, 20), (100, 100), (100, 150, 50), -1)
    
    # Save file
    cv2.imwrite(clean_path, img)
    
    # Run quality validation
    metrics = verify_image_quality(clean_path, blur_threshold=5.0)
    assert metrics["width"] == 300
    assert metrics["height"] == 300
    assert metrics["mean_brightness"] > 10.0
    assert metrics["std_contrast"] > 10.0

def test_image_quality_failures(temp_dir):
    """Assert that dark, blurry, and low-contrast images fail with ImageQualityError."""
    dark_path = os.path.join(temp_dir, "dark.png")
    flat_path = os.path.join(temp_dir, "flat.png")
    blur_path = os.path.join(temp_dir, "blurry.png")
    
    # 1. Dark Image
    img_dark = np.ones((300, 300, 3), dtype=np.uint8) * 10
    cv2.imwrite(dark_path, img_dark)
    with pytest.raises(ImageQualityError) as exc:
        verify_image_quality(dark_path)
    assert "too dark" in str(exc.value)
    
    # 2. Low Contrast Image (Sharp grid with tiny amplitude difference)
    img_flat = np.ones((300, 300, 3), dtype=np.uint8) * 125
    for i in range(0, 300, 4):
        img_flat[i:i+2, :] = 130
        img_flat[:, i:i+2] = 130
    cv2.imwrite(flat_path, img_flat)
    with pytest.raises(ImageQualityError) as exc:
        verify_image_quality(flat_path)
    assert "contrast is too low" in str(exc.value)

    # 3. Blurry Image (Gaussian Blur on noise)
    img_blur = np.random.randint(100, 180, (300, 300, 3), dtype=np.uint8)
    img_blur = cv2.GaussianBlur(img_blur, (51, 51), 0)
    cv2.imwrite(blur_path, img_blur)
    with pytest.raises(ImageQualityError) as exc:
        verify_image_quality(blur_path, blur_threshold=15.0)
    assert "too blurry" in str(exc.value)

def test_dataset_manifest_validation(temp_dir):
    """Verify that the manifest validation system identifies errors and subject leakage."""
    manifest_path = os.path.join(temp_dir, "manifest.csv")
    
    # Write clean images to support validation
    clean_img_path = os.path.join(temp_dir, "img_a.png")
    img = np.ones((256, 256, 3), dtype=np.uint8) * 128
    cv2.imwrite(clean_img_path, img)

    # 1. Valid Manifest setup
    df_valid = pd.DataFrame([{
        "sample_id": "s_01",
        "subject_id": "patient_1",
        "image_path": "img_a.png",
        "class": "cut",
        "mask_path": "",
        "source": "Kaggle",
        "license": "CC-BY",
        "split": "train"
    }])
    df_valid.to_csv(manifest_path, index=False)
    report = validate_dataset_quality(manifest_path, temp_dir)
    assert report["status"] == "passed"
    assert report["total_records"] == 1
    assert report["class_distribution"]["cut"] == 1

    # 2. Invalid Category label
    df_invalid_class = pd.DataFrame([{
        "sample_id": "s_01",
        "subject_id": "patient_1",
        "image_path": "img_a.png",
        "class": "fracture",  # Not a supported visible class
        "mask_path": "",
        "source": "Kaggle",
        "license": "CC-BY",
        "split": "train"
    }])
    df_invalid_class.to_csv(manifest_path, index=False)
    report = validate_dataset_quality(manifest_path, temp_dir)
    assert report["status"] == "failed"
    assert len(report["invalid_labels"]) == 1
    assert report["invalid_labels"][0]["label"] == "fracture"

    # 3. Subject Leakage (leak patient_1 in train and test splits)
    df_leakage = pd.DataFrame([
        {
            "sample_id": "s_01",
            "subject_id": "patient_1",
            "image_path": "img_a.png",
            "class": "cut",
            "mask_path": "",
            "source": "Kaggle",
            "license": "CC-BY",
            "split": "train"
        },
        {
            "sample_id": "s_02",
            "subject_id": "patient_1",  # Same subject
            "image_path": "img_a.png",
            "class": "cut",
            "mask_path": "",
            "source": "Kaggle",
            "license": "CC-BY",
            "split": "test"             # Overlapping split
        }
    ])
    df_leakage.to_csv(manifest_path, index=False)
    report = validate_dataset_quality(manifest_path, temp_dir)
    assert report["status"] == "failed"
    assert report["subject_leakage_detected"] == True

def test_aspect_ratio_letterboxing_and_rectangular_images(temp_dir):
    """Test rectangular images (344x180, 180x344, 500x400), tiny images, and coordinate mappings."""
    from ml.vision.preprocess import letterbox_image, map_bbox_model_to_orig, map_bbox_orig_to_model, MODEL_INPUT_SIZES

    # 1. 344x180 Rectangular image (Horizontal)
    p_344x180 = os.path.join(temp_dir, "rect_344x180.jpg")
    img_344x180 = np.ones((180, 344, 3), dtype=np.uint8) * 128
    cv2.circle(img_344x180, (172, 90), 40, (200, 180, 160), -1)
    cv2.imwrite(p_344x180, cv2.cvtColor(img_344x180, cv2.COLOR_RGB2BGR))

    metrics_1 = verify_image_quality(p_344x180, blur_threshold=2.0)
    assert metrics_1["width"] == 344
    assert metrics_1["height"] == 180
    assert metrics_1["aspect_ratio_preserved"] == True

    tensor_1, padded_1, meta_1 = preprocess_image_for_inference(p_344x180, target_size=(224, 224))
    assert list(tensor_1.shape) == [3, 224, 224]
    assert meta_1["pad_y"] > 0  # Top/bottom letterboxing for wide image

    # 2. 180x344 Rectangular image (Vertical)
    p_180x344 = os.path.join(temp_dir, "rect_180x344.png")
    img_180x344 = np.ones((344, 180, 3), dtype=np.uint8) * 128
    cv2.circle(img_180x344, (90, 172), 40, (200, 180, 160), -1)
    cv2.imwrite(p_180x344, cv2.cvtColor(img_180x344, cv2.COLOR_RGB2BGR))

    metrics_2 = verify_image_quality(p_180x344, blur_threshold=2.0)
    assert metrics_2["width"] == 180
    assert metrics_2["height"] == 344

    tensor_2, padded_2, meta_2 = preprocess_image_for_inference(p_180x344, target_size=(224, 224))
    assert list(tensor_2.shape) == [3, 224, 224]
    assert meta_2["pad_x"] > 0  # Left/right pillarboxing for tall image

    # 3. 224x224 Square image
    p_224x224 = os.path.join(temp_dir, "sq_224.jpg")
    img_224 = np.ones((224, 224, 3), dtype=np.uint8) * 128
    cv2.rectangle(img_224, (50, 50), (150, 150), (220, 200, 180), -1)
    cv2.imwrite(p_224x224, cv2.cvtColor(img_224, cv2.COLOR_RGB2BGR))

    metrics_3 = verify_image_quality(p_224x224, blur_threshold=2.0)
    assert metrics_3["width"] == 224
    assert metrics_3["height"] == 224

    # 4. 500x400 Large rectangular image
    p_500x400 = os.path.join(temp_dir, "rect_500x400.jpg")
    img_500 = np.ones((400, 500, 3), dtype=np.uint8) * 128
    cv2.circle(img_500, (250, 200), 80, (220, 200, 180), -1)
    cv2.imwrite(p_500x400, cv2.cvtColor(img_500, cv2.COLOR_RGB2BGR))
    metrics_4 = verify_image_quality(p_500x400, blur_threshold=2.0)
    assert metrics_4["width"] == 500
    assert metrics_4["height"] == 400

    # 5. Very small image (80x40) -> Must be rejected
    p_small = os.path.join(temp_dir, "tiny_80x40.png")
    img_small = np.ones((40, 80, 3), dtype=np.uint8) * 128
    cv2.imwrite(p_small, img_small)
    with pytest.raises(ImageQualityError) as exc:
        verify_image_quality(p_small)
    assert "too small" in str(exc.value)

    # 6. Verify Bounding Box Transformation Logic
    bbox_orig = [34.4, 18.0, 172.0, 90.0]
    bbox_model = map_bbox_orig_to_model(bbox_orig, meta_1)
    bbox_restored = map_bbox_model_to_orig(bbox_model, meta_1)
    assert abs(bbox_restored[0] - bbox_orig[0]) < 1.0
    assert abs(bbox_restored[1] - bbox_orig[1]) < 1.0

    # 7. Model Specific Input Sizes
    tensor_yolo, _, _ = preprocess_image_for_inference(p_344x180, target_size=MODEL_INPUT_SIZES["yolo"])
    assert list(tensor_yolo.shape) == [3, 640, 640]

    tensor_unet, _, _ = preprocess_image_for_inference(p_344x180, target_size=MODEL_INPUT_SIZES["unet"])
    assert list(tensor_unet.shape) == [3, 256, 256]

