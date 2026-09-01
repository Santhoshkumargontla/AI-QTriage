import cv2
import numpy as np
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageQualityError(ValueError):
    """Raised when an image fails quality checking for research analysis."""
    pass

MODEL_INPUT_SIZES = {
    "yolo": (640, 640),
    "unet": (256, 256),
    "efficientnet": (224, 224),
    "default": (224, 224)
}

def verify_image_quality(image_path: str, min_width: int = 100, min_height: int = 100,
                         min_area: int = 10000, blur_threshold: float = 12.0,
                         min_brightness: float = 40.0, max_brightness: float = 230.0, 
                         min_contrast: float = 15.0) -> dict:
    """
    Performs standard OpenCV-based quality checks on the uploaded image.
    Supports rectangular images by evaluating min dimensions and total area instead of rigid square constraints.
    Raises ImageQualityError if the image is blurry, corrupted, too dark/bright, or low contrast.
    Returns a dictionary of computed quality metrics.
    """
    if not os.path.exists(image_path):
        raise ImageQualityError("Image file does not exist.")

    # 1. Integrity & Format Check
    try:
        with Image.open(image_path) as img:
            img.verify()
    except (OSError, ValueError, Image.UnidentifiedImageError) as e:
        raise ImageQualityError(f"Image file is corrupted or unreadable. Error: {str(e)}")

    # Read image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ImageQualityError("Image format could not be decoded by OpenCV.")

    height, width, channels = img.shape

    # 2. Resolution / Minimum Information Check
    if height < min_height or width < min_width or (height * width) < min_area:
        raise ImageQualityError(
            f"Image resolution is too small ({width}x{height}). Minimum required: {min_width}x{min_height} with at least {min_area} total pixels."
        )

    # Convert to grayscale for statistical metrics
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Brightness Check (Grayscale Mean)
    mean_brightness = gray.mean()
    if mean_brightness < min_brightness:
        raise ImageQualityError(
            f"Image quality is insufficient for reliable research inference (image is too dark. Mean: {mean_brightness:.1f}, Threshold: {min_brightness})."
        )
    if mean_brightness > max_brightness:
        raise ImageQualityError(
            f"Image quality is insufficient for reliable research inference (image is overexposed/too bright. Mean: {mean_brightness:.1f}, Threshold: {max_brightness})."
        )

    # 4. Blur Check (Laplacian Variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        raise ImageQualityError(
            f"Image quality is insufficient for reliable research inference (image is too blurry. Variance: {laplacian_var:.2f}, Threshold: {blur_threshold})."
        )

    # 5. Contrast Check (Grayscale Standard Deviation)
    std_contrast = gray.std()
    if std_contrast < min_contrast:
        raise ImageQualityError(
            f"Image quality is insufficient for reliable research inference (contrast is too low. StdDev: {std_contrast:.1f}, Threshold: {min_contrast})."
        )

    return {
        "width": width,
        "height": height,
        "aspect_ratio_preserved": True,
        "preprocessing_note": "Image dimensions will be adjusted automatically while preserving original aspect ratio using letterbox padding.",
        "status_message": "✓ Image prepared for AI analysis",
        "blur_variance": round(float(laplacian_var), 2),
        "mean_brightness": round(float(mean_brightness), 2),
        "std_contrast": round(float(std_contrast), 2)
    }

def letterbox_image(img_rgb: np.ndarray, target_size: tuple = (224, 224), pad_color: tuple = (128, 128, 128)) -> tuple:
    """
    Resizes an image while preserving its aspect ratio and pads it with a neutral color to fit target_size (height, width).
    Returns (padded_img_rgb, transformation_metadata).
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    target_h, target_w = target_size
    orig_h, orig_w = img_rgb.shape[:2]

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    metadata = {
        "original_width": orig_w,
        "original_height": orig_h,
        "target_width": target_w,
        "target_height": target_h,
        "resized_width": new_w,
        "resized_height": new_h,
        "scale": float(scale),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
        "aspect_ratio_preserved": True,
        "preprocessing_version": "img-v2-letterbox"
    }
    return padded, metadata

def map_bbox_model_to_orig(bbox_padded: list, metadata: dict) -> list:
    """
    Converts bounding box [x1, y1, x2, y2] from padded model space to original image space.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_padded]
    scale = metadata.get("scale", 1.0)
    pad_x = metadata.get("pad_x", 0)
    pad_y = metadata.get("pad_y", 0)
    orig_w = metadata.get("original_width", 224)
    orig_h = metadata.get("original_height", 224)

    orig_x1 = max(0.0, min(float(orig_w), (x1 - pad_x) / scale))
    orig_y1 = max(0.0, min(float(orig_h), (y1 - pad_y) / scale))
    orig_x2 = max(0.0, min(float(orig_w), (x2 - pad_x) / scale))
    orig_y2 = max(0.0, min(float(orig_h), (y2 - pad_y) / scale))

    return [round(orig_x1, 1), round(orig_y1, 1), round(orig_x2, 1), round(orig_y2, 1)]

def map_bbox_orig_to_model(bbox_orig: list, metadata: dict) -> list:
    """
    Converts bounding box [x1, y1, x2, y2] from original image space to padded model space.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_orig]
    scale = metadata.get("scale", 1.0)
    pad_x = metadata.get("pad_x", 0)
    pad_y = metadata.get("pad_y", 0)
    target_w = metadata.get("target_width", 224)
    target_h = metadata.get("target_height", 224)

    mod_x1 = max(0.0, min(float(target_w), x1 * scale + pad_x))
    mod_y1 = max(0.0, min(float(target_h), y1 * scale + pad_y))
    mod_x2 = max(0.0, min(float(target_w), x2 * scale + pad_x))
    mod_y2 = max(0.0, min(float(target_h), y2 * scale + pad_y))

    return [round(mod_x1, 1), round(mod_y1, 1), round(mod_x2, 1), round(mod_y2, 1)]

def preprocess_image_for_inference(image_path: str, target_size: tuple = (224, 224)) -> tuple:
    """
    Reads, applies letterbox padding to preserve aspect ratio, and normalizes image for PyTorch model input.
    Returns (preprocessed_tensor, padded_image_rgb, metadata).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("Could not read image file.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    padded_rgb, metadata = letterbox_image(img_rgb, target_size=target_size)

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    transformed = transform(image=padded_rgb)
    tensor = transformed["image"]

    metadata["normalized"] = True
    return tensor, padded_rgb, metadata
