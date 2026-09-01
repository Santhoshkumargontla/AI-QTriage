import torch
import cv2
import numpy as np

from ml.vision.input_quality import (
    STATUS_LOW_QUALITY,
    STATUS_OOD,
    STATUS_UNAVAILABLE,
    STATUS_UNTRUSTWORTHY,
    STATUS_VALID,
)

GRADCAM_SOURCE_MODEL = "EfficientNetV2"
GRADCAM_LABEL = "MODEL VISUALIZATION"
GRADCAM_NOT_CLINICAL = "NOT CLINICAL EXPLANATION"
EXPLANATION_GENERATED = "GENERATED"
EXPLANATION_WITHHELD = "WITHHELD"
EXPLANATION_UNAVAILABLE = "MODEL_UNAVAILABLE"

_WITHHOLD_STATUSES = {
    STATUS_LOW_QUALITY,
    STATUS_OOD,
    STATUS_UNAVAILABLE,
    STATUS_UNTRUSTWORTHY,
    "NOT_TRUSTWORTHY",
    "MODEL_OUTPUT_NOT_TRUSTWORTHY",
}


def effnet_training_status(parsed: dict = None) -> str:
    """Training/readiness status for the production EfficientNet weights."""
    parsed = parsed or {}
    override = parsed.get("model_training_status")
    if override is not None and str(override).strip():
        return str(override)
    try:
        from ml.models.canonical_paths import EFFNET_METADATA, exists, read_json

        if not exists(EFFNET_METADATA):
            return "UNKNOWN"
        meta = read_json(EFFNET_METADATA) or {}
        return str(meta.get("training_status") or meta.get("status") or "UNKNOWN")
    except (OSError, TypeError, ValueError):
        return "UNKNOWN"


def training_status_blocks_gradcam(training_status: str) -> bool:
    text = str(training_status or "").upper()
    return "NOT_TRUSTWORTHY" in text or text in {s.upper() for s in _WITHHOLD_STATUSES}


def classifier_allows_gradcam(parsed: dict) -> bool:
    """Grad-CAM is allowed only for a VALID, confident prediction from a trustworthy model."""
    if not parsed:
        return False
    status = parsed.get("status")
    if status in _WITHHOLD_STATUSES or status != STATUS_VALID:
        return False
    if training_status_blocks_gradcam(effnet_training_status(parsed)):
        return False
    if not parsed.get("is_confident"):
        return False
    if not parsed.get("winner"):
        return False
    return True


def _metadata(
    parsed: dict,
    explanation_status: str,
    overlay_generated: bool,
    reason: str = None,
    model_status: str = None,
) -> dict:
    parsed = parsed or {}
    status = model_status or parsed.get("status") or STATUS_UNAVAILABLE
    return {
        "source_model": GRADCAM_SOURCE_MODEL,
        "predicted_class": parsed.get("winner") if explanation_status == EXPLANATION_GENERATED else None,
        "confidence": parsed.get("max_prob") if explanation_status == EXPLANATION_GENERATED else None,
        "model_status": status,
        "explanation_status": explanation_status,
        "gradcam_label": GRADCAM_LABEL,
        "gradcam_explanation": GRADCAM_NOT_CLINICAL,
        "gradcam_reliability": "NOT_CLINICAL_EXPLANATION",
        "overlay_generated": bool(overlay_generated),
        "withheld_reason": reason or parsed.get("reason"),
    }


def maybe_generate_gradcam(classifier, image_rgb: np.ndarray, parsed: dict) -> tuple:
    """
    Run Grad-CAM only when the classifier output is VALID and the model is not
    marked NOT_TRUSTWORTHY.

    Returns (overlay_or_none, metadata). Overlay is never generated for
    LOW_QUALITY_INPUT, OUT_OF_DISTRIBUTION, MODEL_UNAVAILABLE, or NOT_TRUSTWORTHY.
    Grad-CAM is a model visualization, not clinical evidence.
    """
    parsed = parsed or {}
    status = parsed.get("status") or STATUS_UNAVAILABLE
    if status == STATUS_UNAVAILABLE or not getattr(classifier, "is_loaded", False):
        return None, _metadata(parsed, EXPLANATION_UNAVAILABLE, False, parsed.get("reason") or "classifier_weights_not_loaded")
    training = effnet_training_status(parsed)
    if training_status_blocks_gradcam(training):
        return None, _metadata(
            parsed,
            EXPLANATION_WITHHELD,
            False,
            "classifier_model_not_trustworthy",
            model_status="NOT_TRUSTWORTHY",
        )
    if not classifier_allows_gradcam(parsed):
        return None, _metadata(parsed, EXPLANATION_WITHHELD, False)
    if image_rgb is None:
        return None, _metadata(parsed, EXPLANATION_WITHHELD, False, "missing_image")

    explainer = GradCAMExplain(classifier)
    winner = str(parsed["winner"]).lower()
    try:
        target_idx = classifier.classes.index(winner)
    except (ValueError, AttributeError):
        target_idx = None
    _heatmap, _color, overlay = explainer.generate_heatmap(image_rgb, target_class_idx=target_idx)
    return overlay, _metadata(parsed, EXPLANATION_GENERATED, True)


class GradCAMExplain:
    """Implements Grad-CAM visual attention mapping for EfficientNetV2."""
    
    def __init__(self, classifier_wrapper):
        self.wrapper = classifier_wrapper
        self.model = classifier_wrapper.model
        self.device = classifier_wrapper.device
        
        # We hook into the final conv layer of EfficientNetV2
        self.target_layer = None
        
        # In timm's EfficientNetV2-S, the final feature extractor is conv_head
        if hasattr(self.model, "conv_head"):
            self.target_layer = self.model.conv_head
        else:
            # Fallback to last block if conv_head is not present
            self.target_layer = list(self.model.children())[-2]
            
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, image_rgb: np.ndarray, target_class_idx: int = None) -> tuple:
        """
        Generates the Grad-CAM heatmap and blended overlay.

        For user-facing output, call maybe_generate_gradcam() so invalid
        classifier statuses cannot produce a misleading explanation.
        """
        h, w = image_rgb.shape[:2]
        
        # 1. Preprocess image for the model
        img_resized = cv2.resize(image_rgb, (224, 224))
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self.device)
        tensor.requires_grad_()

        # 2. Forward pass
        self.model.eval()
        self.model.zero_grad()
        outputs = self.model(tensor)
        
        if target_class_idx is None:
            # Default to winning class
            target_class_idx = int(torch.argmax(outputs, dim=1).item())
            
        score = outputs[0, target_class_idx]
        
        # 3. Backward pass to calculate gradients
        score.backward()

        # Retrieve hooked values
        gradients = self.gradients.cpu().data.numpy()[0]  # Shape [C, H, W]
        activations = self.activations.cpu().data.numpy()[0]  # Shape [C, H, W]

        # 4. Compute channel-wise weights (average gradients)
        weights = np.mean(gradients, axis=(1, 2))  # Shape [C]

        # 5. Compute weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w_i in enumerate(weights):
            cam += w_i * activations[i, :, :]

        # 6. Apply ReLU (keep positive activations only) and normalize
        cam = np.maximum(cam, 0)
        
        if cam.max() > 0:
            cam = cam / cam.max()
            
        # Resize CAM back to the original image dimensions
        cam_resized = cv2.resize(cam, (w, h))

        # Convert to 8-bit scale
        heatmap = np.uint8(255 * cam_resized)
        
        # Apply colormap using OpenCV
        color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend overlay: original_image * 0.6 + heatmap_image * 0.4
        overlay = cv2.addWeighted(image_rgb, 0.6, color_heatmap_rgb, 0.4, 0)

        return heatmap, color_heatmap_rgb, overlay
