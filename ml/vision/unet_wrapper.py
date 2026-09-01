import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from ml.vision.input_quality import (
    STATUS_LOW_QUALITY,
    STATUS_UNAVAILABLE,
    STATUS_UNTRUSTWORTHY,
    STATUS_VALID,
    assess_input_quality,
)
from ml.vision.segmentation_quality import (
    assess_mask_sanity,
    map_input_status,
    postprocess_mask,
    resolve_max_area,
    resolve_threshold,
)


class UNetSegmenter:
    """Wrapper for U-Net visible injury region segmentation."""

    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = False
        self.model_path = model_path
        self.threshold = resolve_threshold()
        self.max_area = resolve_max_area()

        from ml.models.canonical_paths import UNET_CANONICAL, resolve_existing

        if model_path:
            located = resolve_existing(model_path)
            if os.path.exists(located):
                self.load_model(located)
            return

        canonical = resolve_existing(UNET_CANONICAL)
        if os.path.exists(canonical):
            self.load_model(canonical)

    def load_model(self, model_path: str):
        """Loads U-Net state dictionary weights."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"U-Net weights not found at: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        self.model_path = model_path
        print(f"U-Net segmentation model successfully loaded from {model_path}")

    def _empty_mask(self, h: int, w: int) -> np.ndarray:
        return np.zeros((max(h, 0), max(w, 0)), dtype=np.uint8)

    def _info(
        self,
        *,
        h: int,
        w: int,
        roi_h: int,
        roi_w: int,
        status: str,
        reason: str,
        display_message: str,
        pixel_count: int = 0,
        total_pixels: int = 0,
        affected_ratio: float = 0.0,
        raw_min: float = 0.0,
        raw_max: float = 0.0,
        raw_mean: float = 0.0,
        raw_positive_ratio: float = 0.0,
        raw_positive_pixels: int = 0,
        threshold: float = 0.5,
        mask_withheld: bool = True,
        is_reliable: bool = False,
        quality: dict = None,
        bbox_used: bool = False,
        model_input_shape=None,
    ) -> dict:
        confidence_status = "confident" if is_reliable else "insufficient"
        return {
            "checkpoint_status": "loaded" if self.is_loaded else "uninitialized",
            "original_image_shape": f"{h}x{w}",
            "preprocessed_roi_shape": f"{roi_h}x{roi_w}",
            "model_input_shape": model_input_shape or [1, 3, 256, 256],
            "raw_output_min": round(float(raw_min), 4),
            "raw_output_max": round(float(raw_max), 4),
            "raw_output_mean": round(float(raw_mean), 4),
            "raw_positive_pixels": int(raw_positive_pixels),
            "raw_positive_ratio": round(float(raw_positive_ratio), 6),
            "false_positive_area": round(float(raw_positive_ratio), 6) if mask_withheld else 0.0,
            "threshold_used": round(float(threshold), 2),
            "positive_pixels": int(pixel_count),
            "total_pixels": int(total_pixels),
            "affected_area_percentage": round(float(affected_ratio) * 100, 2),
            "confidence_status": confidence_status,
            "status": status,
            "reason": reason,
            "is_reliable": bool(is_reliable),
            "mask_withheld": bool(mask_withheld),
            "trust_status": status,
            "display_message": display_message,
            "denominator_type": "detected_region" if bbox_used else "full_image",
            "quality": (quality or {}).get("metrics", {}),
            "max_reasonable_area": self.max_area,
        }

    def _forward_probs(self, roi_rgb: np.ndarray) -> np.ndarray:
        roi_resized = cv2.resize(roi_rgb, (256, 256))
        if roi_resized.dtype != np.uint8:
            if np.issubdtype(roi_resized.dtype, np.floating) and float(np.nanmax(roi_resized)) <= 1.5:
                roi_resized = np.clip(roi_resized * 255.0, 0, 255).astype(np.uint8)
            else:
                roi_resized = np.clip(roi_resized, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        return np.asarray(probs, dtype=np.float32)

    def segment_raw(self, image_rgb: np.ndarray, bbox: list = None) -> dict:
        """Model output without quality gates. For audits — do not display as a trusted mask."""
        if image_rgb is None or not self.is_loaded:
            return {
                "probs": None,
                "binary": None,
                "positive_ratio": None,
                "loaded": self.is_loaded,
            }
        roi = self._roi(image_rgb, bbox)[0]
        if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
            return {"probs": None, "binary": None, "positive_ratio": None, "loaded": True}
        probs = self._forward_probs(roi)
        binary = (probs > self.threshold).astype(np.uint8)
        return {
            "probs": probs,
            "binary": binary,
            "positive_ratio": float(binary.mean()),
            "mean_prob": float(probs.mean()),
            "max_prob": float(probs.max()),
            "min_prob": float(probs.min()),
            "loaded": True,
            "threshold": self.threshold,
        }

    def _roi(self, image_rgb: np.ndarray, bbox: list = None):
        h, w = image_rgb.shape[:2]
        if bbox:
            x1, y1, x2, y2 = [int(val) for val in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = image_rgb[y1:y2, x1:x2]
        else:
            roi = image_rgb
        return roi, h, w

    def segment(self, image_rgb: np.ndarray, bbox: list = None) -> tuple:
        """
        Runs U-Net segmentation on the ROI crop or full image.

        Displayed mask is empty unless status is VALID and the mask is sane.
        Raw positive area is always recorded so false-positive collapse is not hidden.

        Returns:
            mask_resized, pixel_count, affected_ratio, debug_info
        """
        bbox_used = bbox is not None
        if image_rgb is None:
            info = self._info(
                h=0, w=0, roi_h=0, roi_w=0,
                status=STATUS_LOW_QUALITY, reason="missing_image",
                display_message="Missing image — segmentation withheld.",
                bbox_used=bbox_used,
            )
            return self._empty_mask(0, 0), 0, 0.0, info

        arr = np.asarray(image_rgb)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        roi, h, w = self._roi(arr, bbox)
        roi_h, roi_w = (int(roi.shape[0]), int(roi.shape[1])) if roi.size else (0, 0)
        display_shape = (roi_h, roi_w) if roi_h > 0 and roi_w > 0 else (h, w)

        if not self.is_loaded:
            info = self._info(
                h=h, w=w, roi_h=roi_h, roi_w=roi_w,
                status=STATUS_UNAVAILABLE, reason="segmenter_weights_not_loaded",
                display_message="Segmentation model unavailable.",
                total_pixels=roi_h * roi_w,
                bbox_used=bbox_used,
            )
            return self._empty_mask(*display_shape), 0, 0.0, info

        if roi_h == 0 or roi_w == 0:
            info = self._info(
                h=h, w=w, roi_h=roi_h, roi_w=roi_w,
                status=STATUS_LOW_QUALITY, reason="empty_roi",
                display_message="Empty region of interest — segmentation withheld.",
                bbox_used=bbox_used,
            )
            return self._empty_mask(h, w), 0, 0.0, info

        quality = assess_input_quality(roi)
        input_status, input_reason = map_input_status(quality)

        probs = self._forward_probs(roi)
        raw_min = float(np.min(probs))
        raw_max = float(np.max(probs))
        raw_mean = float(np.mean(probs))
        raw_binary = (probs > self.threshold).astype(np.uint8)
        raw_positive_pixels = int(raw_binary.sum())
        raw_positive_ratio = float(raw_binary.mean())

        processed = postprocess_mask(raw_binary)
        sanity = assess_mask_sanity(probs, processed, max_area=self.max_area)

        if input_status != STATUS_VALID:
            status, reason = input_status, input_reason
        elif sanity["status"] != STATUS_VALID:
            status, reason = sanity["status"], sanity["reason"]
        elif sanity["empty"]:
            status, reason = STATUS_VALID, sanity["reason"]
        else:
            status, reason = STATUS_VALID, "passed_quality_and_mask_sanity"

        mask_ok = status == STATUS_VALID and not sanity["empty"]
        if mask_ok:
            mask_resized = cv2.resize(processed, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            pixel_count = int(np.sum(mask_resized))
            total_pixels = roi_h * roi_w
            affected_ratio = float(pixel_count) / total_pixels if total_pixels else 0.0
            is_reliable = pixel_count > 0
            mask_withheld = False
            display_message = "Segmentation mask identified."
        else:
            mask_resized = self._empty_mask(roi_h, roi_w)
            pixel_count = 0
            total_pixels = roi_h * roi_w
            affected_ratio = 0.0
            is_reliable = False
            mask_withheld = True
            if status == STATUS_LOW_QUALITY:
                display_message = f"Low-quality input — segmentation withheld ({reason})."
            elif status == STATUS_UNTRUSTWORTHY:
                display_message = (
                    f"Untrustworthy segmentation — overlay withheld ({reason}). "
                    f"Raw positive area {raw_positive_ratio:.3f} was not hidden."
                )
            elif sanity["empty"]:
                display_message = "Empty mask — no positive region identified."
            else:
                display_message = "Segmentation region not confidently identified."

        info = self._info(
            h=h, w=w, roi_h=roi_h, roi_w=roi_w,
            status=status, reason=reason, display_message=display_message,
            pixel_count=pixel_count, total_pixels=total_pixels,
            affected_ratio=affected_ratio,
            raw_min=raw_min, raw_max=raw_max, raw_mean=raw_mean,
            raw_positive_ratio=raw_positive_ratio,
            raw_positive_pixels=raw_positive_pixels,
            threshold=self.threshold,
            mask_withheld=mask_withheld, is_reliable=is_reliable,
            quality=quality, bbox_used=bbox_used,
            model_input_shape=[1, 3, 256, 256],
        )

        print("\n=== U-Net Segmentation Debug Output ===")
        print(f"  Checkpoint status: {info['checkpoint_status']}")
        print(f"  Status: {info['status']} ({info['reason']})")
        print(f"  Original image shape: {info['original_image_shape']}")
        print(f"  Preprocessed ROI shape: {info['preprocessed_roi_shape']}")
        print(f"  Raw U-Net min/max/mean: {info['raw_output_min']}/{info['raw_output_max']}/{info['raw_output_mean']}")
        print(f"  Raw positive ratio: {info['raw_positive_ratio']}")
        print(f"  Threshold used: {info['threshold_used']}")
        print(f"  Displayed positive pixels: {info['positive_pixels']} / {info['total_pixels']}")
        print(f"  Mask withheld: {info['mask_withheld']}")
        print("=======================================\n")

        return mask_resized, pixel_count, affected_ratio, info


def interpret_segmentation(mask, pixel_count, affected_ratio, debug_info: dict) -> dict:
    """Normalize wrapper output for API callers. Never promotes a non-VALID mask."""
    status = debug_info.get("status") or STATUS_UNAVAILABLE
    reliable = bool(debug_info.get("is_reliable")) and status == STATUS_VALID and int(pixel_count or 0) > 0
    return {
        "status": status,
        "reason": debug_info.get("reason"),
        "is_reliable": reliable,
        "mask_withheld": bool(debug_info.get("mask_withheld", not reliable)),
        "pixel_count": int(pixel_count or 0) if reliable else 0,
        "affected_ratio": float(affected_ratio) if reliable else None,
        "raw_positive_ratio": debug_info.get("raw_positive_ratio"),
        "false_positive_area": debug_info.get("false_positive_area"),
        "display_message": debug_info.get("display_message"),
        "confidence_status": "confident" if reliable else "insufficient",
    }
