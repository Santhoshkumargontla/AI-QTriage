"""Segmentation input gates, mask sanity, and post-processing.

U-Net was trained only on synthetic drawings with non-empty masks. Uniform
black/white fills therefore produce ~1.0 positive area. These checks withhold
the displayed overlay; they still record the raw false-positive area.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from ml.vision.input_quality import (
    STATUS_LOW_QUALITY,
    STATUS_OOD,
    STATUS_UNAVAILABLE,
    STATUS_UNTRUSTWORTHY,
    STATUS_VALID,
    assess_input_quality,
)

ENV_THRESHOLD = "UNET_MASK_THRESHOLD"
ENV_MAX_AREA = "UNET_MAX_MASK_AREA"
ENV_MIN_AREA = "UNET_MIN_MASK_AREA"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_AREA = 0.70
DEFAULT_MIN_AREA = 0.001


def resolve_threshold(explicit: Optional[float] = None) -> float:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(ENV_THRESHOLD)
    if raw is not None and str(raw).strip():
        return float(raw)
    return DEFAULT_THRESHOLD


def resolve_max_area(explicit: Optional[float] = None) -> float:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(ENV_MAX_AREA)
    if raw is not None and str(raw).strip():
        return float(raw)
    return DEFAULT_MAX_AREA


def resolve_min_area(explicit: Optional[float] = None) -> float:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(ENV_MIN_AREA)
    if raw is not None and str(raw).strip():
        return float(raw)
    return DEFAULT_MIN_AREA


def map_input_status(quality: dict) -> tuple[str, str]:
    """Map shared input-quality status onto the U-Net status vocabulary."""
    status = quality.get("status")
    reason = str(quality.get("reason") or "unspecified")
    if status == STATUS_VALID:
        return STATUS_VALID, reason
    if status == STATUS_LOW_QUALITY:
        return STATUS_LOW_QUALITY, reason
    if status == STATUS_OOD:
        return STATUS_UNTRUSTWORTHY, f"ood_{reason}"
    if status == STATUS_UNAVAILABLE:
        return STATUS_UNAVAILABLE, reason
    return STATUS_UNTRUSTWORTHY, reason


def postprocess_mask(binary: np.ndarray) -> np.ndarray:
    """Remove speckle. Keep the largest component only when it dominates."""
    mask = (np.asarray(binary) > 0).astype(np.uint8)
    if mask.size == 0 or int(mask.sum()) == 0:
        return mask
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if int(opened.sum()) == 0:
        return opened
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n <= 2:
        return opened
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    largest = int(areas.max())
    total = int(areas.sum())
    if total > 0 and (largest / total) >= 0.80:
        return (labels == largest_idx).astype(np.uint8)
    return opened


def assess_mask_sanity(
    probs: np.ndarray,
    binary: np.ndarray,
    *,
    max_area: Optional[float] = None,
    min_area: Optional[float] = None,
) -> dict:
    """Judge whether a predicted mask is an unreasonable collapse."""
    max_area = resolve_max_area(max_area)
    min_area = resolve_min_area(min_area)
    prob = np.asarray(probs, dtype=np.float64)
    mask = (np.asarray(binary) > 0).astype(np.uint8)
    n = int(mask.size) if mask.size else 1
    positive = int(mask.sum())
    area = float(positive) / float(n)
    raw_min = float(prob.min()) if prob.size else 0.0
    raw_max = float(prob.max()) if prob.size else 0.0
    raw_mean = float(prob.mean()) if prob.size else 0.0
    reasons = []
    if area > max_area:
        reasons.append("unreasonable_mask_area")
    if raw_mean >= 0.85 and area >= 0.50:
        reasons.append("saturated_positive_map")
    if raw_min >= 0.50:
        reasons.append("entire_map_above_threshold")
    empty = area < min_area or positive == 0
    if empty:
        status = STATUS_VALID
        reason = "empty_mask_no_positive_region"
    elif reasons:
        status = STATUS_UNTRUSTWORTHY
        reason = "+".join(reasons)
    else:
        status = STATUS_VALID
        reason = "mask_area_within_bounds"
    return {
        "status": status,
        "reason": reason,
        "empty": empty,
        "positive_pixels": positive,
        "positive_ratio": area,
        "max_area": max_area,
        "min_area": min_area,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_mean": raw_mean,
    }
