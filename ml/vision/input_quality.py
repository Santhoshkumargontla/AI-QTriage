"""Input quality and OOD gates for vision models.

These checks exist because trained heads overfit synthetic drawings and fire on
uniform/blank inputs. Gates are not a clinical filter.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

STATUS_VALID = "VALID"
STATUS_LOW_QUALITY = "LOW_QUALITY_INPUT"
STATUS_OOD = "OUT_OF_DISTRIBUTION"
STATUS_UNAVAILABLE = "MODEL_UNAVAILABLE"
STATUS_UNTRUSTWORTHY = "UNTRUSTWORTHY_OUTPUT"

ENV_MIN_CONF = "EFFNET_MIN_CONFIDENCE"
ENV_TEMPERATURE = "EFFNET_TEMPERATURE"
DEFAULT_MIN_CONFIDENCE = 0.80
DEFAULT_TEMPERATURE = 1.5


def resolve_min_confidence(explicit: Optional[float] = None) -> float:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(ENV_MIN_CONF)
    if raw is not None and str(raw).strip():
        return float(raw)
    return DEFAULT_MIN_CONFIDENCE


def resolve_temperature(explicit: Optional[float] = None) -> float:
    if explicit is not None:
        return max(float(explicit), 1e-3)
    raw = os.environ.get(ENV_TEMPERATURE)
    if raw is not None and str(raw).strip():
        return max(float(raw), 1e-3)
    return DEFAULT_TEMPERATURE


def _skin_and_red_fractions(image_rgb: np.ndarray) -> tuple[float, float]:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    skin = (
        (((h <= 25) | (h >= 160)) & (s >= 20) & (s <= 180) & (v >= 50) & (v <= 250))
        | ((h <= 40) & (s >= 15) & (s <= 120) & (v >= 80) & (v <= 230))
    )
    red = ((h <= 12) | (h >= 170)) & (s >= 80) & (v >= 40)
    n = float(image_rgb.shape[0] * image_rgb.shape[1])
    return float(skin.mean()) if n else 0.0, float(red.mean()) if n else 0.0


def assess_input_quality(image_rgb: np.ndarray) -> dict:
    """Return status, reason, and quality metrics. Does not run the classifier."""
    if image_rgb is None:
        return {
            "status": STATUS_LOW_QUALITY,
            "reason": "missing_image",
            "metrics": {},
        }
    arr = np.asarray(image_rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return {
            "status": STATUS_LOW_QUALITY,
            "reason": "invalid_image_shape",
            "metrics": {"shape": list(arr.shape)},
        }
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h < 16 or w < 16:
        return {
            "status": STATUS_LOW_QUALITY,
            "reason": "image_too_small",
            "metrics": {"height": h, "width": w},
        }

    rgb = arr[:, :, :3]
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating) and float(np.nanmax(rgb)) <= 1.5:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    std = float(np.std(gray))
    ptp = float(int(gray.max()) - int(gray.min()))
    unique16 = int(np.unique(gray // 16).size)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 50, 150)
    edge_frac = float((edges > 0).mean())
    skin_frac, red_frac = _skin_and_red_fractions(rgb)

    metrics = {
        "height": h,
        "width": w,
        "std": round(std, 4),
        "ptp": round(ptp, 4),
        "unique16": unique16,
        "laplacian_var": round(lap_var, 4),
        "edge_frac": round(edge_frac, 6),
        "skin_frac": round(skin_frac, 6),
        "red_frac": round(red_frac, 6),
    }

    if std < 12.0 and ptp < 40:
        return {
            "status": STATUS_LOW_QUALITY,
            "reason": "uniform_or_blank_image",
            "metrics": metrics,
        }
    if unique16 <= 3 and std < 20.0 and lap_var < 50.0:
        return {
            "status": STATUS_LOW_QUALITY,
            "reason": "near_uniform_palette",
            "metrics": metrics,
        }
    if lap_var < 12.0 and edge_frac < 0.012:
        return {
            "status": STATUS_LOW_QUALITY,
            "reason": "blank_or_blurred_insufficient_detail",
            "metrics": metrics,
        }
    if lap_var < 25.0 and std < 18.0 and edge_frac < 0.02:
        return {
            "status": STATUS_LOW_QUALITY,
            "reason": "low_contrast_blur",
            "metrics": metrics,
        }
    if lap_var > 5000.0 and edge_frac > 0.20 and skin_frac < 0.35:
        return {
            "status": STATUS_OOD,
            "reason": "high_frequency_unstructured",
            "metrics": metrics,
        }
    if skin_frac < 0.04 and red_frac < 0.06:
        return {
            "status": STATUS_OOD,
            "reason": "colors_not_injury_like",
            "metrics": metrics,
        }
    return {
        "status": STATUS_VALID,
        "reason": "passed_input_quality_gates",
        "metrics": metrics,
    }


def softmax_entropy(probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())
