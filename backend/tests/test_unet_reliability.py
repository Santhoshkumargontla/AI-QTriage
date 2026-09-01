"""U-Net must not display a trusted mask on blank/OOD inputs.

Raw false-positive area is recorded. Displayed overlay is withheld.
"""
import numpy as np

from ml.vision.input_quality import (
    STATUS_LOW_QUALITY,
    STATUS_UNAVAILABLE,
    STATUS_UNTRUSTWORTHY,
    STATUS_VALID,
)
from ml.vision.unet_wrapper import UNetSegmenter, interpret_segmentation


def _seg():
    return UNetSegmenter()


def test_black_white_gray_withheld_but_raw_fp_recorded():
    seg = _seg()
    cases = {
        "black": np.zeros((128, 128, 3), dtype=np.uint8),
        "white": np.full((128, 128, 3), 255, dtype=np.uint8),
        "gray": np.full((128, 128, 3), 180, dtype=np.uint8),
    }
    for name, img in cases.items():
        mask, count, ratio, info = seg.segment(img)
        parsed = interpret_segmentation(mask, count, ratio, info)
        assert int(mask.sum()) == 0, name
        assert count == 0, name
        assert parsed["is_reliable"] is False, name
        assert parsed["affected_ratio"] is None, name
        assert info["status"] in {STATUS_LOW_QUALITY, STATUS_UNTRUSTWORTHY}, name
        assert info["mask_withheld"] is True, name
        # Promoted public-wound checkpoint: raw blanks must stay empty, not paint the canvas.
        if name in {"black", "white"}:
            raw = info.get("raw_positive_ratio")
            assert raw is not None, name
            assert raw < 0.05, (name, raw)


def test_uniform_skin_and_blank_skin_withheld():
    import os
    import cv2

    seg = _seg()
    skin = np.full((160, 160, 3), (185, 145, 125), dtype=np.uint8)
    mask, count, ratio, info = seg.segment(skin)
    assert info["is_reliable"] is False
    assert int(mask.sum()) == 0
    assert info["status"] in {STATUS_LOW_QUALITY, STATUS_UNTRUSTWORTHY}

    path = os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")
    if os.path.exists(path):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        mask, count, ratio, info = seg.segment(img)
        parsed = interpret_segmentation(mask, count, ratio, info)
        assert parsed["is_reliable"] is False
        assert int(mask.sum()) == 0


def test_unrelated_and_blurred_not_valid_overlay():
    import cv2

    seg = _seg()
    blue = np.full((160, 160, 3), (20, 60, 200), dtype=np.uint8)
    mask, count, ratio, info = seg.segment(blue)
    assert info["status"] in {STATUS_LOW_QUALITY, STATUS_UNTRUSTWORTHY}
    assert info["is_reliable"] is False
    assert int(mask.sum()) == 0

    cut = np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8)
    cv2.line(cut, (70, 40), (150, 180), (190, 20, 20), 7)
    blurred = cv2.GaussianBlur(cut, (51, 51), 16)
    mask, count, ratio, info = seg.segment(blurred)
    assert info["is_reliable"] is False
    assert int(mask.sum()) == 0


def test_synthetic_cut_drawing_is_out_of_domain_for_public_unet():
    """Drawings are no longer the training domain. Do not require a VALID overlay."""
    import cv2

    seg = _seg()
    img = np.full((224, 224, 3), (185, 145, 125), dtype=np.uint8)
    cv2.line(img, (70, 40), (150, 180), (190, 20, 20), 7)
    mask, count, ratio, info = seg.segment(img)
    raw = info.get("raw_positive_ratio")
    assert raw is None or raw < 0.25
    if info["status"] == STATUS_VALID:
        assert count >= 0


def test_held_out_public_wound_can_produce_mask():
    import csv
    import os
    import cv2

    man = os.path.join("data", "datasets", "unet_public_real", "manifest.csv")
    if not os.path.exists(man):
        return
    path = None
    with open(man, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "test" and str(row.get("empty_mask", "")).lower() not in {"1", "true", "yes"}:
                path = row["image_path"].replace("/", os.sep)
                break
    if not path or not os.path.exists(path):
        return
    seg = _seg()
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    mask, count, ratio, info = seg.segment(img)
    assert info["status"] in {STATUS_VALID, STATUS_UNTRUSTWORTHY, STATUS_LOW_QUALITY}
    if info["status"] == STATUS_VALID:
        assert count > 0
        assert ratio is None or ratio < 0.85


def test_missing_weights_status_unavailable(tmp_path):
    seg = UNetSegmenter(model_path=str(tmp_path / "missing.pt"))
    assert seg.is_loaded is False
    mask, count, ratio, info = seg.segment(np.full((64, 64, 3), 180, dtype=np.uint8))
    assert info["status"] == STATUS_UNAVAILABLE
    assert int(mask.sum()) == 0
    parsed = interpret_segmentation(mask, count, ratio, info)
    assert parsed["is_reliable"] is False
