"""OOD raw vs gated probes. Does not write metrics into model metadata."""
import os, sys, json
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier, interpret_prediction
from ml.vision.unet_wrapper import UNetSegmenter

eff = EfficientNetV2Classifier()
unet = UNetSegmenter()
out = {}
for name, arr in [
    ("gray", np.full((256, 256, 3), 128, np.uint8)),
    ("black", np.zeros((256, 256, 3), np.uint8)),
    ("white", np.full((256, 256, 3), 255, np.uint8)),
    ("portrait", np.full((400, 220, 3), 128, np.uint8)),
    ("landscape", np.full((180, 500, 3), 128, np.uint8)),
]:
    raw = eff.predict_raw(arr, temperature=1.0)
    gated = interpret_prediction(eff.predict(arr))
    raw_u = unet.segment_raw(arr)
    mask, pc, ratio, dbg = unet.segment(arr)
    out[name] = {
        "effnet_raw_winner": raw.get("winner"),
        "effnet_raw_max": raw.get("max_prob"),
        "effnet_raw_probs": raw.get("probs"),
        "effnet_gated_status": gated.get("status"),
        "effnet_gated_winner": gated.get("winner"),
        "unet_raw_positive_ratio": raw_u.get("positive_ratio"),
        "unet_raw_mean": raw_u.get("mean_prob"),
        "unet_raw_max": raw_u.get("max_prob"),
        "unet_display_ratio": float(ratio),
        "unet_status": dbg.get("status"),
        "unet_withheld": dbg.get("mask_withheld"),
        "unet_fp_area": dbg.get("false_positive_area"),
    }
print(json.dumps(out, indent=2))
