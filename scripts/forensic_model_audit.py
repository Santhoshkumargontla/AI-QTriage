"""
Forensic model audit script.
Inspects the actual YOLO11 and EfficientNet checkpoints,
prints class names, weights metadata, and runs dummy inference
on several test images to reveal what the models actually predict.
"""
import os, sys, json, hashlib, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"))

ROOT = Path(__file__).parent.parent
from ml.models.canonical_paths import YOLO_CANONICAL, EFFNET_CANONICAL

# ──────────────────────────────────────────────────────────────
# 1. YOLO11 checkpoint audit
# ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  YOLO11 CHECKPOINT AUDIT")
print("="*70)

yolo_path = ROOT / YOLO_CANONICAL
print(f"  Path:  {yolo_path}")
print(f"  Exists: {yolo_path.exists()}")
if yolo_path.exists():
    stat = yolo_path.stat()
    print(f"  Size:   {stat.st_size:,} bytes ({stat.st_size/1024:.1f} KB)")
    print(f"  Mtime:  {datetime.fromtimestamp(stat.st_mtime).isoformat()}")
    # MD5
    h = hashlib.md5(yolo_path.read_bytes()).hexdigest()
    print(f"  MD5:    {h}")

try:
    from ultralytics import YOLO
    yolo = YOLO(str(yolo_path))
    names = yolo.names
    print(f"\n  model.names ({len(names)} classes):")
    for idx, name in sorted(names.items()):
        print(f"    {idx}: {name}")
    task = getattr(yolo, 'task', 'unknown')
    print(f"  Task: {task}")
except Exception as e:
    print(f"  ERROR loading YOLO: {e}")

# ──────────────────────────────────────────────────────────────
# 2. EfficientNet checkpoint audit
# ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  EFFICIENTNET CHECKPOINT AUDIT")
print("="*70)

effnet_path = ROOT / EFFNET_CANONICAL
print(f"  Path:  {effnet_path}")
print(f"  Exists: {effnet_path.exists()}")
if effnet_path.exists():
    stat2 = effnet_path.stat()
    print(f"  Size:   {stat2.st_size:,} bytes ({stat2.st_size/1024:.1f} KB)")
    print(f"  Mtime:  {datetime.fromtimestamp(stat2.st_mtime).isoformat()}")
    h2 = hashlib.md5(effnet_path.read_bytes()).hexdigest()
    print(f"  MD5:    {h2}")

try:
    import torch
    sd = torch.load(str(effnet_path), map_location="cpu")
    if isinstance(sd, dict):
        print(f"  State dict keys (first 5): {list(sd.keys())[:5]}")
        # Find classifier head to infer output dimension
        for k in reversed(list(sd.keys())):
            if 'classifier' in k or 'head' in k or 'fc' in k:
                v = sd[k]
                print(f"  Classifier head key: {k}  shape: {v.shape}")
                break
    print("  State dict load: OK")
except Exception as e:
    print(f"  ERROR loading EfficientNet state dict: {e}")

try:
    from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
    eff = EfficientNetV2Classifier(str(effnet_path))
    print(f"\n  EfficientNetV2 classes (wrapper): {eff.classes}")
    print(f"  is_loaded: {eff.is_loaded}")
except Exception as e:
    print(f"  ERROR: {e}")

# ──────────────────────────────────────────────────────────────
# 3. Multi-image inference test
# ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  MULTI-IMAGE INFERENCE TEST")
print("="*70)

import numpy as np
import cv2

UPLOAD_DIR = ROOT / "uploads"
test_cases = []

# Synthetic images for testing
def make_test_img(name, description, pixel_fn):
    path = ROOT / f"data/test_images/{name}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    img = pixel_fn()
    cv2.imwrite(str(path), img)
    return str(path), description

def red_patch():  # simulate cut/wound: red region on skin-tone background
    img = np.ones((480, 640, 3), np.uint8) * np.array([180, 150, 130], np.uint8)
    img[180:260, 280:360] = [20, 20, 200]  # red patch (BGR)
    return img

def purple_patch():  # simulate bruise: dark purple region
    img = np.ones((480, 640, 3), np.uint8) * np.array([180, 150, 130], np.uint8)
    img[180:260, 280:360] = [120, 30, 90]  # purple (BGR)
    return img

def blank_skin():  # uniform skin tone, no injury
    return np.ones((480, 640, 3), np.uint8) * np.array([180, 150, 130], np.uint8)

def black_image():  # completely black
    return np.zeros((480, 640, 3), np.uint8)

def random_noise():  # random noise
    return np.random.randint(0, 256, (480, 640, 3), np.uint8)

def green_image():  # unrelated color — green
    return np.ones((480, 640, 3), np.uint8) * np.array([30, 200, 30], np.uint8)

test_cases = [
    make_test_img("test_red_patch",    "Synthetic cut (red patch on skin)", red_patch),
    make_test_img("test_purple_patch", "Synthetic bruise (purple on skin)", purple_patch),
    make_test_img("test_blank_skin",   "Blank skin tone (no injury)",       blank_skin),
    make_test_img("test_black",        "Black image",                        black_image),
    make_test_img("test_noise",        "Random noise",                       random_noise),
    make_test_img("test_green",        "Unrelated (green image)",            green_image),
]

# Check if real uploaded images exist to test with
for up_img in sorted(UPLOAD_DIR.glob("*.jpg"))[:3]:
    test_cases.append((str(up_img), f"Real uploaded: {up_img.name}"))

try:
    from ml.vision.yolo_wrapper import YOLO11Detector
    from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
    from ml.vision.preprocess import preprocess_image_for_inference

    yolo_det = YOLO11Detector(str(ROOT / "ml/models/yolo11n_best.pt"))
    effnet_clf = EfficientNetV2Classifier(str(ROOT / "ml/models/effnet_best.pt"))

    print(f"\n  YOLO status: {yolo_det.status}")
    print(f"  YOLO model loaded: {yolo_det.model is not None}")
    print(f"  YOLO model.names: {yolo_det.model.names if yolo_det.model else 'N/A'}")
    print(f"  YOLO supported_classes: {yolo_det.supported_classes}")
    print(f"  EfficientNet classes: {effnet_clf.classes}")
    print(f"  EfficientNet is_loaded: {effnet_clf.is_loaded}")

    print(f"\n{'─'*90}")
    header = f"{'Image':<35} {'YOLO raw dets':>12} {'YOLO finding':<14} {'YOLO conf':>9} {'EffNet winner':<14} {'EffNet conf':>10}"
    print(header)
    print("─"*90)

    results = []
    for img_path, description in test_cases:
        if not os.path.exists(img_path):
            continue
        try:
            detections = yolo_det.detect(img_path)
            raw_dets_str = f"{len(detections)} det(s)"
            if detections:
                best = max(detections, key=lambda d: d["confidence"])
                yolo_cls = best["finding"]
                yolo_conf = best["confidence"]
                yolo_above_thresh = not best.get("low_confidence", True)
            else:
                yolo_cls = "none"
                yolo_conf = None
                yolo_above_thresh = False

            img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            probs = effnet_clf.predict(img_rgb)
            eff_winner = max(probs, key=probs.get)
            eff_conf = probs[eff_winner]

            final_yolo = yolo_cls if (detections and yolo_above_thresh) else "NO DETECTION"
            label = description[:34]
            conf_str = f"{yolo_conf:.3f}" if yolo_conf is not None else "N/A"
            print(f"  {label:<33} {raw_dets_str:>12} {final_yolo:<14} {conf_str:>9} {eff_winner:<14} {eff_conf:>10.3f}")

            results.append({
                "image": description,
                "yolo_raw_count": len(detections),
                "yolo_finding": yolo_cls if detections else None,
                "yolo_conf": yolo_conf,
                "yolo_above_threshold": yolo_above_thresh,
                "effnet_winner": eff_winner,
                "effnet_probs": {k: round(float(v), 4) for k, v in probs.items()},
            })
        except Exception as e:
            print(f"  {description[:33]:<33}  ERROR: {e}")

    print("─"*90)
    print(f"\n  Full EfficientNet probability breakdown for each image:")
    for r in results:
        print(f"\n  [{r['image'][:50]}]")
        for cls, prob in r["effnet_probs"].items():
            bar = "█" * int(prob * 20)
            print(f"    {cls:<12} {prob:>6.1%}  {bar}")

except Exception as e:
    import traceback
    print(f"\n  INFERENCE ERROR: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("  AUDIT COMPLETE")
print("="*70)
