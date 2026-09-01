"""Fresh forensic audit — filesystem + runtime truth only. Do not trust prior JSON."""
import os, sys, json, hashlib, glob, traceback
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

out = {"audit_started": datetime.now(timezone.utc).isoformat(), "root": ROOT}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def rel(p):
    return os.path.relpath(p, ROOT)

# ---------------------------------------------------------------------------
# 1. Model artifact inventory
# ---------------------------------------------------------------------------
print("=" * 80)
print("PHASE 1: MODEL ARTIFACT HASH INVENTORY")
print("=" * 80)
exts = {".pt", ".pth", ".onnx", ".pkl", ".joblib", ".h5", ".keras", ".json", ".npz", ".npy"}
skip_dirs = {"node_modules", "venv", ".git", "__pycache__", "audit", ".pytest_cache", "_archive"}
artifacts = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in skip_dirs]
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in exts:
            continue
        # skip huge unrelated json dumps
        if ext == ".json" and "models" not in dirpath.replace("\\", "/").lower() and fn not in (
            "model_registry.json",
        ):
            continue
        full = os.path.join(dirpath, fn)
        try:
            size = os.path.getsize(full)
        except OSError:
            continue
        if size > 200 * 1024 * 1024:
            continue
        rec = {
            "path": rel(full),
            "exists": True,
            "size": size,
            "sha256": sha256(full),
        }
        artifacts.append(rec)
        print(f"{size:12d}  {rec['sha256'][:16]}  {rec['path']}")

out["artifacts"] = artifacts

# ---------------------------------------------------------------------------
# 2. YOLO runtime path + model metadata
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 3: YOLO RUNTIME LOAD")
print("=" * 80)
yolo_section = {}
try:
    from ml.vision.yolo_wrapper import YOLO11Detector
    det = YOLO11Detector()
    info = det.get_info()
    yolo_section["wrapper_info"] = info
    yolo_section["model_is_none"] = det.model is None
    print("WRAPPER INFO:", json.dumps(info, indent=2))
    if det.model is not None:
        m = det.model
        yolo_section["names"] = dict(m.names) if hasattr(m, "names") else None
        yolo_section["task"] = getattr(m, "task", None)
        yolo_section["overrides"] = dict(getattr(m, "overrides", {}) or {})
        ckpt = getattr(m, "ckpt_path", None) or getattr(m, "pt_path", None)
        yolo_section["ckpt_path"] = str(ckpt) if ckpt else None
        print("model.names:", yolo_section["names"])
        print("model.task:", yolo_section["task"])
        print("model.overrides:", yolo_section["overrides"])
        print("model.ckpt_path:", yolo_section["ckpt_path"])
except Exception:
    yolo_section["load_error"] = traceback.format_exc()
    print(yolo_section["load_error"])
    det = None

# ---------------------------------------------------------------------------
# 3. Dataset counts
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 3.4: YOLO DATASET COUNTS")
print("=" * 80)

def count_split(base, split):
    img_dir = os.path.join(base, "images", split)
    lbl_dir = os.path.join(base, "labels", split)
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        imgs.extend(glob.glob(os.path.join(img_dir, ext)))
    labels = glob.glob(os.path.join(lbl_dir, "*.txt"))
    class_counts = {}
    empty_labels = 0
    invalid_coords = 0
    missing_label_files = 0
    images_without_ann = 0
    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]
        lf = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lf):
            missing_label_files += 1
            images_without_ann += 1
            continue
        with open(lf, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            empty_labels += 1
            images_without_ann += 1
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                invalid_coords += 1
                continue
            try:
                cid = int(float(parts[0]))
                vals = [float(x) for x in parts[1:5]]
            except ValueError:
                invalid_coords += 1
                continue
            class_counts[cid] = class_counts.get(cid, 0) + 1
            x, y, w, h = vals
            if min(x, y, w, h) < 0 or max(x, y) > 1 or w > 1 or h > 1:
                invalid_coords += 1
    return {
        "images": len(imgs),
        "label_files": len(labels),
        "class_label_counts": class_counts,
        "empty_labels": empty_labels,
        "invalid_coords": invalid_coords,
        "missing_label_files": missing_label_files,
        "images_without_ann": images_without_ann,
        "sample_images": [rel(p) for p in imgs[:3]],
    }

dataset_bases = {
    "yolo_real_wound": os.path.join(ROOT, "data", "datasets", "yolo_real_wound"),
    "yolo_injury": os.path.join(ROOT, "data", "datasets", "yolo_injury"),
    "synthetic_wound": os.path.join(ROOT, "data", "datasets", "raw", "synthetic_wound"),
}
dataset_stats = {}
for name, base in dataset_bases.items():
    dataset_stats[name] = {"exists": os.path.exists(base)}
    if not os.path.exists(base):
        continue
    yaml_files = glob.glob(os.path.join(base, "*.yaml")) + glob.glob(os.path.join(base, "*.yml"))
    dataset_stats[name]["yaml_files"] = [rel(p) for p in yaml_files]
    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = count_split(base, split)
        print(f"{name}/{split}: {splits[split]['images']} images, {splits[split]['label_files']} labels, classes={splits[split]['class_label_counts']}")
    dataset_stats[name]["splits"] = splits

out["datasets"] = dataset_stats

# other dataset dirs
other_ds = {}
ds_root = os.path.join(ROOT, "data", "datasets")
if os.path.exists(ds_root):
    for d in sorted(os.listdir(ds_root)):
        p = os.path.join(ds_root, d)
        if os.path.isdir(p):
            nfiles = sum(len(files) for _, _, files in os.walk(p))
            other_ds[d] = nfiles
            print(f"data/datasets/{d}: {nfiles} files")
out["dataset_dir_file_counts"] = other_ds

# ---------------------------------------------------------------------------
# 4. Direct YOLO inference at multiple confs
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 3.6: DIRECT YOLO INFERENCE")
print("=" * 80)

# pick images
injury_candidates = []
for base in dataset_bases.values():
    for split in ("val", "test", "train"):
        injury_candidates.extend(glob.glob(os.path.join(base, "images", split, "*.jpg")))
        injury_candidates.extend(glob.glob(os.path.join(base, "images", split, "*.png")))
        injury_candidates.extend(glob.glob(os.path.join(base, "images", split, "*.jpeg")))

sample_img = os.path.join(ROOT, "data", "sample", "image", "football_injury.jpg")
uploads = glob.glob(os.path.join(ROOT, "data", "uploads", "*.jpg")) + glob.glob(os.path.join(ROOT, "data", "uploads", "*.png"))
test_imgs_dir = glob.glob(os.path.join(ROOT, "data", "test_images", "*")) + glob.glob(os.path.join(ROOT, "data", "test_suite", "*"))

print(f"sample football_injury.jpg exists: {os.path.exists(sample_img)}")
print(f"dataset images found: {len(injury_candidates)}")
print(f"uploads images: {len(uploads)}")
print(f"test_images/test_suite files: {len(test_imgs_dir)}")

# Create a blank non-injury image
import numpy as np
import cv2
non_injury_path = os.path.join(ROOT, "data", "debug", "yolo", "forensic_non_injury.png")
os.makedirs(os.path.dirname(non_injury_path), exist_ok=True)
cv2.imwrite(non_injury_path, np.full((480, 640, 3), 180, dtype=np.uint8))  # uniform gray

test_images = {}
if os.path.exists(sample_img):
    test_images["known_or_demo_injury"] = sample_img
elif injury_candidates:
    test_images["known_or_demo_injury"] = injury_candidates[0]
if injury_candidates:
    test_images["dataset_image"] = injury_candidates[0]
    if len(injury_candidates) > 1:
        test_images["dataset_image_2"] = injury_candidates[1]
test_images["non_injury_uniform_gray"] = non_injury_path
if uploads:
    test_images["app_upload"] = uploads[0]

out["test_image_paths"] = {k: rel(v) for k, v in test_images.items()}

inference_table = {}
raw_predict_table = {}
if det is not None and det.model is not None:
    for label, img_path in test_images.items():
        print(f"\n--- Image: {label} ({rel(img_path)}) ---")
        row = {}
        raw_row = {}
        for conf in (0.01, 0.05, 0.10, 0.25, 0.50):
            try:
                results = det.model(img_path, conf=conf, verbose=False)
                r = results[0]
                n = 0 if r.boxes is None else len(r.boxes)
                class_ids, class_names, confs, boxes = [], [], [], []
                if r.boxes is not None:
                    for box in r.boxes:
                        cid = int(box.cls[0].item())
                        class_ids.append(cid)
                        class_names.append(det.model.names.get(cid, str(cid)))
                        confs.append(round(float(box.conf[0].item()), 4))
                        boxes.append([round(v, 2) for v in box.xyxy[0].cpu().numpy().tolist()])
                rec = {
                    "n_detections": n,
                    "class_ids": class_ids,
                    "class_names": class_names,
                    "confidences": confs,
                    "boxes": boxes,
                }
                raw_row[str(conf)] = rec
                print(f"  conf={conf:.2f}  n={n}  names={class_names}  confs={confs}")
            except Exception:
                raw_row[str(conf)] = {"error": traceback.format_exc()}
                print(f"  conf={conf:.2f} ERROR")
            try:
                # wrapper detect() always uses conf=0.10 internally
                pass
            except Exception:
                pass
        # wrapper path (fixed conf=0.10 + class filter)
        try:
            wrapped = det.detect(img_path)
            row["wrapper_detect_conf0.10_filtered"] = wrapped
            print(f"  WRAPPER detect(): {len(wrapped)} findings { [w.get('finding') for w in wrapped] }")
        except Exception:
            row["wrapper_detect_error"] = traceback.format_exc()
            print("  WRAPPER detect() ERROR")
        inference_table[label] = row
        raw_predict_table[label] = raw_row
else:
    print("YOLO model not loaded — skipping inference")

out["yolo"] = yolo_section
out["yolo_raw_predict"] = raw_predict_table
out["yolo_wrapper_detect"] = inference_table

# ---------------------------------------------------------------------------
# 5. EfficientNet / U-Net / Grad-CAM
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 4: EFFICIENTNET / UNET / GRADCAM")
print("=" * 80)
vision_other = {}
try:
    from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
    from ml.vision.unet_wrapper import UNetSegmenter
    from ml.explainability.grad_cam import GradCAMExplain
    import torch

    clf = EfficientNetV2Classifier()
    seg = UNetSegmenter()
    vision_other["effnet_loaded"] = clf.is_loaded
    vision_other["effnet_classes"] = clf.classes
    vision_other["effnet_path_guess"] = None
    for cp in [
        os.path.join("ml", "models", "vision", "efficientnetv2_injury_best.pt"),
    ]:
        if os.path.exists(cp):
            vision_other["effnet_path_guess"] = cp
            break
    vision_other["unet_loaded"] = seg.is_loaded
    print(f"EfficientNet loaded={clf.is_loaded} classes={clf.classes}")
    print(f"U-Net loaded={seg.is_loaded}")

    # architecture sanity
    n_params_eff = sum(p.numel() for p in clf.model.parameters())
    n_params_unet = sum(p.numel() for p in seg.model.parameters())
    vision_other["effnet_param_count"] = int(n_params_eff)
    vision_other["unet_param_count"] = int(n_params_unet)
    print(f"EfficientNet params={n_params_eff}  U-Net params={n_params_unet}")

    inf_results = {}
    for label, img_path in test_images.items():
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            inf_results[label] = {"error": "imread failed"}
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # raw logits
        img_resized = cv2.resize(img_rgb, (224, 224))
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(clf.device)
        with torch.no_grad():
            logits = clf.model(tensor).squeeze().cpu().numpy()
        probs_dict = clf.predict(img_rgb)
        mask, pix, ratio, debug = seg.segment(img_rgb)
        rec = {
            "logits": [round(float(x), 4) for x in logits.tolist()],
            "softmax_via_wrapper": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in probs_dict.items()},
            "unet": {
                "pixel_count": pix,
                "affected_ratio": round(float(ratio), 6),
                "mask_shape": list(mask.shape),
                "mask_min": int(mask.min()) if mask.size else None,
                "mask_max": int(mask.max()) if mask.size else None,
                "debug": debug,
            },
        }
        inf_results[label] = rec
        print(f"{label}: winner={probs_dict.get('__winner')} maxp={probs_dict.get('__max_prob')} unet_ratio={ratio:.4f} raw_max={debug.get('raw_output_max')}")

        # Grad-CAM hash for two images if possible
        try:
            cam = GradCAMExplain(clf)
            heatmap, color_hm, overlay = cam.generate_heatmap(img_rgb)
            rec["gradcam"] = {
                "heatmap_shape": list(heatmap.shape),
                "overlay_shape": list(overlay.shape),
                "heatmap_mean": round(float(heatmap.mean()), 3),
                "heatmap_max": int(heatmap.max()),
                "layer": type(cam.target_layer).__name__,
            }
        except Exception:
            rec["gradcam_error"] = traceback.format_exc().splitlines()[-1]

    vision_other["inference"] = inf_results
except Exception:
    vision_other["error"] = traceback.format_exc()
    print(vision_other["error"])
out["vision_other"] = vision_other

# ---------------------------------------------------------------------------
# 6. XGBoost / fusion / PCA
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 4.5: XGBOOST / FUSION / PCA")
print("=" * 80)
xgb_section = {}
try:
    from ml.classifiers.xgboost_classifier import XGBoostClassifier
    from ml.fusion.feature_fusion import MultimodalFeatureFusion
    import pickle, joblib

    fusion = MultimodalFeatureFusion()
    dummy_case = {
        "vision_analysis": {
            "classification": {"Cut": 0.7, "Bruise": 0.1, "Swelling": 0.1, "Other": 0.1},
            "segmentation": {"affected_ratio": 0.12},
        },
        "questionnaire": {"answers": {"pain_level": 6, "cause": "fall", "bleeding": "yes", "movement": "limited", "limb_use": "with_pain"}},
        "sensor_summary": {"peak_g_force": 3.2, "pre_impact_delta_v": 1.1, "post_impact_stabilization_seconds": 0.8, "optical_lux_drop": False},
    }
    fused_dict, vector, names = fusion.fuse_features(dummy_case)
    xgb_section["fusion_vector_shape"] = list(vector.shape)
    xgb_section["len_feature_names"] = len(names)
    xgb_section["feature_names"] = names
    print(f"fusion_vector.shape={vector.shape} len(feature_names)={len(names)}")

    xgb_path = "ml/models/xgboost_best.json"
    clf_x = XGBoostClassifier(xgb_path)
    n_in = getattr(clf_x.model, "n_features_in_", None)
    xgb_section["loaded_from"] = xgb_path
    xgb_section["is_trained"] = clf_x.is_trained
    xgb_section["n_features_in_"] = int(n_in) if n_in is not None else None
    xgb_section["wrapper_feature_names_len"] = len(clf_x.feature_names)
    print(f"XGBoost n_features_in_={n_in} wrapper_names={len(clf_x.feature_names)} trained={clf_x.is_trained}")

    pred_idx, probs = clf_x.predict(vector)
    xgb_section["sample_pred"] = {"idx": pred_idx, "probs": [round(p, 4) for p in probs]}
    print(f"sample predict idx={pred_idx} probs={probs}")

    # PCA / scaler
    for pkl_path, key in [
        ("ml/models/pca.pkl", "pca_root"),
        ("ml/models/classical/pca.pkl", "pca_classical"),
        ("ml/models/vqc/pca.pkl", "pca_vqc"),
        ("ml/models/scaler.pkl", "scaler_root"),
        ("ml/models/classical/scaler.pkl", "scaler_classical"),
        ("ml/models/vqc/scaler.pkl", "scaler_vqc"),
    ]:
        if not os.path.exists(pkl_path):
            xgb_section[key] = {"exists": False}
            continue
        obj = None
        try:
            obj = joblib.load(pkl_path)
        except Exception:
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)
        rec = {"exists": True, "type": type(obj).__name__}
        if hasattr(obj, "n_features_in_"):
            rec["n_features_in_"] = int(obj.n_features_in_)
        if hasattr(obj, "n_components_"):
            rec["n_components_"] = int(obj.n_components_)
        if hasattr(obj, "explained_variance_ratio_"):
            rec["explained_variance_ratio_"] = [round(float(x), 4) for x in obj.explained_variance_ratio_]
        xgb_section[key] = rec
        print(f"{key}: {rec}")
except Exception:
    xgb_section["error"] = traceback.format_exc()
    print(xgb_section["error"])
out["xgboost"] = xgb_section

# ---------------------------------------------------------------------------
# 7. VQC fresh evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 5: VQC FRESH EVALUATION")
print("=" * 80)
vqc_section = {}
try:
    from ml.classifiers.vqc_classifier import VQCClassifier
    from ml.training.train_xgboost import generate_multimodal_dataset
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, matthews_corrcoef, f1_score, accuracy_score

    vqc = VQCClassifier("ml/models/vqc")
    vqc_section["is_trained"] = vqc.is_trained
    vqc_section["num_qubits"] = vqc.num_qubits
    vqc_section["q_weights_shape"] = list(vqc.q_weights.shape)
    vqc_section["lin_weights_shape"] = list(vqc.lin_weights.shape)
    vqc_section["scaler_n_features"] = int(getattr(vqc.scaler, "n_features_in_", -1))
    vqc_section["pca_n_features"] = int(getattr(vqc.pca, "n_features_in_", -1))
    vqc_section["pca_n_components"] = int(getattr(vqc.pca, "n_components_", -1))
    print(f"VQC trained={vqc.is_trained} scaler_in={vqc_section['scaler_n_features']} pca_in={vqc_section['pca_n_features']} pca_comp={vqc_section['pca_n_components']}")

    X, y = generate_multimodal_dataset(num_samples=200, seed=42)
    X_train, y_train = X[:140], y[:140]
    X_val, y_val = X[140:170], y[140:170]
    X_test, y_test = X[170:], y[170:]
    vqc_section["split_sizes"] = {"train": len(y_train), "val": len(y_val), "test": len(y_test)}
    vqc_section["train_equals_test"] = bool(np.array_equal(X_train, X_test))
    print(f"splits train={len(y_train)} val={len(y_val)} test={len(y_test)} train==test? {vqc_section['train_equals_test']}")

    xgb_preds, vqc_preds, vqc_probs = [], [], []
    xgb_probs = []
    for i in range(len(X_test)):
        xi = X_test[i]
        pi, pp = clf_x.predict(xi)
        xgb_preds.append(pi)
        xgb_probs.append(pp)
        qi, qp = vqc.predict(xi)
        vqc_preds.append(qi)
        vqc_probs.append(qp)

    y_test_l = y_test.tolist()
    vqc_section["len_y_test"] = len(y_test_l)
    vqc_section["len_xgb_predictions"] = len(xgb_preds)
    vqc_section["len_vqc_predictions"] = len(vqc_preds)
    print(f"len(y_test)={len(y_test_l)} len(xgb)={len(xgb_preds)} len(vqc)={len(vqc_preds)}")

    def metrics(y_true, y_pred, probs):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        mcc = float(matthews_corrcoef(y_true, y_pred))
        # brier
        y_oh = np.zeros((len(y_true), 3))
        for i, v in enumerate(y_true):
            y_oh[i, int(v)] = 1.0
        brier = float(np.mean(np.sum((np.array(probs) - y_oh) ** 2, axis=1)))
        # ece
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == np.array(y_true))
        ece = 0.0
        bins = np.linspace(0, 1, 6)
        for i in range(5):
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
            prop = float(np.mean(mask))
            if prop > 0:
                ece += abs(float(np.mean(accuracies[mask])) - float(np.mean(confidences[mask]))) * prop
        return {
            "confusion_matrix": cm,
            "correct_predictions": int(np.sum(np.array(y_pred) == np.array(y_true))),
            "n": len(y_true),
            "accuracy": round(acc, 6),
            "macro_precision": round(float(prec), 6),
            "macro_recall": round(float(rec), 6),
            "macro_f1": round(float(f1), 6),
            "mcc": round(mcc, 6),
            "brier": round(brier, 6),
            "ece": round(float(ece), 6),
        }

    vqc_section["xgb_fresh"] = metrics(y_test_l, xgb_preds, xgb_probs)
    vqc_section["vqc_fresh"] = metrics(y_test_l, vqc_preds, vqc_probs)
    print("XGB fresh:", vqc_section["xgb_fresh"])
    print("VQC fresh:", vqc_section["vqc_fresh"])
except Exception:
    vqc_section["error"] = traceback.format_exc()
    print(vqc_section["error"])
out["vqc"] = vqc_section

# ---------------------------------------------------------------------------
# 8. Sensor classifier
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 6: SENSOR CLASSIFIER")
print("=" * 80)
sensor_section = {}
try:
    from ml.classifiers.sensor_classifier import SensorClassifier
    from ml.sensor.sensor_processor import process_sensor_data
    sc = SensorClassifier()
    sensor_section["is_trained"] = sc.is_trained
    sensor_section["n_features_in_"] = int(getattr(sc.model, "n_features_in_", -1)) if sc.model is not None else None
    print(f"SensorClassifier trained={sc.is_trained} n_in={sensor_section['n_features_in_']}")

    demo_csv = os.path.join(ROOT, "data", "sample", "sensor", "football_fall.csv")
    sensor_section["demo_csv_exists"] = os.path.exists(demo_csv)
    if os.path.exists(demo_csv):
        summary = process_sensor_data(demo_csv)
        sensor_section["demo_summary_keys"] = list(summary.keys())
        keep = {}
        for k in ("peak_g_force", "pre_impact_delta_v", "post_impact_stabilization_seconds", "optical_lux_drop", "peak_acceleration"):
            if k in summary:
                keep[k] = summary[k]
        sensor_section["demo_key_metrics"] = keep
        pred = sc.predict_from_summary(summary)
        sensor_section["demo_classifier_pred"] = pred
        print("demo metrics:", keep)
        print("demo classifier:", pred)
except Exception:
    sensor_section["error"] = traceback.format_exc()
    print(sensor_section["error"])
out["sensor"] = sensor_section

# ---------------------------------------------------------------------------
# 9. Twilio env (no secrets)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("PHASE 7: TWILIO CONFIG (no secrets)")
print("=" * 80)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, "backend", ".env"))
twilio_keys = [
    "TWILIO_ENABLED",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_PHONE_NUMBER",
    "EMERGENCY_PHONE_NUMBER",
    "EMERGENCY_CONTACT_PHONE",
    "TWILIO_FROM_NUMBER",
    "TWILIO_TO_NUMBER",
    "TWILIO_WHATSAPP_FROM",
    "TWILIO_WHATSAPP_TO",
]
twilio_cfg = {}
for k in twilio_keys:
    v = os.environ.get(k, "")
    twilio_cfg[k] = {"configured": bool(str(v).strip()), "length": len(str(v).strip())}
    print(f"{k}: configured={bool(str(v).strip())}")
out["twilio_env"] = twilio_cfg

try:
    from backend.services.twilio_service import twilio_service
    configured, msg = twilio_service.is_configured()
    status_info = twilio_service.get_status_info()
    out["twilio_service"] = {
        "is_configured": configured,
        "status_message": msg,
        "status_info": status_info,
    }
    print("twilio_service.is_configured:", configured, msg)
except Exception:
    out["twilio_service"] = {"error": traceback.format_exc()}
    print(out["twilio_service"]["error"])

# ---------------------------------------------------------------------------
# 10. YOLO training artifacts
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("YOLO TRAINING ARTIFACTS")
print("=" * 80)
run_dir = os.path.join(ROOT, "ml", "models", "yolo_real_training", "run_real_wound")
expected_plots = [
    "results.csv", "args.yaml", "confusion_matrix.png", "confusion_matrix_normalized.png",
    "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png", "labels.jpg",
    "results.png", "BoxP_curve.png",
]
train_art = {"dir_exists": os.path.exists(run_dir), "files": {}}
if os.path.exists(run_dir):
    all_files = []
    for dp, dn, fns in os.walk(run_dir):
        for fn in fns:
            all_files.append(rel(os.path.join(dp, fn)))
    train_art["all_files"] = all_files
    print("run_real_wound files:", all_files)
for name in expected_plots:
    p = os.path.join(run_dir, name)
    train_art["files"][name] = os.path.exists(p)
    print(f"  {name}: {os.path.exists(p)}")
out["yolo_training_artifacts"] = train_art

# sample image existence
out["sample_files"] = {
    "football_injury.jpg": os.path.exists(os.path.join(ROOT, "data", "sample", "image", "football_injury.jpg")),
    "football_fall.csv": os.path.exists(os.path.join(ROOT, "data", "sample", "sensor", "football_fall.csv")),
}

out["audit_finished"] = datetime.now(timezone.utc).isoformat()
dest = os.path.join(ROOT, "scratch", "fresh_forensic_audit_output.json")
with open(dest, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
print("\nWrote", dest)
print("DONE")
