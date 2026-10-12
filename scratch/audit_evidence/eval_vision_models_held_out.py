import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
from ml.vision.unet_wrapper import UNetSegmenter
from ml.models.canonical_paths import EFFNET_CANONICAL, UNET_CANONICAL, ROOT

out_dir = "scratch/audit_evidence"
os.makedirs(out_dir, exist_ok=True)
results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "evaluation_mode": "FULL_HELD_OUT_DATASET"
}

# -------------------------------------------------------------
# 1. FULL EFFICIENTNETV2 EVALUATION ON 473 TEST PHOTOS
# -------------------------------------------------------------
print("--> Running Full Held-Out Evaluation for EfficientNetV2...")
eff_manifest = os.path.join(ROOT, "data", "datasets", "efficientnet_kaggle_v1", "manifest.csv")
if os.path.exists(eff_manifest) and os.path.exists(os.path.join(ROOT, EFFNET_CANONICAL)):
    df_eff = pd.read_csv(eff_manifest)
    test_eff = df_eff[df_eff["split"] == "test"].copy()
    
    classifier = EfficientNetV2Classifier()
    classes = [c.lower() for c in classifier.classes]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    y_true = []
    y_pred = []
    
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    
    classifier.model.eval()
    with torch.no_grad():
        for idx, row in test_eff.iterrows():
            img_rel = row["image_path"].replace("/", os.sep)
            img_path = os.path.join(ROOT, img_rel)
            true_cls = str(row["class"]).lower()
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_res = cv2.resize(img_rgb, (224, 224))
            tensor = (img_res.astype(np.float32) / 255.0 - mean) / std
            tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).to(classifier.device)
            
            logits = classifier.model(tensor)
            pred_idx = int(logits.argmax(1).cpu().item())
            
            if true_cls in class_to_idx:
                y_true.append(class_to_idx[true_cls])
                y_pred.append(pred_idx)
                
    n_eval = len(y_true)
    acc = float(accuracy_score(y_true, y_pred)) if n_eval else 0.0
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()
    
    p_i, r_i, f_i, s_i = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class = {}
    for i, c in enumerate(classes):
        per_class[c] = {
            "precision": float(p_i[i]) if i < len(p_i) else 0.0,
            "recall": float(r_i[i]) if i < len(r_i) else 0.0,
            "f1": float(f_i[i]) if i < len(f_i) else 0.0,
            "support": int(s_i[i]) if i < len(s_i) else 0
        }
        
    print(f"    EfficientNetV2: Evaluated {n_eval}/{len(test_eff)} photos | Accuracy: {acc*100:.2f}% | Macro-F1: {f1:.4f}")
    results["efficientnetv2_full_eval"] = {
        "n_evaluated": n_eval,
        "n_expected": len(test_eff),
        "accuracy": acc,
        "macro_f1": float(f1),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "confusion_matrix": cm,
        "per_class": per_class
    }
else:
    print("    EfficientNetV2 evaluation BLOCKED: manifest or checkpoint missing")
    results["efficientnetv2_full_eval"] = {"status": "BLOCKED"}

# -------------------------------------------------------------
# 2. FULL U-NET EVALUATION ON 69 HELD-OUT TEST PAIRS
# -------------------------------------------------------------
print("--> Running Full Held-Out Evaluation for ResNet34-U-Net...")
unet_manifest = os.path.join(ROOT, "data", "datasets", "unet_deduped_subject", "manifest.csv")
if os.path.exists(unet_manifest) and os.path.exists(os.path.join(ROOT, UNET_CANONICAL)):
    df_unet = pd.read_csv(unet_manifest)
    test_unet = df_unet[df_unet["split"] == "test"].copy()
    
    segmenter = UNetSegmenter()
    
    dices = []
    ious = []
    precisions = []
    recalls = []
    
    segmenter.model.eval()
    with torch.no_grad():
        for idx, row in test_unet.iterrows():
            img_rel = row["image_path"].replace("/", os.sep)
            mask_rel = row["mask_path"].replace("/", os.sep)
            img_path = os.path.join(ROOT, img_rel)
            mask_path = os.path.join(ROOT, mask_rel)
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue
                
            img = cv2.imread(img_path)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or gt_mask is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = gt_mask.shape[:2]
            
            # Predict through wrapper model
            img_res = cv2.resize(img_rgb, (256, 256))
            tensor = torch.from_numpy(img_res).permute(2, 0, 1).float() / 255.0
            mean_unet = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std_unet = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean_unet) / std_unet
            tensor = tensor.unsqueeze(0).to(segmenter.device)
            
            logits = segmenter.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask_256 = (probs > 0.5).astype(np.uint8)
            pred_bin = cv2.resize(pred_mask_256, (w, h), interpolation=cv2.INTER_NEAREST)
            gt_bin = (gt_mask > 127).astype(np.uint8)
            
            intersection = np.logical_and(pred_bin, gt_bin).sum()
            union = np.logical_or(pred_bin, gt_bin).sum()
            pred_sum = pred_bin.sum()
            gt_sum = gt_bin.sum()
            
            if gt_sum == 0 and pred_sum == 0:
                dice = 1.0
                iou = 1.0
                prec = 1.0
                rec = 1.0
            elif gt_sum == 0 and pred_sum > 0:
                dice = 0.0
                iou = 0.0
                prec = 0.0
                rec = 1.0
            elif gt_sum > 0 and pred_sum == 0:
                dice = 0.0
                iou = 0.0
                prec = 1.0
                rec = 0.0
            else:
                dice = float((2.0 * intersection) / (pred_sum + gt_sum))
                iou = float(intersection / union) if union > 0 else 0.0
                prec = float(intersection / pred_sum) if pred_sum > 0 else 0.0
                rec = float(intersection / gt_sum) if gt_sum > 0 else 0.0
                
            dices.append(dice)
            ious.append(iou)
            precisions.append(prec)
            recalls.append(rec)
            
    n_unet_eval = len(dices)
    mean_dice = float(np.mean(dices)) if n_unet_eval else 0.0
    mean_iou = float(np.mean(ious)) if n_unet_eval else 0.0
    mean_prec = float(np.mean(precisions)) if n_unet_eval else 0.0
    mean_rec = float(np.mean(recalls)) if n_unet_eval else 0.0
    
    print(f"    ResNet34-U-Net: Evaluated {n_unet_eval}/{len(test_unet)} pairs | Mean Dice: {mean_dice:.4f} | Mean IoU: {mean_iou:.4f}")
    results["unet_full_eval"] = {
        "n_evaluated": n_unet_eval,
        "n_expected": len(test_unet),
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "mean_precision": mean_prec,
        "mean_recall": mean_rec
    }
else:
    print("    ResNet34-U-Net evaluation BLOCKED: manifest or checkpoint missing")
    results["unet_full_eval"] = {"status": "BLOCKED"}

with open(os.path.join(out_dir, "vision_full_reproduced_eval.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n[DONE] Saved fresh full vision evaluation to {os.path.join(out_dir, 'vision_full_reproduced_eval.json')}")
