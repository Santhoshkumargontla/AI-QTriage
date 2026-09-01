import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """Calculates intersection-over-union (IoU) of two bounding boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    return intersection / union

def xywh_to_xyxy(x_c, y_c, w, h, img_w=640, img_h=640):
    """Converts normalized xywh coordinates to pixel xyxy coordinates."""
    x1 = (x_c - w/2) * img_w
    y1 = (y_c - h/2) * img_h
    x2 = (x_c + w/2) * img_w
    y2 = (y_c + h/2) * img_h
    return [x1, y1, x2, y2]

def main():
    print("====================================================")
    print("YOLO11n TEST SET EVALUATION")
    print("====================================================")

    model_path = os.path.join("ml", "models", "yolo11n_best.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Run train_yolo11.py first.")
        sys.exit(1)
        
    yaml_path = os.path.abspath(os.path.join("data", "datasets", "yolo_injury", "yolo11.yaml"))
    manifest_path = os.path.abspath(os.path.join("data", "datasets", "yolo_injury", "manifest.csv"))
    
    if not os.path.exists(yaml_path) or not os.path.exists(manifest_path):
        print("ERROR: Dataset config or manifest not found.")
        sys.exit(1)
        
    # Load model
    print(f"Loading trained model from {model_path}...")
    model = YOLO(model_path)
    
    # Run evaluation on test split
    print("Running validation command on test set...")
    val_results = model.val(data=yaml_path, split="test", save_json=True, verbose=False)
    
    # Retrieve overall metrics
    # metrics dict keys: precision, recall, map50, map50-95
    results_dict = val_results.results_dict
    overall_p = results_dict.get("metrics/precision(B)", 0.0)
    overall_r = results_dict.get("metrics/recall(B)", 0.0)
    overall_map50 = results_dict.get("metrics/mAP50(B)", 0.0)
    overall_map95 = results_dict.get("metrics/mAP50-95(B)", 0.0)
    
    print("\nOverall Metrics:")
    print(f"  Precision: {overall_p:.4f}")
    print(f"  Recall:    {overall_r:.4f}")
    print(f"  mAP@50:    {overall_map50:.4f}")
    print(f"  mAP@50:95: {overall_map95:.4f}")
    
    # Per-class metrics
    class_names = ["cut", "bruise", "abrasion", "laceration"]
    per_class_metrics = {}
    
    # val_results.box.p, r, ap are lists or numpy arrays corresponding to the class map indices
    for i, name in enumerate(class_names):
        try:
            # maps can map class names index
            ap50 = val_results.box.ap50[i]
            ap95 = val_results.box.ap[i] # AP@50:95
            p = val_results.box.p[i]
            r = val_results.box.r[i]
        except (IndexError, AttributeError, TypeError, KeyError):
            ap50 = 0.0
            ap95 = 0.0
            p = 0.0
            r = 0.0
            
        per_class_metrics[name] = {
            "precision": float(p),
            "recall": float(r),
            "ap50": float(ap50),
            "ap50_95": float(ap95)
        }
        
    print("\nPer-Class Metrics:")
    for name, metrics in per_class_metrics.items():
        print(f"  {name.capitalize()}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, AP50={metrics['ap50']:.4f}")

    # Read test split from manifest
    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"]
    test_images_count = len(test_df)
    
    # Get class distribution in test set
    test_class_dist = {}
    for cls_list in test_df["class"].dropna():
        for c in cls_list.split("|"):
            if c and c != "none":
                test_class_dist[c] = test_class_dist.get(c, 0) + 1
                
    print(f"\nTest set size: {test_images_count} images")
    print(f"Test set class distribution: {test_class_dist}")

    # Select 1 manual test case from the held-out test split
    print("\nSelecting manual test case from test split...")
    manual_case = {}
    if len(test_df) > 0:
        row = test_df.iloc[0]
        test_img_path = row["image_path"]
        test_lbl_path = row["annotation_path"]
        
        # Run inference
        results = model(test_img_path, verbose=False)
        pred_box = None
        pred_class = None
        pred_conf = 0.0
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            best_box = results[0].boxes[0]
            pred_class_id = int(best_box.cls[0].item())
            pred_class = class_names[pred_class_id]
            pred_conf = float(best_box.conf[0].item())
            pred_box = best_box.xyxy[0].cpu().numpy().tolist()
            
        # Parse actual annotation
        actual_class = None
        actual_box = None
        if os.path.exists(test_lbl_path):
            with open(test_lbl_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    parts = lines[0].strip().split()
                    if len(parts) == 5:
                        act_class_id = int(parts[0])
                        actual_class = class_names[act_class_id]
                        # normalized coordinates: x_c, y_c, w, h
                        xc, yc, w, h = [float(v) for v in parts[1:]]
                        actual_box = xywh_to_xyxy(xc, yc, w, h)
                        
        # Calc IoU
        iou = 0.0
        correct = False
        if pred_box and actual_box:
            iou = calculate_iou(pred_box, actual_box)
            correct = (pred_class == actual_class) and (iou >= 0.5)
            
        manual_case = {
            "image_path": test_img_path,
            "actual_class": actual_class,
            "actual_bounding_box": actual_box,
            "predicted_class": pred_class,
            "predicted_confidence": pred_conf,
            "predicted_bounding_box": pred_box,
            "iou": iou,
            "correct": bool(correct)
        }
        print(f"  Image: {test_img_path}")
        print(f"  Actual: {actual_class} at {actual_box}")
        print(f"  Predicted: {pred_class} (conf: {pred_conf:.4f}) at {pred_box}")
        print(f"  IoU: {iou:.4f}, Match Status: {'CORRECT' if correct else 'INCORRECT'}")
    
    # 6. Test blank skin tone for no confident detection
    print("\nTesting blank skin tone for no confident detection...")
    blank_img_path = os.path.join("data", "datasets", "yolo_injury", "blank_skin.jpg")
    img = Image.new("RGB", (640, 640), (220, 180, 160))
    img.save(blank_img_path)
    
    blank_results = model(blank_img_path, verbose=False)
    has_detections = len(blank_results) > 0 and len(blank_results[0].boxes) > 0
    print(f"  Detections on blank skin image: {has_detections}")
    
    # Save results to json
    eval_results = {
        "overall": {
            "precision": float(overall_p),
            "recall": float(overall_r),
            "mAP50": float(overall_map50),
            "mAP50_95": float(overall_map95)
        },
        "per_class": per_class_metrics,
        "test_metadata": {
            "images_count": test_images_count,
            "class_distribution": test_class_dist
        },
        "manual_case": manual_case,
        "blank_skin_test": {
            "image_path": blank_img_path,
            "has_detections": has_detections
        }
    }
    
    out_dir = os.path.join("ml", "evaluation", "yolo11")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
        
    print(f"\nEvaluation results written to {out_path}")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
