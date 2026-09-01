import os
import sys
import pandas as pd
from ultralytics import YOLO

# Set search path to include project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.vision.yolo_wrapper import YOLO11Detector

def main():
    model_path = "ml/models/yolo11n_best.pt"
    detector = YOLO11Detector(model_path)
    
    manifest_path = "data/datasets/yolo_injury/manifest.csv"
    if not os.path.exists(manifest_path):
        print("Manifest not found.")
        return
        
    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"]
    
    print(f"Total test images: {len(test_df)}")
    found_any = False
    
    for idx, row in test_df.iterrows():
        img_path = row["image_path"]
        # Run detect
        res = detector.detect(img_path)
        if res:
            found_any = True
            print(f"\nImage: {img_path}")
            print(f"  Actual classes: {row['class']}")
            for det in res:
                print(f"  Prediction: {det['finding']} (conf: {det['confidence']:.4f}) at {det['bounding_box']}")
                
    if not found_any:
        print("No detections found across all test images at the default confidence threshold.")
        # Try lower confidence to see what the raw model outputs
        print("\nChecking with lower threshold (0.05) to see raw capability...")
        detector.conf_threshold = 0.05
        for idx, row in test_df.iterrows():
            img_path = row["image_path"]
            res = detector.detect(img_path)
            if res:
                print(f"\nImage: {img_path}")
                print(f"  Actual classes: {row['class']}")
                for det in res:
                    print(f"  Prediction: {det['finding']} (conf: {det['confidence']:.4f}) at {det['bounding_box']}")

if __name__ == "__main__":
    main()
