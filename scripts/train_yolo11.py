import os
import sys
import shutil
import time
import torch
import pandas as pd
from ultralytics import YOLO

def main():
    print("====================================================")
    print("YOLO11n TRAINING")
    print("====================================================")

    # 1. Pretrained Weights Copy
    pretrained_src = "yolo11n.pt"
    pretrained_dest = os.path.join("ml", "models", "yolo11n_pretrained.pt")
    
    os.makedirs(os.path.join("ml", "models"), exist_ok=True)
    if os.path.exists(pretrained_src):
        print(f"Copying pretrained weights from {pretrained_src} to {pretrained_dest}...")
        shutil.copy2(pretrained_src, pretrained_dest)
    elif not os.path.exists(pretrained_dest):
        print(f"Downloading pretrained weights directly to {pretrained_dest}...")
        # If the local file is not in the root, YOLO will download it automatically when we construct it
        pass
        
    yaml_path = os.path.abspath(os.path.join("data", "datasets", "yolo_injury", "yolo11.yaml"))
    if not os.path.exists(yaml_path):
        print(f"ERROR: Dataset configuration not found at {yaml_path}. Run prepare_yolo_dataset.py first.")
        sys.exit(1)
        
    print(f"Dataset configuration path: {yaml_path}")
    
    # 2. Check Device
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    # Check dataset size
    manifest_path = os.path.join("data", "datasets", "yolo_injury", "manifest.csv")
    dataset_size = 0
    class_distribution = {}
    if os.path.exists(manifest_path):
        df = pd.read_csv(manifest_path)
        dataset_size = len(df)
        train_df = df[df["split"] == "train"]
        for cls_list in train_df["class"].dropna():
            for c in cls_list.split("|"):
                if c and c != "none":
                    class_distribution[c] = class_distribution.get(c, 0) + 1
                    
    print(f"Training dataset size: {dataset_size} total unique images")
    print(f"Class distribution in training set: {class_distribution}")

    # Initialize model from pretrained base
    print(f"Initializing YOLO model from {pretrained_dest if os.path.exists(pretrained_dest) else 'yolo11n.pt'}...")
    model_init_path = pretrained_dest if os.path.exists(pretrained_dest) else "yolo11n.pt"
    model = YOLO(model_init_path)
    
    # 3. Training Run
    epochs = 50
    batch_size = 16
    patience = 10
    seed = 42
    img_size = 640
    
    project_dir = os.path.join("ml", "models", "yolo_training")
    
    print(f"Starting fine-tuning with epochs={epochs}, batch={batch_size}, imgsz={img_size}, seed={seed}...")
    start_time = time.time()
    
    try:
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            patience=patience,
            imgsz=img_size,
            batch=batch_size,
            seed=seed,
            device=device,
            project=project_dir,
            name="yolo11_fine_tuned",
            exist_ok=True,
            verbose=True
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and batch_size > 2:
            print("WARNING: CUDA Out Of Memory. Attempting with reduced batch size = 8...")
            batch_size = 8
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                patience=patience,
                imgsz=img_size,
                batch=batch_size,
                seed=seed,
                device=device,
                project=project_dir,
                name="yolo11_fine_tuned",
                exist_ok=True,
                verbose=True
            )
        else:
            raise e
            
    duration = time.time() - start_time
    print(f"Training completed in {duration:.2f} seconds.")
    
    # Get run directory
    run_dir = os.path.join(project_dir, "yolo11_fine_tuned")
    best_weights = os.path.join(run_dir, "weights", "best.pt")
    
    # Destination best weight path
    final_best_path = os.path.join("ml", "models", "yolo11n_best.pt")
    
    if os.path.exists(best_weights):
        print(f"Fine-tuning succeeded. Copying best weights to {final_best_path}...")
        shutil.copy2(best_weights, final_best_path)
    else:
        # Fallback in case weights are in a slightly different location
        alternative_best = os.path.join(project_dir, "yolo11_fine_tuned2", "weights", "best.pt")
        if os.path.exists(alternative_best):
            shutil.copy2(alternative_best, final_best_path)
            best_weights = alternative_best
        else:
            print("ERROR: Could not find fine-tuned weights!")
            sys.exit(1)
            
    # Record metadata
    metadata = {
        "model_architecture": "YOLO11n Fine-Tuned",
        "pretrained_base": "yolo11n_pretrained.pt",
        "epochs_completed": len(results.fitness) if hasattr(results, "fitness") else epochs,
        "batch_size": batch_size,
        "device": device,
        "gpu_available": torch.cuda.is_available(),
        "training_duration_seconds": round(duration, 2),
        "training_dataset_size": dataset_size,
        "class_distribution": class_distribution,
        "weights_path": final_best_path,
        "file_size_bytes": os.path.getsize(final_best_path)
    }
    
    # Write metadata JSON
    import json
    metadata_path = os.path.join(project_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Training metadata saved to {metadata_path}")
    print("YOLO11n fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
