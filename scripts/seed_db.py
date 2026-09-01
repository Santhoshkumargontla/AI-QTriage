import sys
import os
from datetime import datetime

# Include project root in python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.connection import get_database, init_db_indexes

def seed_db():
    print("="*60)
    print("AI-QTriage - Seeding MongoDB Database")
    print("="*60)
    
    try:
        db = get_database()
        init_db_indexes()
        
        # 1. Seed Safety Guidance Collection
        print("\n[1/2] Seeding 'safety_guidance' collection...")
        safety_data = [
            {
                "finding": "Localized swelling",
                "content": "Avoid unnecessary weight-bearing on the affected area. Keep the limb elevated above heart level where possible. Protect the joint/area from further movement or impact. Seek appropriate professional evaluation if pain persists or severe movement limitations occur.",
                "source": "Curated Emergency First-Aid Guidelines - Rev 2026",
                "review_status": "approved",
                "version": "1.0.0",
                "updated_at": datetime.utcnow()
            },
            {
                "finding": "Cut",
                "content": "Clean the area gently with water. Apply light pressure with a clean cloth to stop minor bleeding. Keep the area covered with a clean, dry dressing. Seek professional clinical evaluation if the cut is deep, shows signs of infection, or requires suturing.",
                "source": "Curated Emergency First-Aid Guidelines - Rev 2026",
                "review_status": "approved",
                "version": "1.0.0",
                "updated_at": datetime.utcnow()
            },
            {
                "finding": "Bruise",
                "content": "Apply a cold pack wrapped in a cloth to reduce localized discomfort. Elevate the affected limb if possible to assist circulation. Avoid heavy friction or massage on the bruised area. Seek professional clinical review if bruises appear without direct impact.",
                "source": "Curated Emergency First-Aid Guidelines - Rev 2026",
                "review_status": "approved",
                "version": "1.0.0",
                "updated_at": datetime.utcnow()
            }
        ]
        
        # Clear existing guidance to keep it idempotent
        db.safety_guidance.delete_many({})
        db.safety_guidance.insert_many(safety_data)
        print("      Success! Inserted safety guidance templates.")
        
        # 2. Seed Mock Model Versions for display
        print("\n[2/2] Seeding 'model_versions' collection...")
        model_versions_data = [
            {
                "model_name": "YOLO11",
                "model_version": "v1.0.0",
                "dataset_version": "v1.0",
                "feature_version": "v1.0",
                "rule_version": "v1.0",
                "training_date": datetime.utcnow(),
                "hyperparameters": {"imgsz": 640, "batch": 16},
                "metrics": {"mAP50": 0.82, "mAP50-95": 0.58},
                "status": "Ready - PyTorch Weights Loaded"
            },
            {
                "model_name": "EfficientNetV2",
                "model_version": "v1.0.0",
                "dataset_version": "v1.0",
                "feature_version": "v1.0",
                "rule_version": "v1.0",
                "training_date": datetime.utcnow(),
                "hyperparameters": {"lr": 0.001, "backbone": "efficientnet_v2_s"},
                "metrics": {"accuracy": 0.88, "f1_macro": 0.86},
                "status": "Ready - PyTorch Weights Loaded"
            },
            {
                "model_name": "U-Net",
                "model_version": "v1.0.0",
                "dataset_version": "v1.0",
                "feature_version": "v1.0",
                "rule_version": "v1.0",
                "training_date": datetime.utcnow(),
                "hyperparameters": {"encoder": "resnet34"},
                "metrics": {"dice": 0.85, "iou": 0.74},
                "status": "Ready - PyTorch Weights Loaded"
            },
            {
                "model_name": "XGBoost",
                "model_version": "v1.0.0",
                "dataset_version": "v1.0",
                "feature_version": "v1.0",
                "rule_version": "v1.0",
                "training_date": datetime.utcnow(),
                "hyperparameters": {"max_depth": 4, "learning_rate": 0.1},
                "metrics": {"accuracy": 0.89, "f1_macro": 0.88},
                "status": "Ready - Classifier Trained"
            },
            {
                "model_name": "VQC",
                "model_version": "v1.0.0",
                "dataset_version": "v1.0",
                "feature_version": "v1.0",
                "rule_version": "v1.0",
                "training_date": datetime.utcnow(),
                "hyperparameters": {"qubits": 4, "layers": 2},
                "metrics": {"accuracy": 0.72, "f1_macro": 0.70},
                "status": "Ready - Quantum Circuit Trained"
            }
        ]
        
        db.model_versions.delete_many({})
        db.model_versions.insert_many(model_versions_data)
        print("      Success! Seeding model registry complete.")
        
        print("\nDatabase seeding completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Database seeding failed: {str(e)}", file=sys.stderr)
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    seed_db()
