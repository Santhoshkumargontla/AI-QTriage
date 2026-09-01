import os
import sys
import numpy as np
import pandas as pd
import cv2
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app
from backend.database.connection import get_database

def create_synthetic_data():
    os.makedirs("data/uploads", exist_ok=True)
    
    # 1. Create a synthetic image representing swelling
    # Generate a skin-like background and draw a swelling area
    img = np.ones((300, 300, 3), dtype=np.uint8) * 180
    cv2.circle(img, (150, 150), 45, (120, 100, 200), -1)
    
    # Add random high-frequency texture noise to increase Laplacian variance
    noise = np.random.randint(0, 60, (300, 300, 3), dtype=np.int16)
    img_noise = np.clip(img.astype(np.int16) + noise - 30, 0, 255).astype(np.uint8)
    
    # Add text and rectangular borders to introduce sharp edges
    cv2.rectangle(img_noise, (10, 10), (290, 290), (0, 0, 0), 2)
    cv2.putText(img_noise, "AI-QTriage Ankle Swelling Demo", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    
    image_path = "data/uploads/synthetic_swollen_ankle.jpg"
    cv2.imwrite(image_path, img_noise)
    print(f"Created synthetic image at: {image_path}")
    
    # 2. Create synthetic sensor CSV at 50Hz (20ms interval) for 3 seconds
    num_samples = 150
    timestamps = [i * 0.02 for i in range(num_samples)]
    
    # Simulate a baseline, a spike, and then stabilization
    accel_x = np.random.normal(0, 0.1, num_samples)
    accel_y = np.random.normal(9.8, 0.1, num_samples)  # gravity
    accel_z = np.random.normal(0, 0.1, num_samples)
    
    # Inject impact spike at t = 1.0s (index 50)
    # Peak magnitude: ~45 m/s^2 (~4.5g)
    accel_y[50] = 45.0
    accel_x[50] = 10.0
    accel_z[50] = 10.0
    
    # Light drop at t=1.0s (lux drops to 2)
    optical_lux = [100.0 if t < 0.98 or t > 1.02 else 2.0 for t in timestamps]
    
    sensor_df = pd.DataFrame({
        "timestamp": timestamps,
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "gyro_x": np.random.normal(0, 0.01, num_samples),
        "gyro_y": np.random.normal(0, 0.01, num_samples),
        "gyro_z": np.random.normal(0, 0.01, num_samples),
        "optical_lux": optical_lux
    })
    
    sensor_path = "data/uploads/synthetic_sensor_log.csv"
    sensor_df.to_csv(sensor_path, index=False)
    print(f"Created synthetic sensor CSV at: {sensor_path}")
    
    return image_path, sensor_path

def main():
    print("="*60)
    print("AI-QTriage - End-to-End Synthetic Demonstration Test")
    print("="*60)
    
    img_path, sensor_path = create_synthetic_data()
    
    # Connect with TestClient
    client = TestClient(app)
    
    # Step 1: Initialize Case
    print("\n[Step 1] Initializing Case...")
    resp = client.post("/api/cases", json={"notes": "Synthetic Demonstration Case notes"})
    assert resp.status_code == 201
    case_data = resp.json()
    case_id = case_data["case_id"]
    print(f"      Case initialized with ID: {case_id}")
    
    # Step 2: Upload Image
    print("\n[Step 2] Uploading Image...")
    with open(img_path, "rb") as f:
        resp = client.post(
            f"/api/cases/{case_id}/image",
            files={"file": (os.path.basename(img_path), f, "image/jpeg")}
        )
    print(f"      Response status: {resp.status_code}, body: {resp.text}")
    assert resp.status_code == 200
    print("      Image uploaded successfully.")
    print("      Quality Metrics:", resp.json()["quality_metrics"])
    
    # Step 3: Submit Questionnaire
    print("\n[Step 3] Submitting Questionnaire answers...")
    questionnaire_payload = {
        "answers": {
            "pain_level": 8,
            "injury_mechanism": "sports",
            "movement_limitation": "severe",
            "weight_bearing": "no",
            "visible_bleeding": "no",
            "crack_pop": "yes",
            "direct_impact": "yes"
        },
        "voice_used": False
    }
    resp = client.post(f"/api/cases/{case_id}/questionnaire", json=questionnaire_payload)
    assert resp.status_code == 200
    print("      Questionnaire submitted successfully.")
    
    # Step 4: Upload Sensor Data
    print("\n[Step 4] Uploading Sensor Log...")
    with open(sensor_path, "rb") as f:
        resp = client.post(
            f"/api/cases/{case_id}/sensor",
            files={"file": (os.path.basename(sensor_path), f, "text/csv")}
        )
    assert resp.status_code == 200
    sensor_res = resp.json()
    print("      Sensor data processed successfully.")
    print(f"      Peak G-Force: {sensor_res['summary']['peak_g_force']}g")
    print(f"      Stabilization Duration: {sensor_res['summary']['post_impact_stabilization_seconds']}s")
    print(f"      SOS Triggered: {sensor_res['sos_triggered']}")
    
    # Step 5: Run Hybrid Classical-Quantum Analysis
    print("\n[Step 5] Triggering Multimodal Classical-Quantum Analysis...")
    resp = client.post(f"/api/cases/{case_id}/analyze")
    assert resp.status_code == 200
    analysis_res = resp.json()
    print("      Analysis pipeline finished.")
    print("      XGBoost Prediction:", analysis_res["xgboost"])
    print("      Quantum VQC Prediction:", analysis_res["quantum"])
    print("      Consistency Status:", analysis_res["consistency"])
    
    # Step 6: Retrieve Report & Verify PDF Generation
    print("\n[Step 6] Compiling PDF Report...")
    resp = client.get(f"/api/cases/{case_id}/report/pdf")
    assert resp.status_code == 200
    pdf_bytes = resp.content
    print(f"      PDF Report generated successfully ({len(pdf_bytes)} bytes).")
    
    print("\n" + "="*60)
    print("AI-QTriage End-to-End Test completed. ALL CHECKS PASSED!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
