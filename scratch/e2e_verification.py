import os
import requests
import json
import time

def run_e2e_verification():
    print("====================================================")
    # Target URL of the running FastAPI server
    base_url = "http://127.0.0.1:8000"
    
    print(f"Connecting to FastAPI backend at {base_url}...")
    
    # 1. Health Check
    try:
        res = requests.get(f"{base_url}/api/health")
        assert res.status_code == 200, "Health check failed."
        print("[Step 1] Health Check: SUCCESS")
    except Exception as e:
        print(f"[Step 1] Health Check: FAILED ({e})")
        return

    # 2. Check /api/models Endpoint
    try:
        res = requests.get(f"{base_url}/api/models")
        assert res.status_code == 200, "Models endpoint failed."
        models = res.json()
        yolo_info = next((m for m in models if m["model_name"] == "YOLO11"), None)
        assert yolo_info is not None, "YOLO11 not found in registry."
        print(f"[Step 2] Models check: SUCCESS (YOLO11 Status: {yolo_info['status']}, Weights Loaded: {yolo_info['weights_loaded']})")
    except Exception as e:
        print(f"[Step 2] Models check: FAILED ({e})")
        return

    # 3. Create Case
    try:
        payload = {"notes": "E2E automated validation case"}
        res = requests.post(f"{base_url}/api/cases", json=payload)
        assert res.status_code == 201, "Case creation failed."
        case_data = res.json()
        case_id = case_data["case_id"]
        print(f"[Step 3] Case creation: SUCCESS (Case ID: {case_id})")
    except Exception as e:
        print(f"[Step 3] Case creation: FAILED ({e})")
        return

    # 4. Upload Image
    try:
        # Create a dummy image for upload
        import numpy as np
        import cv2
        dummy_img_path = "scratch/temp_e2e_upload.jpg"
        os.makedirs(os.path.dirname(dummy_img_path), exist_ok=True)
        # 300x300 canvas with a green ellipse (simulating a bruise) to trigger a valid upload
        canvas = np.ones((300, 300, 3), dtype=np.uint8) * 180
        cv2.ellipse(canvas, (150, 150), (40, 20), 0, 0, 360, (0, 0, 255), -1)
        cv2.imwrite(dummy_img_path, canvas)
        
        with open(dummy_img_path, "rb") as f:
            files = {"file": (os.path.basename(dummy_img_path), f, "image/jpeg")}
            res = requests.post(f"{base_url}/api/cases/{case_id}/image", files=files)
            
        assert res.status_code == 200, f"Image upload failed: {res.text}"
        img_upload_data = res.json()
        assert img_upload_data["message"] == "Image uploaded and verified successfully", "Incorrect message."
        print(f"[Step 4] Image upload and quality verify: SUCCESS")
    except Exception as e:
        print(f"[Step 4] Image upload: FAILED ({e})")
        return

    # 5. Questionnaire Submission
    try:
        questionnaire_payload = {
            "answers": {
                "pain_level": 6,
                "injury_location": "wrist",
                "movement_limitation": "yes",
                "weight_bearing": "no",
                "redness": "yes",
                "warmth": "yes",
                "onset": "acute",
                "previous_injury": "no",
                "injury_mechanism": "fall",
                "direct_impact": "yes"
            },
            "voice_used": False,
            "answer_source": "typed"
        }
        res = requests.post(f"{base_url}/api/cases/{case_id}/questionnaire", json=questionnaire_payload)
        assert res.status_code == 200, "Questionnaire submission failed."
        print("[Step 5] Questionnaire Submission: SUCCESS")
    except Exception as e:
        print(f"[Step 5] Questionnaire Submission: FAILED ({e})")
        return

    # 6. Sensor Ingestion (Simulated path)
    try:
        sensor_payload = {
            "scenario": "football_fall"
        }
        res = requests.post(f"{base_url}/api/cases/{case_id}/sensor/simulate", json=sensor_payload)
        assert res.status_code == 200, f"Sensor simulation failed: {res.text}"
        res_data = res.json()
        sensor_data = res_data["summary"]
        print(f"[Step 6] Sensor Ingestion (Simulation): SUCCESS (Peak g-force: {sensor_data['peak_g_force']})")
    except Exception as e:
        print(f"[Step 6] Sensor Ingestion: FAILED ({e})")
        return

    # 7. Run AI Analysis
    try:
        res = requests.post(f"{base_url}/api/cases/{case_id}/analyze")
        assert res.status_code == 200, f"AI Analysis trigger failed: {res.text}"
        
        # Now fetch the updated case details from GET /api/cases/{case_id}
        res_case = requests.get(f"{base_url}/api/cases/{case_id}")
        assert res_case.status_code == 200, "Failed to retrieve case after analysis."
        case_detail = res_case.json()
        
        # Verify predictions and vision results exist in case detail
        assert "visible_injury" in case_detail and case_detail["visible_injury"] is not None, "Missing visible injury results."
        assert "xgboost_prediction" in case_detail and case_detail["xgboost_prediction"] is not None, "Missing XGBoost prediction."
        assert "quantum_prediction" in case_detail and case_detail["quantum_prediction"] is not None, "Missing Quantum VQC prediction."
        assert "agreement_score" in case_detail and case_detail["agreement_score"] is not None, "Missing agreement score."
        
        print(f"[Step 7] Run AI Analysis & Case Retrieve: SUCCESS")
        print(f"  - Finding detected: {case_detail['visible_injury']['finding_detected']}")
        print(f"  - XGBoost prediction: {case_detail['xgboost_prediction']['class']}")
        print(f"  - Quantum prediction: {case_detail['quantum_prediction']['class']}")
        print(f"  - Fusion modality config used: {case_detail['model_configuration_used']}")
    except Exception as e:
        print(f"[Step 7] Run AI Analysis: FAILED ({e})")
        return

    # 8. Trigger Demo SOS
    try:
        sos_payload = {"mode": "local_demo"}
        res = requests.post(f"{base_url}/api/cases/{case_id}/sos/demo/trigger", json=sos_payload)
        assert res.status_code == 200, f"SOS trigger failed: {res.text}"
        
        # Fetch updated case to check SOS status
        res_case = requests.get(f"{base_url}/api/cases/{case_id}")
        assert res_case.status_code == 200
        case_detail = res_case.json()
        
        print(f"[Step 8] Trigger Demo SOS: SUCCESS (SOS Status: {case_detail['sos_status']})")
    except Exception as e:
        print(f"[Step 8] Trigger Demo SOS: FAILED ({e})")
        return

    # 9. Generate Reports
    try:
        # JSON Report
        res_json = requests.get(f"{base_url}/api/cases/{case_id}/report/json")
        assert res_json.status_code == 200, f"JSON Report generation failed: {res_json.text}"
        report_json = res_json.json()
        
        # PDF Report
        res_pdf = requests.get(f"{base_url}/api/cases/{case_id}/report/pdf")
        assert res_pdf.status_code == 200, f"PDF Report generation failed: {res_pdf.text}"
        
        # Cross-check values
        assert report_json["case_id"] == case_id
        assert report_json["xgboost"]["class"] == case_detail["xgboost_prediction"]["class"]
        assert report_json["questionnaire"]["answers"]["pain_level"] == 6
        assert report_json["sensor"]["peak_g_force"] == sensor_data["peak_g_force"]
        
        print("[Step 9] Generate and Cross-check Reports: SUCCESS")
    except Exception as e:
        print(f"[Step 9] Generate and Cross-check Reports: FAILED ({e})")
        return

    print("====================================================")
    print("ALL E2E VALIDATION STEPS PASSED SUCCESSFULLY!")
    print("====================================================")
    
    # Cleanup temp upload file
    if os.path.exists(dummy_img_path):
        os.remove(dummy_img_path)

if __name__ == "__main__":
    run_e2e_verification()
