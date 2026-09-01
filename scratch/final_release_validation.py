import os
import requests
import json
import time

def create_high_quality_dummy_image(path, size=(300, 300)):
    import numpy as np
    import cv2
    # Create canvas
    canvas = np.ones((size[0], size[1], 3), dtype=np.uint8) * 180
    # Draw a high-contrast sharp rectangle
    cv2.rectangle(canvas, (50, 50), (250, 250), (0, 0, 255), 5)
    # Draw text to increase Laplacian variance
    cv2.putText(canvas, "INJURY SHARP", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    # Draw a filled circle inside
    cv2.circle(canvas, (150, 150), 30, (0, 255, 0), -1)
    cv2.imwrite(path, canvas)

def run_release_validation():
    print("====================================================")
    print("AI-QTriage Final Release Validation & E2E Verification")
    print("====================================================")
    
    base_url = "http://127.0.0.1:8000"
    
    # 1. MongoDB connectivity diagnostics (Phase 7)
    print("\n--- Phase 7: MongoDB Connectivity Diagnostic ---")
    try:
        from backend.database.connection import get_database
        db = get_database()
        # Verify DNS / TLS / Authentication by reading/writing to the database
        db_name = db.name
        print(f"  [MongoDB Connection] Successful! Connected to database: {db_name}")
        
        # Test collection read/write
        test_doc = {"test_key": "release_audit_val", "timestamp": time.time()}
        db.test_collection.insert_one(test_doc)
        read_back = db.test_collection.find_one({"test_key": "release_audit_val"})
        assert read_back is not None, "Failed to read back test document."
        db.test_collection.delete_many({"test_key": "release_audit_val"})
        print("  [MongoDB CRUD Check] SUCCESS (Collection read, write, and delete verified).")
    except Exception as e:
        print(f"  [MongoDB Check] FAILED: {e}")
        return

    # 2. Check models load status (Phase 3)
    print("\n--- Phase 3: Models Endpoint Load Status ---")
    try:
        res = requests.get(f"{base_url}/api/models")
        assert res.status_code == 200, "Models endpoint failed."
        models = res.json()
        for m in models:
            print(f"  Model: {m['model_name']} | Status: {m['status']} | Loaded: {m['weights_loaded']}")
            if m["model_name"] == "YOLO11":
                assert m["status"] == "TRAINED / LOADED", "YOLO11 is not TRAINED / LOADED."
    except Exception as e:
        print(f"  [Models Check] FAILED: {e}")
        return

    # 3. Create and verify Test Case A (Sensor Available - Phase 5)
    print("\n--- Phase 5: TEST A - SENSOR AVAILABLE ---")
    try:
        # Create Case
        res_case = requests.post(f"{base_url}/api/cases", json={"notes": "TEST A - Sensor Available"})
        assert res_case.status_code == 201
        case_id_a = res_case.json()["case_id"]
        print(f"  Created Case A: {case_id_a}")
        
        # Upload dummy image
        img_path = "scratch/temp_release_a.jpg"
        create_high_quality_dummy_image(img_path)
        
        with open(img_path, "rb") as f:
            res_img = requests.post(f"{base_url}/api/cases/{case_id_a}/image", files={"file": (os.path.basename(img_path), f, "image/jpeg")})
        assert res_img.status_code == 200, f"Image upload failed: {res_img.text}"
        
        # Submit Questionnaire
        q_payload = {
            "answers": {
                "pain_level": 7,
                "injury_location": "knee",
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
        res_q = requests.post(f"{base_url}/api/cases/{case_id_a}/questionnaire", json=q_payload)
        assert res_q.status_code == 200
        
        # Ingest Sensor Data (simulate scenario)
        res_sensor = requests.post(f"{base_url}/api/cases/{case_id_a}/sensor/simulate", json={"scenario": "football_fall"})
        assert res_sensor.status_code == 200
        sensor_metrics = res_sensor.json()["summary"]
        
        # Run AI Analysis
        res_analysis = requests.post(f"{base_url}/api/cases/{case_id_a}/analyze")
        assert res_analysis.status_code == 200
        
        # Verify MongoDB Persistence for Case A
        case_doc_a = db.cases.find_one({"case_id": case_id_a})
        assert case_doc_a is not None
        assert case_doc_a.get("sensor_available") is True, "Sensor available flag should be True."
        assert "sensor_summary" in case_doc_a and case_doc_a["sensor_summary"] is not None
        assert case_doc_a["sensor_summary"].get("peak_g_force") == sensor_metrics["peak_g_force"]
        assert len(case_doc_a.get("modalities_used", [])) == 3, f"Should use 3 modalities: {case_doc_a.get('modalities_used')}"
        
        print("  [TEST A SUCCESS] Full Multimodal (image + questionnaire + sensor) verified and persisted in DB.")
        if os.path.exists(img_path):
            os.remove(img_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [TEST A] FAILED: {e}")
        return

    # 4. Create and verify Test Case B (Sensor Not Provided - Phase 5)
    print("\n--- Phase 5: TEST B - SENSOR NOT PROVIDED ---")
    try:
        # Create Case
        res_case = requests.post(f"{base_url}/api/cases", json={"notes": "TEST B - Sensor Skip"})
        assert res_case.status_code == 201
        case_id_b = res_case.json()["case_id"]
        print(f"  Created Case B: {case_id_b}")
        
        # Upload dummy image
        img_path = "scratch/temp_release_b.jpg"
        create_high_quality_dummy_image(img_path)
        
        with open(img_path, "rb") as f:
            res_img = requests.post(f"{base_url}/api/cases/{case_id_b}/image", files={"file": (os.path.basename(img_path), f, "image/jpeg")})
        assert res_img.status_code == 200, f"Image upload failed: {res_img.text}"
        
        # Submit Questionnaire
        res_q = requests.post(f"{base_url}/api/cases/{case_id_b}/questionnaire", json=q_payload)
        assert res_q.status_code == 200
        
        # Choose CONTINUE WITHOUT SENSOR DATA (Skip sensor)
        res_skip = requests.post(f"{base_url}/api/cases/{case_id_b}/sensor/skip")
        assert res_skip.status_code == 200
        
        # Run AI Analysis
        res_analysis = requests.post(f"{base_url}/api/cases/{case_id_b}/analyze")
        assert res_analysis.status_code == 200
        
        # Verify MongoDB Persistence for Case B
        case_doc_b = db.cases.find_one({"case_id": case_id_b})
        assert case_doc_b is not None
        assert case_doc_b.get("sensor_available") is False, "Sensor available flag should be False."
        assert case_doc_b.get("sensor_summary") is None, "Sensor summary should be None (no fabricated sensor data)."
        assert len(case_doc_b.get("modalities_used", [])) == 2, f"Should use 2 modalities: {case_doc_b.get('modalities_used')}"
        
        print("  [TEST B SUCCESS] Reduced Modality (image + questionnaire) verified. Zero sensor data fabrication.")
        if os.path.exists(img_path):
            os.remove(img_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [TEST B] FAILED: {e}")
        return

    # 5. Complete SOS Countdown & Cancel Test (Phase 6)
    print("\n--- Phase 6: SOS Countdown and Response Test ---")
    try:
        # Trigger SOS on Case A
        res_trigger = requests.post(f"{base_url}/api/cases/{case_id_a}/sos/demo/trigger", json={"mode": "local_demo"})
        assert res_trigger.status_code == 200
        print("  SOS Countdown started. Checking initial status...")
        
        # Poll status
        res_status = requests.get(f"{base_url}/api/cases/{case_id_a}/sos/status")
        assert res_status.status_code == 200
        status_data = res_status.json()
        print(f"  Current Status: {status_data['status']} | Remaining Seconds: {status_data['remaining_seconds']}")
        assert status_data["status"] == "countdown"
        assert 0.0 < status_data["remaining_seconds"] <= 10.0
        
        # Cancel SOS using I'M SAFE
        res_respond = requests.post(f"{base_url}/api/cases/{case_id_a}/sos/demo/respond", json={"user_response": "safe", "mode": "local_demo"})
        assert res_respond.status_code == 200
        assert res_respond.json()["sos_status"] == "cancelled"
        
        # Verify MongoDB updated
        case_doc_a_sos = db.cases.find_one({"case_id": case_id_a})
        assert case_doc_a_sos["sos_status"] == "cancelled"
        print("  [SOS Cancel Check] SUCCESS (Status correctly set to cancelled in MongoDB).")
        
        # Trigger SOS again and allow it to reach zero (simulate zero response)
        print("  Triggering SOS again to let countdown complete...")
        requests.post(f"{base_url}/api/cases/{case_id_a}/sos/demo/trigger", json={"mode": "local_demo"})
        # Respond with no_response to simulate timeout
        res_timeout = requests.post(f"{base_url}/api/cases/{case_id_a}/sos/demo/respond", json={"user_response": "no_response", "mode": "local_demo"})
        assert res_timeout.status_code == 200
        assert res_timeout.json()["sos_status"] == "demo_triggered"
        
        case_doc_a_timeout = db.cases.find_one({"case_id": case_id_a})
        assert case_doc_a_timeout["sos_status"] == "demo_triggered"
        print("  [SOS Complete Check] SUCCESS (Final status correctly set to demo_triggered in MongoDB).")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [SOS Check] FAILED: {e}")
        return

    # 6. Report Consistency Cross-Check (Phase 11)
    print("\n--- Phase 11: Cross-System Report Consistency Verification ---")
    try:
        # Generate JSON Report for Case A
        res_json = requests.get(f"{base_url}/api/cases/{case_id_a}/report/json")
        assert res_json.status_code == 200
        report_json = res_json.json()
        
        # Fetch Case from DB
        case_db = db.cases.find_one({"case_id": case_id_a})
        
        # Cross check consistency
        assert report_json["case_id"] == case_db["case_id"], "Case ID mismatch."
        assert report_json["xgboost"]["class"] == case_db["xgboost_prediction"]["class"], "XGBoost class mismatch."
        assert report_json["quantum"]["class"] == case_db["quantum_prediction"]["class"], "Quantum class mismatch."
        assert report_json["questionnaire"]["answers"]["pain_level"] == case_db["questionnaire"]["answers"]["pain_level"], "Pain level mismatch."
        assert report_json["sensor"]["peak_g_force"] == case_db["sensor_summary"]["peak_g_force"], "Sensor peak g-force mismatch."
        assert report_json["multimodal"]["modalities_used"] == case_db["modalities_used"], "Modalities used mismatch."
        
        print("  [Consistency Check] SUCCESS. DB and JSON Report values match exactly.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [Consistency Check] FAILED: {e}")
        return

    print("\n====================================================")
    print("RELEASE VERIFICATION COMPLETED SUCCESSFULLY!")
    print("====================================================")

if __name__ == "__main__":
    run_release_validation()
