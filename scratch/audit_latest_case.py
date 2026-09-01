import os
import json
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ai_qtriage_db"]

cases = list(db.cases.find().sort("created_at", -1))
print(f"Total cases in DB: {len(cases)}")

if not cases:
    print("No cases found in DB!")
    exit(0)

latest_case = cases[0]
case_id = latest_case.get("case_id")
print("\n" + "="*80)
print(f"LATEST CASE ID: {case_id}")
print("="*80)

q_data = latest_case.get("questionnaire", {})
q_answers = q_data.get("answers", {})
print("\n--- 1. QUESTIONNAIRE DATA ---")
print(f"Location: {q_answers.get('location')}")
print(f"Pain: {q_answers.get('pain')}")
print(f"Movement Limitation: {q_answers.get('movement_limitation')}")
print(f"Weight Bearing: {q_answers.get('weight_bearing')}")
print(f"Redness: {q_answers.get('redness')}")
print(f"Warmth: {q_answers.get('warmth')}")
print(f"Onset: {q_answers.get('onset_hours')}")
print(f"Previous Injury: {q_answers.get('previous_injury')}")

s_data = latest_case.get("sensor_summary", {})
print("\n--- 2. SENSOR DATA ---")
print(f"Sensor Available: {latest_case.get('sensor_available')}")
print(f"Source Type: {s_data.get('source_type')}")
print(f"Peak G-Force: {s_data.get('peak_g_force')}")
print(f"Stabilization Seconds: {s_data.get('post_impact_stabilization_seconds')}")

xgb = latest_case.get("xgboost_prediction", {})
print("\n--- 3. XGBOOST PREDICTION ---")
print(f"Class: {xgb.get('class')}")
print(f"Probability: {xgb.get('probability')}")

vqc = latest_case.get("quantum_prediction", {})
print("\n--- 4. VQC PREDICTION ---")
print(f"Class: {vqc.get('class')}")
print(f"Score: {vqc.get('score')}")

print("\n--- 5. MULTIMODAL FUSION & SAFETY GUIDANCE ---")
print(f"Agreement: {latest_case.get('agreement_score')}")
print(f"Uncertainty Status: {latest_case.get('uncertainty_status')}")
print(f"Safety Information: {latest_case.get('safety_information')}")

print("\n" + "="*80)
