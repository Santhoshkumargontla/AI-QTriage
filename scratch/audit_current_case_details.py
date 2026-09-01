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

for i, case in enumerate(cases[:3]):
    case_id = case.get("case_id")
    print("="*80)
    print(f"CASE [{i+1}] ID: {case_id}")
    print("="*80)

    q = case.get("questionnaire", {})
    if isinstance(q, dict):
        q_answers = q.get("answers", {})
    else:
        q_answers = {}

    print("\n--- 1. QUESTIONNAIRE ---")
    for k, v in q_answers.items():
        print(f"  {k}: {v}")

    s = case.get("sensor_summary", {})
    if not isinstance(s, dict):
        s = {}
    print("\n--- 2. SENSOR ---")
    print(f"  sensor_available: {case.get('sensor_available')}")
    print(f"  source_type: {s.get('source_type')}")
    print(f"  peak_g_force: {s.get('peak_g_force')}")
    print(f"  stabilization_seconds: {s.get('post_impact_stabilization_seconds')}")

    xgb = case.get("xgboost_prediction", {})
    if not isinstance(xgb, dict): xgb = {}
    print("\n--- 3. XGBOOST ---")
    print(f"  class: {xgb.get('class')}")
    print(f"  probability: {xgb.get('probability')}")

    vqc = case.get("quantum_prediction", {})
    if not isinstance(vqc, dict): vqc = {}
    print("\n--- 4. VQC ---")
    print(f"  class: {vqc.get('class')}")
    print(f"  score: {vqc.get('score')}")

    print("\n--- 5. MULTIMODAL FUSION ---")
    print(f"  modalities_used: {case.get('modalities_used')}")
    print(f"  agreement_score: {case.get('agreement_score')}")
    print(f"  uncertainty_status: {case.get('uncertainty_status')}")
    print(f"  uncertainty_reasons: {case.get('uncertainty_reasons')}")
    print(f"  consistency_analysis: {case.get('consistency_analysis')}")

    safety_info = case.get("safety_information", [])
    print("\n--- 6. SAFETY GUIDANCE INFO ---")
    print(f"  safety_information: {safety_info}")
