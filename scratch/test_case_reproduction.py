import uuid
from datetime import datetime, timezone
from backend.database.connection import get_database
from backend.services.report_service import ResearchReportGenerator

def test_reproduce_exact_case():
    db = get_database()
    case_id = f"repro_{uuid.uuid4().hex[:8]}"
    
    # Recreate exact scenario:
    # Visible Finding: Swelling
    # Data Provenance: SYNTHETIC
    # XGBoost: HIGH
    # VQC: LOW
    # Segmentation: Unavailable/unreliable
    
    case_doc = {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc),
        "status": "analyzed",
        "is_demo": True,
        "sensor_available": False,
        "sensor_source_type": "not_provided",
        "sensor_summary": None,
        "visible_injury": {
            "finding": "Swelling",
            "finding_detected": False,
            "detection_message": "Visible finding not confidently detected by YOLO11 (supported: abrasion, bruise, cut, laceration). Research classifier category: Swelling.",
            "confidence": None,
            "bounding_box": None,
            "affected_ratio": None,
            "affected_area_ratio": None,
            "segmentation_available": False,
            "segmentation_reliable": False,
            "segmentation_status": "insufficient",
            "segmentation_reason": "No reliable segmentation mask available (YOLO detected no target region)",
            "classification": {"swelling": 0.95, "cut": 0.03, "bruise": 0.02},
            "classifier_finding": "Swelling",
            "classifier_probability": 0.95,
            "classifier_yolo_coverage": "NOT AVAILABLE",
            "source_type": "demo",
            "data_provenance": "synthetic"
        },
        "questionnaire": {
            "answers": {
                "pain_level": 6,
                "location": "ankle",
                "injury_mechanism": "sports"
            }
        },
        "xgboost_prediction": {
            "class": "HIGH",
            "probability": 0.92,
            "model_version": "v1.0.0",
            "model_id": "xgb_full_v1"
        },
        "quantum_prediction": {
            "class": "LOW",
            "score": [0.65, 0.25, 0.10],
            "model_version": "v1.0.0",
            "model_id": "vqc_sim_v1"
        },
        "agreement_score": "DISAGREEMENT",
        "prediction_agreement": "DISAGREEMENT",
        "uncertainty_status": "HIGH UNCERTAINTY",
        "uncertainty_level": "HIGH UNCERTAINTY",
        "uncertainty_reasons": [
            "Classical and quantum model predictions disagree.",
            "Segmentation output was unavailable or below the reliability threshold.",
            "This case uses synthetic or demo data."
        ],
        "consistency_analysis": {
            "score": 40.0,
            "status": "Conflicting Evidence Detected",
            "conflicts": [
                "Model Disagreement: Classical XGBoost (HIGH) and Quantum VQC (LOW) predicted different severity classes.",
                "Segmentation Warning: No reliable segmentation mask was available for area calculation.",
                "Data Notice: Assessment uses synthetic or demonstration data."
            ],
            "agreements": []
        },
        "sos_status": "not_triggered",
        "modalities_used": ["image", "questionnaire"],
        "model_configuration_used": "img_q_v1"
    }

    db.cases.insert_one(case_doc)

    try:
        generator = ResearchReportGenerator()
        report = generator.compile_report_data(case_id)
        
        print("\n=== REPRODUCED & CORRECTED CASE REPORT OUTPUT ===")
        print(f"Case ID: {report['case_id']}")
        print(f"Data Provenance: {report['data_provenance']}")
        print(f"Research Warning: {report['research_demo_warning']}")
        print(f"Visible Finding: {report['vision']['finding']}")
        print(f"YOLO11 Detection Result: {report['vision']['finding_detected']} (finding={report['vision'].get('yolo', {}).get('finding', 'None')}, confidence={report['vision'].get('confidence')}, bbox={report['vision'].get('bounding_box')})")
        print(f"EfficientNet Classifier Result: {report['vision'].get('classifier', {}).get('finding', 'Swelling')} (prob={report['vision'].get('classifier', {}).get('probability', 0.95)})")
        print(f"Segmentation Result: reliable={report['vision']['segmentation_reliable']}, status={report['vision']['segmentation_status']}")
        print(f"Affected Area: {report['vision']['affected_ratio']} (reason: {report['vision']['segmentation_reason']})")
        print(f"Grad-CAM Label: {report['vision']['gradcam_label']} - {report['vision']['gradcam_explanation']}")
        print(f"XGBoost Prediction: {report['xgboost']['class']} (prob={report['xgboost']['probability']:.2f})")
        print(f"Experimental VQC Output: {report['quantum']['class']} (scores={report['quantum']['experimental_vqc_outputs']})")
        print(f"Model Agreement: {report['prediction_agreement']}")
        print(f"Evidence Consistency: {report['multimodal_evidence_consistency']} ({report['multimodal_evidence_consistency']})")
        print(f"Uncertainty Level: {report['uncertainty_level']}")
        print(f"Uncertainty Reasons: {report['uncertainty_reasons']}")
        print(f"Sensor Source: {report['sensor_provenance']}")
        print(f"SOS Status: {report['sos']['status']}")
        print("=================================================\n")
    finally:
        db.cases.delete_one({"case_id": case_id})

if __name__ == "__main__":
    test_reproduce_exact_case()
