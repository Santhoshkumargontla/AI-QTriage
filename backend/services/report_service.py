import os
from datetime import datetime, timezone
from backend.database.connection import get_database
from backend.services.first_aid_service import first_aid_service
from backend.services.safety_service import SafetyGuidanceService

class ResearchReportGenerator:

    """Aggregates all case data into standardized JSON and downloadable PDF formats."""
    
    def __init__(self):
        self.safety_service = SafetyGuidanceService()

    def compile_report_data(self, case_id: str) -> dict:
        """
        Loads the case document from MongoDB and structures it into a comprehensive report dictionary.
        """
        db = get_database()
        case = db.cases.find_one({"case_id": case_id})
        if not case:
            raise ValueError("Case not found")
            
        # Parse visible findings details
        visible_injury = case.get("visible_injury") or {}
        finding_name = visible_injury.get("finding", "Cut")
        
        # Load rule-derived research category from canonical case fields
        xgb_pred = case.get("xgboost_prediction") or {}
        rule_category = (
            case.get("rule_derived_category") or
            case.get("safety_guidance_level") or
            xgb_pred.get("class", "LOW")
        )
        
        # Check fracture risk criteria (pain >= 7, crack/pop reported)
        q_data = case.get("questionnaire") or {}
        questionnaire_answers = q_data.get("answers") or {}
        has_fracture_risk = (
            int(questionnaire_answers.get("pain_level", 0) or questionnaire_answers.get("pain", 0)) >= 7 or
            questionnaire_answers.get("crack_pop", "no") == "yes"
        )

        
        # Get safety guidance
        safety_data = self.safety_service.get_safety_guidance(
            finding=finding_name,
            has_fracture_risk=has_fracture_risk,
            severity_category=rule_category
        )

        # Generate basic first-aid guidance from actual questionnaire answers & evidence
        first_aid_data = first_aid_service.generate_first_aid_guidance(
            questionnaire_answers=questionnaire_answers,
            sensor_summary=case.get("sensor_summary"),
            visible_injury=visible_injury,
            rule_derived_category=rule_category,
            xgboost_pred=xgb_pred,
            quantum_pred=case.get("quantum_prediction")
        )
        
        sensor_sum = case.get("sensor_summary") or {}
        vis_inj = case.get("visible_injury") or {}
        is_demo = case.get("is_demo", False) or (sensor_sum.get("source_type") in ["demo", "simulated"]) or (vis_inj.get("source_type") == "demo")
        
        # Explicit Data Provenance & Warning
        data_prov = "SYNTHETIC" if is_demo else "REAL_UPLOADED"
        research_demo_warning = (
            "Research/Demo Data Warning: Results generated using synthetic or simulated data and must not be interpreted as real clinical validation."
            if is_demo else None
        )
        sensor_prov = case.get("sensor_source_type", "not_provided")
        image_prov = "synthetic_demo" if is_demo else "uploaded"
        
        # Check SOS Status
        sos_status = case.get("sos_status", "not_triggered")

        # Extract vision fields
        vis_conf = visible_injury.get("confidence")
        vis_is_low_conf = visible_injury.get("is_low_confidence", False) or (vis_conf is not None and vis_conf < 0.40)
        finding_label = visible_injury.get("finding") or "Cut"
        if vis_is_low_conf and not finding_label.startswith("Possible "):
            finding_label = f"Possible {finding_label}"

        seg_reliable = bool(visible_injury.get("segmentation_reliable", False))
        affected_ratio_raw = visible_injury.get("affected_ratio")
        if seg_reliable and affected_ratio_raw is not None and affected_ratio_raw > 0:
            affected_ratio_display = f"{affected_ratio_raw * 100:.2f}%"
            seg_status_display = visible_injury.get("segmentation_status", "confident")
            seg_reason = visible_injury.get("segmentation_reason", "Valid segmentation mask produced.")
            affected_area_ratio_val = float(affected_ratio_raw)
        else:
            affected_ratio_display = "N/A"
            seg_status_display = "Not confidently detected"
            seg_reason = visible_injury.get("segmentation_reason") or visible_injury.get("segmentation_message") or "No reliable segmentation mask available."
            affected_area_ratio_val = None
            seg_reliable = False

        xgb_prediction_class = xgb_pred.get("class")
        vqc_obj = case.get("quantum_prediction") or {}
        vqc_prediction_class = vqc_obj.get("class")
        stored_agreement = case.get("agreement_score") or case.get("prediction_agreement")
        if stored_agreement:
            prediction_agreement_val = stored_agreement
        elif vqc_prediction_class is None or vqc_obj.get("used_in_main_decision") is False:
            prediction_agreement_val = "VQC_EXCLUDED_FROM_DECISION"
        else:
            prediction_agreement_val = "AGREEMENT" if (xgb_prediction_class == vqc_prediction_class) else "DISAGREEMENT"

        # Determine Uncertainty Level & Reasons
        uncertainty_reasons_val = case.get("uncertainty_reasons")
        if uncertainty_reasons_val is None:
            uncertainty_reasons_val = []
            if prediction_agreement_val == "DISAGREEMENT":
                uncertainty_reasons_val.append("Classical and quantum model predictions disagree.")
            if not seg_reliable:
                uncertainty_reasons_val.append("Segmentation output was unavailable or below the reliability threshold.")
            if is_demo:
                uncertainty_reasons_val.append("This case uses synthetic or demo data.")

        uncertainty_level_val = case.get("uncertainty_level") or case.get("uncertainty_status")
        if not uncertainty_level_val or uncertainty_level_val in ["High Certainty", "High Uncertainty"]:
            if prediction_agreement_val == "DISAGREEMENT" or len(uncertainty_reasons_val) >= 2:
                uncertainty_level_val = "HIGH UNCERTAINTY"
            elif len(uncertainty_reasons_val) == 1:
                uncertainty_level_val = "MODERATE UNCERTAINTY"
            else:
                uncertainty_level_val = "LOW UNCERTAINTY"

        created_at_val = case.get("created_at")
        if hasattr(created_at_val, "isoformat"):
            created_at_str = created_at_val.isoformat()
        else:
            created_at_str = str(created_at_val) if created_at_val else datetime.now(timezone.utc).isoformat()

        consistency_analysis_val = case.get("consistency_analysis") or {
            "score": 100.0 if prediction_agreement_val == "AGREEMENT" else 50.0,
            "status": "Highly Consistent" if prediction_agreement_val == "AGREEMENT" else "Conflicting Evidence Detected",
            "conflicts": [] if prediction_agreement_val == "AGREEMENT" else [f"Model Disagreement: Classical ({xgb_prediction_class}) vs Quantum ({vqc_prediction_class})"],
            "agreements": []
        }

        # Compile comprehensive report data
        report_data = {
            "case_id": case_id,
            "timestamp": created_at_str,
            "status": case.get("status", "unknown"),
            "data_provenance": data_prov,
            "research_demo_warning": research_demo_warning,
            "image_provenance": image_prov,
            "sensor_provenance": sensor_prov,
            "rule_derived_category": rule_category,
            "safety_guidance_level": safety_data.get("rule_derived_category", rule_category),
            "justification": case.get("justification", ""),
            "guidance_mapping_note": f"Research Category ({rule_category}) mapped to Guidance Level ({safety_data.get('rule_derived_category', rule_category)}) based on pain score & fracture risk rules.",
            "prediction_agreement": prediction_agreement_val,
            "uncertainty_level": uncertainty_level_val,
            "uncertainty_status": uncertainty_level_val,
            "uncertainty_reasons": uncertainty_reasons_val,
            "overall_uncertainty": uncertainty_level_val,
            "vision": {
                "finding": finding_label,
                "finding_detected": visible_injury.get("finding_detected", False),
                "confidence": vis_conf,
                "confidence_percentage": f"{vis_conf * 100:.1f}%" if vis_conf is not None else "N/A",
                "confidence_status": "Low Confidence" if vis_is_low_conf else "Confident",
                "affected_ratio": affected_ratio_display,
                "affected_area_ratio": affected_area_ratio_val,
                "segmentation_available": seg_reliable,
                "segmentation_reliable": seg_reliable,
                "segmentation_reason": seg_reason,
                "bounding_box": visible_injury.get("bounding_box", None),
                "segmentation_status": seg_status_display,
                "segmentation_message": seg_reason,
                "overlay_url": visible_injury.get("overlay_url"),
                "mask_url": visible_injury.get("mask_url", ""),
                "gradcam_label": visible_injury.get("gradcam_label") or "MODEL VISUALIZATION",
                "gradcam_explanation": visible_injury.get("gradcam_explanation") or "NOT CLINICAL EXPLANATION",
                "gradcam_source_model": visible_injury.get("gradcam_source_model") or "EfficientNetV2",
                "gradcam_predicted_class": visible_injury.get("gradcam_predicted_class"),
                "gradcam_confidence": visible_injury.get("gradcam_confidence"),
                "gradcam_model_status": visible_injury.get("gradcam_model_status"),
                "gradcam_explanation_status": visible_injury.get("gradcam_explanation_status") or "WITHHELD",
                "gradcam_reliability": visible_injury.get("gradcam_reliability") or "NOT_CLINICAL_EXPLANATION",
                "gradcam_overlay_generated": bool(visible_injury.get("gradcam_overlay_generated")),
            },
            "questionnaire": {
                "answers": questionnaire_answers,
                "template_id": case.get("questionnaire", {}).get("template_id"),
                "template_version": case.get("questionnaire", {}).get("template_version", "1.0"),
                "answer_source": case.get("questionnaire", {}).get("answer_source", "typed")
            },
            "sensor": {
                "provided": case.get("sensor_available", False),
                "source_type": sensor_prov,
                "source_label": "REAL-TIME DEVICE DATA" if sensor_prov == "live" else ("DEMO DATA" if sensor_prov == "demo" else ("SIMULATED DATA" if sensor_prov == "simulated" else ("USER-UPLOADED DATA" if sensor_prov == "uploaded" else "NOT PROVIDED"))),
                "peak_g_force": sensor_sum.get("peak_g_force"),
                "post_impact_stabilization_seconds": sensor_sum.get("post_impact_stabilization_seconds"),
                "pre_impact_delta_v": sensor_sum.get("pre_impact_delta_v"),
                "optical_lux_drop": sensor_sum.get("optical_lux_drop"),
                "predicted_motion_class": sensor_sum.get("predicted_motion_class"),
                "classifier_status": sensor_sum.get("classifier_status"),
                "recording_duration_seconds": sensor_sum.get("recording_duration_seconds"),
                "sample_count": sensor_sum.get("sample_count"),
                "backend_verified_sampling_rate_hz": sensor_sum.get("backend_verified_sampling_rate_hz"),
                "sensor_availability": sensor_sum.get("sensor_availability", {
                    "accelerometer": case.get("sensor_available", False),
                    "gyroscope": case.get("sensor_available", False),
                    "location": False
                })
            },
            "multimodal": {
                "modalities_used": case.get("modalities_used", []),
                "model_configuration_used": case.get("model_configuration_used", "none_registered"),
                "uncertainty_status": uncertainty_level_val,
                "uncertainty_level": uncertainty_level_val,
                "uncertainty_reasons": uncertainty_reasons_val,
                "prediction_agreement": prediction_agreement_val,
                "multimodal_evidence_consistency": f"{consistency_analysis_val.get('score', 100.0)}%"
            },

            "xgboost": case.get("xgboost_prediction") or {},
            "quantum": {
                "class": vqc_prediction_class,
                "experimental_vqc_outputs": case.get("quantum_prediction", {}).get("score") or case.get("quantum_prediction", {}).get("scores"),
                "status": case.get("quantum_prediction", {}).get("status"),
                "error": case.get("quantum_prediction", {}).get("error"),
                "used_in_main_decision": False,
                "model_version": case.get("quantum_prediction", {}).get("model_version", "v1.4.0"),
                "pca_components": None
            },
            "sos": {
                "status": sos_status,
                "delivery_mode": case.get("sos_delivery_mode", "local_demo"),
                "delivery_outcome": case.get("sos_delivery_outcome") or sos_status,
                "twilio_sid": case.get("sos_twilio_sid"),
                "provider_status": case.get("sos_provider_status") or case.get("sos_delivery_status"),
                "timestamp": case.get("sos_send_timestamp"),
                "failure_reason": case.get("sos_failure_reason") or case.get("sos_twilio_error"),
                "disclaimer": "NO REAL EMERGENCY SERVICES WERE CONTACTED. A message SID is required to claim a Twilio request was queued."
            },

            "multimodal_evidence_consistency": consistency_analysis_val,

            "counterfactuals": case.get("counterfactual_analysis") or {
                "explanation": "No counterfactual sweeps performed.",
                "pain_sensitivity": {},
                "g_force_sensitivity": {}
            },
            "safety": safety_data,
            "first_aid_provider": first_aid_data.get("provider", "rule_based_fallback"),
            "first_aid_status": first_aid_data.get("status", "fallback"),
            "first_aid_model": first_aid_data.get("model", "gemini-2.5-flash"),
            "first_aid_evidence_hash": first_aid_data.get("evidence_hash"),
            "structured_evidence": first_aid_data.get("structured_evidence"),
            "first_aid_guidance": first_aid_data
        }
        
        return report_data

    def generate_fhir_bundle(self, case_id: str) -> dict:
        """Generates a standardized HL7 / FHIR R4 Bundle resource for clinical interoperability."""
        report = self.compile_report_data(case_id)
        
        now_iso = datetime.now(timezone.utc).isoformat()
        patient_ref = f"Patient/patient-{case_id[:8]}"
        
        patient_resource = {
            "resourceType": "Patient",
            "id": f"patient-{case_id[:8]}",
            "active": True,
            "gender": "unknown",
            "identifier": [{
                "system": "http://ai-qtriage.org/patients",
                "value": f"ANON-{case_id[:8]}"
            }]
        }
        
        obs_injury = {
            "resourceType": "Observation",
            "id": f"obs-injury-{case_id[:8]}",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "exam",
                    "display": "Exam"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "72170-4",
                    "display": "Photographic image finding"
                }],
                "text": report["vision"].get("finding", "Cut")
            },
            "subject": {"reference": patient_ref},
            "effectiveDateTime": report["timestamp"],
            "valueString": f"Finding: {report['vision'].get('finding')}, Affected Ratio: {report['vision'].get('affected_ratio')}"
        }
        
        obs_triage = {
            "resourceType": "Observation",
            "id": f"obs-triage-{case_id[:8]}",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "triage",
                    "display": "Triage"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "11283-8",
                    "display": "Acuity assessment"
                }],
                "text": "Rule-Derived Triage Category"
            },
            "subject": {"reference": patient_ref},
            "effectiveDateTime": report["timestamp"],
            "valueString": report.get("rule_derived_category", "LOW")
        }

        diagnostic_report = {
            "resourceType": "DiagnosticReport",
            "id": f"report-{case_id[:8]}",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "11526-1",
                    "display": "Pathology study"
                }],
                "text": "AI-QTriage Multimodal Triage Assessment"
            },
            "subject": {"reference": patient_ref},
            "effectiveDateTime": report["timestamp"],
            "issued": now_iso,
            "result": [
                {"reference": f"Observation/obs-injury-{case_id[:8]}"},
                {"reference": f"Observation/obs-triage-{case_id[:8]}"}
            ],
            "conclusion": f"Triage Level: {report.get('rule_derived_category')} | Safety Level: {report.get('safety_guidance_level')}"
        }

        fhir_bundle = {
            "resourceType": "Bundle",
            "id": f"bundle-{case_id}",
            "meta": {
                "lastUpdated": now_iso,
                "profile": ["http://hl7.org/fhir/StructureDefinition/Bundle"]
            },
            "type": "collection",
            "timestamp": now_iso,
            "entry": [
                {"resource": patient_resource},
                {"resource": diagnostic_report},
                {"resource": obs_injury},
                {"resource": obs_triage}
            ]
        }
        return fhir_bundle


    def generate_pdf_bytes(self, case_id: str) -> bytes:
        """
        Generates a professional, multi-page PDF document byte stream using ReportLab.
        Falls back to raw PDF stream if ReportLab encounters an exception.
        """
        report_data = self.compile_report_data(case_id)
        
        try:
            from io import BytesIO
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.pdfgen import canvas

            class UncompressedCanvas(canvas.Canvas):
                def __init__(self, *args, **kwargs):
                    kwargs['pageCompression'] = 0
                    super().__init__(*args, **kwargs)

            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                leftMargin=36,
                rightMargin=36,
                topMargin=36,
                bottomMargin=36
            )

            styles = getSampleStyleSheet()

            # Custom typography styles
            title_style = ParagraphStyle(
                'ReportTitle',
                parent=styles['Heading1'],
                fontName='Helvetica-Bold',
                fontSize=18,
                leading=22,
                textColor=colors.HexColor('#0F172A')
            )
            subtitle_style = ParagraphStyle(
                'ReportSubtitle',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=8.5,
                leading=11,
                textColor=colors.HexColor('#475569')
            )
            h2_style = ParagraphStyle(
                'SectionHeading',
                parent=styles['Heading2'],
                fontName='Helvetica-Bold',
                fontSize=11,
                leading=14,
                textColor=colors.HexColor('#0F172A'),
                spaceBefore=8,
                spaceAfter=5
            )
            body_style = ParagraphStyle(
                'ReportBody',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=8.5,
                leading=12,
                textColor=colors.HexColor('#1E293B')
            )
            table_cell_bold = ParagraphStyle(
                'TableCellBold',
                parent=body_style,
                fontName='Helvetica-Bold',
                textColor=colors.HexColor('#0F172A')
            )
            table_cell = ParagraphStyle(
                'TableCell',
                parent=body_style,
                textColor=colors.HexColor('#334155')
            )
            warning_style = ParagraphStyle(
                'WarningText',
                parent=body_style,
                fontName='Helvetica-Oblique',
                fontSize=8,
                leading=11,
                textColor=colors.HexColor('#7C2D12')
            )

            story = []

            # 1. Header Banner
            story.append(Paragraph("AI-QTriage Research Triage Report", title_style))
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f"<b>Case ID:</b> {case_id} &nbsp;|&nbsp; <b>Timestamp:</b> {report_data['timestamp']} &nbsp;|&nbsp; <b>Data Provenance:</b> {report_data['data_provenance'].upper()}",
                subtitle_style
            ))
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0EA5E9'), spaceBefore=0, spaceAfter=10))

            # 2. Executive Triage Summary Box
            category = report_data.get('rule_derived_category', 'LOW')
            cat_bg = colors.HexColor('#DCFCE7') if category == 'LOW' else (colors.HexColor('#FEF3C7') if category == 'MODERATE' else colors.HexColor('#FEE2E2'))
            cat_fg = colors.HexColor('#166534') if category == 'LOW' else (colors.HexColor('#92400E') if category == 'MODERATE' else colors.HexColor('#991B1B'))

            summary_table_data = [
                [
                    Paragraph(f"<b>Rule-Derived Category:</b> <font color='{cat_fg.hexval()}'><b>{category}</b></font>", body_style),
                    Paragraph(f"<b>Safety Guidance Level:</b> {report_data.get('safety_guidance_level', 'MODERATE')}", body_style)
                ],
                [
                    Paragraph(f"<b>Prediction Agreement:</b> {report_data.get('prediction_agreement', 'AGREEMENT')}", body_style),
                    Paragraph(f"<b>Uncertainty Status:</b> {report_data.get('overall_uncertainty', 'LOW UNCERTAINTY')}", body_style)
                ]
            ]
            summary_table = Table(summary_table_data, colWidths=[270, 270])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), cat_bg),
                ('BOX', (0,0), (-1,-1), 1, cat_fg),
                ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                ('PADDING', (0,0), (-1,-1), 7),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 10))

            # Uncertainty Reasons
            if report_data.get("uncertainty_reasons"):
                story.append(Paragraph("<b>Uncertainty Drivers:</b>", table_cell_bold))
                for r in report_data["uncertainty_reasons"]:
                    story.append(Paragraph(f"• {r}", body_style))
                story.append(Spacer(1, 6))

            # 3. Vision Findings Section
            story.append(Paragraph("1. Visible Image Findings (Vision AI)", h2_style))
            vis_conf_val = report_data['vision'].get('confidence')
            vis_conf_str = f"{vis_conf_val * 100:.1f}%" if vis_conf_val is not None else "N/A"
            aff_ratio = report_data['vision'].get('affected_ratio')
            if aff_ratio == "N/A" or aff_ratio is None:
                aff_str = "Affected Ratio: N/A"
            elif isinstance(aff_ratio, str):
                aff_str = f"Affected Ratio: {aff_ratio}"
            elif isinstance(aff_ratio, (int, float)):
                aff_str = f"Affected Ratio: {aff_ratio * 100:.1f}% ({aff_ratio:.4f})"
            else:
                aff_str = f"Affected Ratio: {aff_ratio}"

            vision_data = [
                [Paragraph("Primary Finding", table_cell_bold), Paragraph(str(report_data['vision'].get('finding', 'N/A')), table_cell)],
                [Paragraph("Detection Confidence", table_cell_bold), Paragraph(f"{vis_conf_str} ({report_data['vision'].get('confidence_status', 'N/A')})", table_cell)],
                [Paragraph("Segmentation Mask Status", table_cell_bold), Paragraph(str(report_data['vision'].get('segmentation_status', 'N/A')), table_cell)],
                [Paragraph("Affected Surface Ratio", table_cell_bold), Paragraph(aff_str, table_cell)],
                [Paragraph("Segmentation Reason", table_cell_bold), Paragraph(str(report_data['vision'].get('segmentation_reason', 'N/A')), table_cell)],
                [Paragraph("Grad-CAM Explanation", table_cell_bold), Paragraph(f"Visual Attention Map (Status: {report_data['vision'].get('gradcam_explanation_status', 'WITHHELD')})", table_cell)]
            ]
            vis_table = Table(vision_data, colWidths=[180, 360])
            vis_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F8FAFC')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
                ('PADDING', (0,0), (-1,-1), 5),
            ]))
            story.append(vis_table)
            story.append(Spacer(1, 10))

            # 4. Sensor Kinetics Section
            story.append(Paragraph("2. Sensor Kinetics & Motion Telemetry", h2_style))
            peak_g = report_data['sensor'].get('peak_g_force')
            peak_g_str = f"{peak_g:.2f} g" if peak_g is not None else "FEATURE_MISSING"
            stab_sec = report_data['sensor'].get('post_impact_stabilization_seconds')
            stab_str = f"{stab_sec:.2f} s" if stab_sec is not None else "FEATURE_MISSING"

            sensor_data = [
                [Paragraph("Peak Impact G-Force", table_cell_bold), Paragraph(peak_g_str, table_cell)],
                [Paragraph("Post-Impact Stabilization Duration", table_cell_bold), Paragraph(stab_str, table_cell)],
                [Paragraph("50Hz Motion Event Classifier Class", table_cell_bold), Paragraph(str(report_data['sensor'].get('predicted_motion_class') or report_data['sensor'].get('classifier_status') or 'FEATURE_MISSING'), table_cell)]
            ]
            sensor_table = Table(sensor_data, colWidths=[180, 360])
            sensor_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F8FAFC')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
                ('PADDING', (0,0), (-1,-1), 5),
            ]))
            story.append(sensor_table)
            story.append(Spacer(1, 10))

            # 5. Classical vs Quantum Section
            story.append(Paragraph("3. Multimodal & Quantum Machine Learning Benchmarks", h2_style))
            q_scores = report_data['quantum'].get('experimental_vqc_outputs')
            q_scores_str = f"{q_scores}" if q_scores is not None else "N/A"

            ml_data = [
                [Paragraph("Classical XGBoost Multimodal Class", table_cell_bold), Paragraph(f"{report_data['xgboost'].get('class', 'N/A')} (training data: {report_data['xgboost'].get('data_provenance', 'synthetic_multimodal_fusion')})", table_cell)],
                [Paragraph("Experimental 4-Qubit VQC Class", table_cell_bold), Paragraph(f"{report_data['quantum'].get('class', 'N/A')} (Status: {report_data['quantum'].get('status', 'ISOLATED')})", table_cell)],
                [Paragraph("VQC Qubit Expectation Scores", table_cell_bold), Paragraph(q_scores_str, table_cell)],
                [Paragraph("Multimodal Evidence Consistency", table_cell_bold), Paragraph(str(report_data['multimodal'].get('multimodal_evidence_consistency', '100%')), table_cell)]
            ]
            ml_table = Table(ml_data, colWidths=[180, 360])
            ml_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F8FAFC')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
                ('PADDING', (0,0), (-1,-1), 5),
            ]))
            story.append(ml_table)
            story.append(Spacer(1, 10))

            # 6. First-Aid Guidance
            story.append(Paragraph("4. Recommended First-Aid Guidance Steps", h2_style))
            first_aid_steps = report_data.get("safety", {}).get("first_aid_steps", [])
            if first_aid_steps:
                for idx, step in enumerate(first_aid_steps, 1):
                    story.append(Paragraph(f"<b>{idx}.</b> {step}", body_style))
                    story.append(Spacer(1, 2))
            else:
                story.append(Paragraph("No specific first-aid steps recorded.", body_style))

            story.append(Spacer(1, 10))

            # 7. Mandatory Disclaimers Box
            story.append(Paragraph("Mandatory Preliminary Research Disclaimers", h2_style))
            disclaimers = report_data.get("safety", {}).get("disclaimers", [])
            disc_text = "<br/>".join([f"• {d}" for d in disclaimers]) if disclaimers else "Research prototype only. Not for clinical diagnosis."
            disc_table = Table([[Paragraph(disc_text, warning_style)]], colWidths=[540])
            disc_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFF7ED')),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#F97316')),
                ('PADDING', (0,0), (-1,-1), 7),
            ]))
            story.append(disc_table)

            doc.build(story, canvasmaker=UncompressedCanvas)
            pdf_bytes = buffer.getvalue().rstrip()
            buffer.close()
            return pdf_bytes

        except Exception as err:
            print(f"[PDF Generator Fallback] ReportLab exception: {err}")
            # Fallback raw PDF stream if ReportLab fails
            vis_conf_val = report_data['vision'].get('confidence')
            vis_conf_str = f"{vis_conf_val * 100:.1f}%" if vis_conf_val is not None else "N/A"
            aff_ratio_str = str(report_data['vision'].get('affected_ratio', 'N/A'))

            title = f"AI-QTriage Research Report - Case: {case_id}"
            body_lines = [
                f"Case ID: {case_id}",
                f"Timestamp: {report_data['timestamp']}",
                f"Data Provenance: {report_data['data_provenance'].upper()}",
            ]
            if report_data.get("research_demo_warning"):
                body_lines.append(f"Notice: {report_data['research_demo_warning']}")

            body_lines.extend([
                f"Rule-Derived Research Category: {report_data['rule_derived_category']}",
                f"Safety Guidance Level: {report_data.get('safety_guidance_level', 'MODERATE')}",
                f"Prediction Agreement: {report_data.get('prediction_agreement', 'AGREEMENT')}",
                f"Overall Research Uncertainty: {report_data.get('overall_uncertainty', 'LOW UNCERTAINTY')}",
                "",
                "--- VISIBLE IMAGE FINDINGS ---",
                f"Finding: {report_data['vision']['finding']}",
                f"Confidence: {vis_conf_str}",
                f"Confidence Status: {report_data['vision'].get('confidence_status', 'Confident')}",
                f"Segmentation Status: {report_data['vision'].get('segmentation_status', 'Not confidently detected')}",
                f"Affected Ratio: {aff_ratio_str}",
                "",
                "--- SENSOR KINETICS ---",
                f"Peak G-Force: {report_data['sensor'].get('peak_g_force')}",
                f"Stabilization Time: {report_data['sensor'].get('post_impact_stabilization_seconds')}",
                f"Motion Classifier: {report_data['sensor'].get('predicted_motion_class')}",
                "",
                "--- CLASSICAL VS QUANTUM ML ---",
                f"XGBoost Prediction: {report_data['xgboost'].get('class', 'N/A')}",
                f"Experimental VQC Output: {report_data['quantum'].get('class')}",
                "",
                "--- FIRST-AID SAFETY GUIDANCE ---"
            ])
            for step in report_data["safety"]["first_aid_steps"]:
                body_lines.append(f"- {step}")

            pdf = [
                b"%PDF-1.4",
                b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj",
                b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj",
                b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj",
                b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj"
            ]
            text_stream = ["BT", "/F1 16 Tf", "50 800 Td", f"({title}) Tj", "/F1 10 Tf", "0 -30 Td"]
            for line in body_lines:
                safe_line = line.replace("(", "\\(").replace(")", "\\)")
                text_stream.append(f"({safe_line}) Tj")
                text_stream.append("0 -15 Td")
            text_stream.append("ET")
            content_bytes = "\n".join(text_stream).encode("utf-8")
            pdf.append(f"4 0 obj\n<< /Length {len(content_bytes)} >>\nstream\n".encode("utf-8") + content_bytes + b"\nendstream\nendobj")
            pdf.append(b"xref\n0 6\n0000000000 65535 f \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF")
            return b"\n".join(pdf)
