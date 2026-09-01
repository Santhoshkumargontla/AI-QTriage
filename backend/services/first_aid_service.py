"""
First-Aid Guidance Service for AI-QTriage Research Prototype.
Generates conservative, basic first-aid suggestions using:
1. StructuredEvidenceBuilder (compiles single canonical evidence object & SHA-256 hash)
2. GeminiFirstAidService (server-side Gemini natural-language guidance generator via google-genai SDK)
3. RuleBasedFirstAidService (deterministic safety authority & fallback engine)

RESEARCH PROTOTYPE ONLY: General first-aid research suggestions only. Not a clinical diagnosis.
"""

import os
import json
from typing import Dict, Any, List, Optional
from backend.services.gemini_first_aid_service import GeminiFirstAidService, compute_evidence_hash, get_gemini_config


class StructuredEvidenceBuilder:
    """Compiles a single canonical structured evidence object from case inputs."""

    @staticmethod
    def build_evidence(
        questionnaire_answers: Dict[str, Any] = None,
        sensor_summary: Dict[str, Any] = None,
        visible_injury: Dict[str, Any] = None,
        rule_derived_category: str = "LOW",
        xgboost_pred: Dict[str, Any] = None,
        quantum_pred: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        answers = questionnaire_answers or {}
        sensor = sensor_summary or {}
        vision = visible_injury or {}
        xgb = xgboost_pred or {}
        vqc = quantum_pred or {}

        # 1. Canonical Questionnaire Data (Preserves null / not_provided)
        clean_answers = {}
        for k in [
            "pain_level", "location", "cause", "onset_hours", "movement_limitation",
            "weight_bearing", "swelling", "bruising_discoloration", "redness", "warmth",
            "open_wound", "bleeding", "crack_pop", "deformity", "numbness_tingling",
            "previous_injury", "symptom_progression"
        ]:
            val = answers.get(k)
            if val is None or str(val).strip() == "" or str(val).lower() == "not_provided":
                clean_answers[k] = None
            elif k in ("pain_level", "pain"):
                try:
                    clean_answers["pain_level"] = int(val)
                except (ValueError, TypeError):
                    clean_answers["pain_level"] = None
            else:
                clean_answers[k] = val

        # Handle legacy pain key if pain_level missing
        if clean_answers.get("pain_level") is None and answers.get("pain") is not None:
            try:
                clean_answers["pain_level"] = int(answers["pain"])
            except (ValueError, TypeError):
                clean_answers["pain_level"] = None

        # 2. YOLO11 Evidence (Strict separation — swelling is NEVER a YOLO class)
        try:
            from ml.vision.yolo_wrapper import YOLO11Detector

            YOLO_CLASSES = [str(n).lower() for n in YOLO11Detector().class_list]
        except Exception:
            YOLO_CLASSES = ["cut", "bruise", "abrasion"]
        yolo_finding_detected = bool(vision.get("yolo_finding_detected", False) or vision.get("finding_detected", False))
        yolo_finding = vision.get("yolo_finding") or vision.get("finding")
        
        # Verify YOLO finding is in supported YOLO classes
        if yolo_finding and str(yolo_finding).lower() not in YOLO_CLASSES:
            yolo_finding = None
            yolo_finding_detected = False

        yolo_conf = vision.get("yolo_confidence")
        if yolo_conf is None:
            yolo_conf = vision.get("confidence")
        yolo_data = {
            "finding_detected": yolo_finding_detected,
            "finding": yolo_finding if yolo_finding_detected else None,
            "confidence": yolo_conf if yolo_finding_detected else None,
            "bounding_box": vision.get("yolo_bbox") or vision.get("bounding_box"),
            "supported_classes": YOLO_CLASSES
        }

        # 3. EfficientNet Research Classifier Evidence
        # Never fall back to YOLO finding/confidence — that conflates detector with classifier.
        classifier_finding = vision.get("classifier_finding")
        classifier_conf = vision.get("classifier_probability")
        if classifier_conf is None:
            classifier_conf = vision.get("classifier_confidence")
        classifier_data = {
            "finding": classifier_finding if classifier_finding not in (None, "Unknown", "Unspecified") else None,
            "confidence": classifier_conf,
            "category_type": "research_classifier",
            "status": vision.get("classifier_status"),
        }

        # 4. Smartphone Sensor Log Evidence
        sensor_provided = bool(sensor and sensor.get("provided", True) and sensor.get("source_type") != "not_provided")
        sensor_data = {
            "provided": sensor_provided,
            "peak_g_force": (
                float(sensor["peak_g_force"])
                if sensor_provided and sensor.get("peak_g_force") is not None
                else None
            ),
            "post_impact_stabilization_seconds": (
                float(sensor["post_impact_stabilization_seconds"])
                if sensor_provided and sensor.get("post_impact_stabilization_seconds") is not None
                else None
            ),
        }

        # 5. Experimental ML Model Predictions
        xgb_class = xgb.get("class")
        vqc_class = vqc.get("class")
        vqc_excluded = (
            vqc.get("used_in_main_decision") is False
            or vqc.get("experimental_only") is True
            or vqc.get("status") in ("MODEL_UNAVAILABLE", "EXPERIMENTAL_ONLY")
            or vqc_class is None
        )
        if vqc_excluded:
            agreement = "VQC_EXCLUDED_FROM_DECISION"
        else:
            agreement = "AGREEMENT" if xgb_class == vqc_class else "DISAGREEMENT"

        experimental_data = {
            "xgboost": xgb_class,
            "vqc": vqc_class,
            "agreement": agreement,
            "uncertainty": None if vqc_excluded else ("HIGH" if agreement == "DISAGREEMENT" else "LOW"),
            "used_in_main_decision": False,
        }

        # 6. Urgent Warning Signs Checklist for Safety Rules
        warning_signs = []
        pain_level = clean_answers.get("pain_level") or 0
        if clean_answers.get("deformity") == "yes":
            warning_signs.append("Visible anatomical deformity or misalignment reported.")
        if clean_answers.get("numbness_tingling") == "yes":
            warning_signs.append("Numbness, tingling, or altered nerve sensation reported.")
        if str(clean_answers.get("bleeding") or "").lower() in ("active", "heavy", "uncontrolled"):
            warning_signs.append("Active or heavy bleeding reported.")
        if clean_answers.get("crack_pop") == "yes":
            warning_signs.append("Audible/felt crack or popping sensation at time of impact.")
        if pain_level >= 8:
            warning_signs.append(f"Severe subjective pain intensity score ({pain_level}/10).")

        safety_rules_data = {
            "guidance_level": rule_derived_category,
            "warning_signs": warning_signs
        }

        return {
            "questionnaire": clean_answers,
            "yolo": yolo_data,
            "research_classifier": classifier_data,
            "sensor": sensor_data,
            "experimental_models": experimental_data,
            "safety_rules": safety_rules_data
        }


class FirstAidGuidanceService:
    """Orchestrates first-aid guidance generation via Gemini or Rule-Based Fallback."""

    def __init__(self):
        self.gemini_service = GeminiFirstAidService()

    def generate_first_aid_guidance(
        self,
        questionnaire_answers: Dict[str, Any] = None,
        sensor_summary: Dict[str, Any] = None,
        visible_injury: Dict[str, Any] = None,
        rule_derived_category: str = "LOW",
        xgboost_pred: Dict[str, Any] = None,
        quantum_pred: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        # 1. Build Canonical Structured Evidence
        evidence = StructuredEvidenceBuilder.build_evidence(
            questionnaire_answers=questionnaire_answers,
            sensor_summary=sensor_summary,
            visible_injury=visible_injury,
            rule_derived_category=rule_derived_category,
            xgboost_pred=xgboost_pred,
            quantum_pred=quantum_pred
        )

        # 2. Compute Evidence Hash
        evidence_hash = compute_evidence_hash(evidence)

        # 3. Always Generate Deterministic Fallback Object for Safety Priority
        deterministic_fallback = self._generate_deterministic_guidance(evidence, rule_derived_category)

        # 4. Attempt Gemini Generation
        gemini_result = self.gemini_service.generate_guidance(evidence, timeout_seconds=8.0)

        # 5. If Gemini succeeds, merge safety rule warnings and return
        if gemini_result.get("provider") == "gemini" and gemini_result.get("status") == "success":
            guidance = gemini_result.get("guidance", {})
            
            # Ensure deterministic urgent warnings are NOT dropped by Gemini
            deterministic_warnings = deterministic_fallback.get("urgent_warning_signs", [])
            gemini_warnings = guidance.get("urgent_evaluation_warning", [])
            merged_warnings = list(dict.fromkeys(deterministic_warnings + gemini_warnings))

            return {
                "provider": "gemini",
                "status": "success",
                "model": gemini_result.get("model", "gemini-2.5-flash"),
                "guidance_level": deterministic_fallback.get("guidance_level", "LOW"),
                "generated_at": gemini_result.get("generated_at"),
                "evidence_hash": evidence_hash,
                "display_message": gemini_result.get("display_message"),
                "structured_evidence": evidence,
                "guidance": {
                    "immediate_first_aid_steps": guidance.get("immediate_first_aid_steps", []),
                    "actions_to_avoid": guidance.get("actions_to_avoid", []),
                    "symptoms_to_monitor": guidance.get("symptoms_to_monitor", []),
                    "urgent_evaluation_warning": merged_warnings,
                    "professional_evaluation_guidance": guidance.get("professional_evaluation_guidance", []),
                    "limitations": guidance.get("limitations", [
                        "AI-QTriage is an academic research prototype and does not provide clinical medical diagnosis."
                    ])
                },
                # Flat keys for report parity & legacy tests
                "evidence_summary": deterministic_fallback.get("evidence_summary", []),
                "immediate_steps": guidance.get("immediate_first_aid_steps", []),
                "avoid": guidance.get("actions_to_avoid", []),
                "monitor": guidance.get("symptoms_to_monitor", []),
                "seek_professional_evaluation": deterministic_fallback.get("seek_professional_evaluation", False),
                "urgent_warning_signs": merged_warnings,
                "professional_evaluation_warning": deterministic_fallback.get("professional_evaluation_warning"),
                "model_limitation_statement": "AI-QTriage is an academic research prototype. Model outputs are experimental and are not a medical diagnosis."
            }

        # 6. If Gemini returned fallback or error, return rule-based fallback guidance
        return {
            "provider": "rule_based_fallback",
            "status": "fallback",
            "model": self.gemini_service.model_name,
            "fallback_reason": gemini_result.get("fallback_reason", "Gemini API unavailable"),
            "guidance_level": deterministic_fallback.get("guidance_level", "LOW"),
            "generated_at": gemini_result.get("generated_at"),
            "evidence_hash": evidence_hash,
            "display_message": "AI-generated guidance unavailable. Showing rule-based research guidance.",
            "structured_evidence": evidence,
            "guidance": {
                "immediate_first_aid_steps": deterministic_fallback.get("immediate_steps", []),
                "actions_to_avoid": deterministic_fallback.get("avoid", []),
                "symptoms_to_monitor": deterministic_fallback.get("monitor", []),
                "urgent_evaluation_warning": deterministic_fallback.get("urgent_warning_signs", []),
                "professional_evaluation_guidance": [deterministic_fallback.get("professional_evaluation_warning", "Seek medical evaluation if symptoms worsen.")],
                "limitations": [deterministic_fallback.get("model_limitation_statement")]
            },
            # Flat keys for report parity & legacy tests
            "evidence_summary": deterministic_fallback.get("evidence_summary", []),
            "immediate_steps": deterministic_fallback.get("immediate_steps", []),
            "avoid": deterministic_fallback.get("avoid", []),
            "monitor": deterministic_fallback.get("monitor", []),
            "seek_professional_evaluation": deterministic_fallback.get("seek_professional_evaluation", False),
            "urgent_warning_signs": deterministic_fallback.get("urgent_warning_signs", []),
            "professional_evaluation_warning": deterministic_fallback.get("professional_evaluation_warning"),
            "model_limitation_statement": deterministic_fallback.get("model_limitation_statement")
        }

    def _generate_deterministic_guidance(self, evidence: Dict[str, Any], rule_derived_category: str) -> Dict[str, Any]:
        """Deterministic rule-based fallback guidance engine."""
        q = evidence.get("questionnaire", {})
        yolo = evidence.get("yolo", {})
        clf = evidence.get("research_classifier", {})
        sensor = evidence.get("sensor", {})
        models = evidence.get("experimental_models", {})

        pain_level = q.get("pain_level") or 0
        location = q.get("location") or "injured area"
        
        swelling = q.get("swelling") == "yes"
        open_wound = q.get("open_wound") == "yes"
        bleeding_val = str(q.get("bleeding") or "none").lower()
        crack_pop = q.get("crack_pop") == "yes"
        deformity = q.get("deformity") == "yes"
        numbness = q.get("numbness_tingling") == "yes"
        movement = str(q.get("movement_limitation") or "none").lower()
        weight_bearing = str(q.get("weight_bearing") or "none").lower()

        # Evidence Summary
        evidence_summary = []
        if pain_level > 0:
            evidence_summary.append(f"User-reported pain level: {pain_level}/10")
        if location and location != "not_provided":
            evidence_summary.append(f"User-reported location: {location}")
        if swelling:
            evidence_summary.append("User-reported swelling: yes")
        if open_wound:
            evidence_summary.append("User-reported open wound: yes")
        if bleeding_val not in ("none", "not_provided"):
            evidence_summary.append(f"User-reported bleeding: {bleeding_val}")
        if crack_pop:
            evidence_summary.append("User-reported crack/pop sensation: yes")
        if deformity:
            evidence_summary.append("User-reported deformity: yes")
        if numbness:
            evidence_summary.append("User-reported numbness/tingling: yes")

        if yolo.get("finding_detected") and yolo.get("finding"):
            conf = yolo.get("confidence")
            conf_str = f" ({conf * 100:.1f}%)" if conf else ""
            evidence_summary.append(f"YOLO11 object detection: {yolo['finding']}{conf_str}")
        else:
            classes = yolo.get("supported_classes") or ["cut", "bruise", "abrasion"]
            evidence_summary.append(
                "YOLO11 object detection: No confident supported-class detection "
                f"(supported: {', '.join(classes)})."
            )

        if clf.get("finding"):
            conf = clf.get("confidence")
            conf_str = f" ({conf * 100:.1f}%)" if conf else ""
            evidence_summary.append(f"Research image classifier category: {clf['finding']}{conf_str}")

        if sensor.get("provided"):
            g = sensor.get("peak_g_force")
            if g is None:
                evidence_summary.append("Sensor log provided; peak_g_force FEATURE_MISSING.")
            else:
                evidence_summary.append(f"Sensor log measured peak impact: {g:.2f}g.")
        else:
            evidence_summary.append("Sensor data was not provided; guidance is based on image/questionnaire evidence.")

        if models.get("agreement") == "DISAGREEMENT":
            evidence_summary.append(f"Model agreement: DISAGREEMENT (XGBoost predicted {models.get('xgboost')}, VQC predicted {models.get('vqc')}). Experimental model outputs disagree; this increases research uncertainty.")

        # Urgent Warning Signs
        urgent_warning_signs = []
        if deformity:
            urgent_warning_signs.append("Visible anatomical deformity or misalignment reported.")
        if numbness:
            urgent_warning_signs.append("Numbness, tingling, or altered nerve sensation reported.")
        if bleeding_val in ("active", "heavy", "uncontrolled"):
            urgent_warning_signs.append("Active or heavy bleeding reported.")
        if movement in ("severe", "cannot_move") or weight_bearing in ("unable", "no"):
            urgent_warning_signs.append("Inability to move or bear weight on the injured area.")
        if crack_pop:
            urgent_warning_signs.append("Audible/felt crack or popping sensation at time of impact.")
        if pain_level >= 8:
            urgent_warning_signs.append(f"Severe subjective pain intensity score ({pain_level}/10).")

        # Immediate Steps
        immediate_steps = []
        if open_wound or bleeding_val != "none":
            immediate_steps.append("Clean the open area gently with clean water, apply a sterile protective dressing, and apply light pressure if bleeding is active.")
        else:
            immediate_steps.append("Keep the affected area clean, dry, and protected from secondary impact or friction.")

        if swelling or clf.get("finding") == "Swelling":
            immediate_steps.append("Rest the affected area in a comfortable, elevated position and consider applying a wrapped cold compress for 15–20 minutes.")
        elif pain_level >= 4:
            immediate_steps.append("Allow the affected area to rest in a comfortable, supported position to avoid strain.")

        if movement in ("moderate", "severe", "limited") or weight_bearing in ("partial", "unable"):
            immediate_steps.append("Immobilize or minimize active movement of the joint/limb to prevent aggravating the area.")

        if pain_level <= 3 and not open_wound and not swelling and not deformity:
            immediate_steps.append("Monitor the area closely while resting. Avoid unnecessary exertion until symptoms fully resolve.")

        # Actions to Avoid
        avoid = []
        if deformity or crack_pop or pain_level >= 8:
            avoid.append("Avoid attempting to bend, manipulate, or forcefully straighten a visibly deformed or severely painful joint/limb.")
        avoid.append("Avoid forcing joint movement or aggressively testing weight-bearing capacity.")
        if open_wound or bleeding_val != "none":
            avoid.append("Avoid applying unsterile powders, home remedies, or touching open wound edges with unwashed hands.")
        if swelling:
            avoid.append("Avoid applying direct heat or intense massage immediately following an acute impact.")
        avoid.append("Avoid continuing sport, running, or heavy physical activity while pain or swelling persists.")

        # Monitoring
        monitor = [
            "Monitor for expanding redness, localized warmth, or increasing tissue swelling over the next 24–48 hours.",
            "Watch for infection indicators such as pus, drainage, red streaks, or systemic fever.",
            "Observe pain trends over time (note if pain score progresses from mild to severe)."
        ]
        if numbness:
            monitor.append("Monitor for worsening numbness, tingling, or loss of motor function in extremities.")

        seek_professional = bool(
            urgent_warning_signs or
            rule_derived_category in ("MODERATE", "HIGH") or
            pain_level >= 6 or
            crack_pop or
            deformity or
            open_wound
        )

        if urgent_warning_signs or rule_derived_category == "HIGH":
            guidance_level = "URGENT_EVALUATION"
        elif seek_professional or rule_derived_category == "MODERATE":
            guidance_level = "CONSERVATIVE_CARE"
        else:
            guidance_level = "BASIC_REST_MONITOR"

        return {
            "guidance_level": guidance_level,
            "evidence_summary": evidence_summary,
            "immediate_steps": immediate_steps,
            "avoid": avoid,
            "monitor": monitor,
            "seek_professional_evaluation": seek_professional,
            "urgent_warning_signs": urgent_warning_signs,
            "professional_evaluation_warning": "Based on the reported warning signs, professional medical evaluation is recommended." if seek_professional else "If symptoms worsen or fail to improve, professional evaluation is recommended.",
            "model_limitation_statement": "AI-QTriage is an academic research prototype. Model outputs are experimental and are not a medical diagnosis. An image model failing to detect an injury does not mean that no injury is present."
        }


first_aid_service = FirstAidGuidanceService()
