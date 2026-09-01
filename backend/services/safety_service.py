from backend.database.connection import get_database

class SafetyGuidanceService:
    """Retrieves predefined database first-aid steps and enforces critical medical disclaimers."""
    
    def __init__(self):
        pass

    def get_safety_guidance(self, finding: str, has_fracture_risk: bool = False, severity_category: str = "LOW") -> dict:
        """
        Fetches static first-aid instructions from MongoDB and appends required disclaimers.
        """
        db = get_database()
        finding_clean = finding.lower().strip()
        
        # Query predefined static templates from safety_guidance collection
        guidance_doc = db.safety_guidance.find_one({"finding": finding_clean})
        
        # If not found, load a default structural template (to prevent failures)
        if not guidance_doc:
            guidance_doc = {
                "finding": finding_clean,
                "steps": [
                    "Keep the affected area clean and dry.",
                    "Monitor for signs of localized discomfort or changes."
                ],
                "red_flags": [
                    "Increasing redness, warmth, or swelling."
                ]
            }

        # 1. Base Predefined Data
        first_aid_steps = list(guidance_doc.get("steps", []))
        red_flags = list(guidance_doc.get("red_flags", []))
        
        # 2. Enforce Mandatory Safety Disclaimers
        disclaimers = [
            "This is a research prototype, not a medical diagnostic device. Please consult a healthcare professional."
        ]
        
        # 3. Enforce Fracture warning if high risk
        # Triggered if crack/pop is True, VQC/XGBoost is MODERATE/HIGH, or explicitly flagged
        if has_fracture_risk or severity_category in ["MODERATE", "HIGH"]:
            disclaimers.append(
                "An ordinary RGB photograph cannot reliably determine a fracture. Appropriate medical imaging and professional assessment are required."
            )

        return {
            "finding": finding_clean,
            "rule_derived_category": severity_category,
            "first_aid_steps": first_aid_steps,
            "red_flags": red_flags,
            "disclaimers": disclaimers
        }
