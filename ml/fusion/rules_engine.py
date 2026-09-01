import numpy as np

class RulesEngine:
    """Evaluates fused feature vectors to assign a 'Rule-Derived Research Category'."""
    
    def __init__(self):
        pass

    def evaluate_rules(self, fused_vector: np.ndarray, feature_names: list) -> tuple:
        """
        Evaluates the fused feature vector and returns the label and the justification.
        Returns:
            - label (str): "LOW", "MODERATE", or "HIGH"
            - justification (str): Explain which rule triggered the category.
        """
        # Convert vector to quick lookup dictionary
        feat = {name: float(fused_vector[i]) for i, name in enumerate(feature_names)}

        # Helper lookups
        pain = round(feat.get("pain_level", 0.5) * 10.0)  # Scale back to 0-10 integer
        bleeding = feat.get("visible_bleeding", 0.0) > 0.5
        g_force = feat.get("peak_g_force", 0.0)
        movement = feat.get("movement_limitation", 0.0)
        weight_bearing = feat.get("weight_bearing", 0.0)
        crack_pop = feat.get("crack_pop", 0.0) > 0.5
        direct_impact = feat.get("direct_impact", 0.0) > 0.5
        
        # --- Rule 1: HIGH Rule-Derived Research Category ---
        if crack_pop and (movement >= 1.0 or weight_bearing >= 1.0):
            return "HIGH", "Assigned HIGH category: User reported hearing/feeling a crack/pop accompanied by severe movement or weight-bearing limitations."
            
        if bleeding and pain >= 7.0:
            return "HIGH", f"Assigned HIGH category: Visible active bleeding reported with high subjective pain score ({pain:.0f}/10)."
            
        if g_force >= 5.0 and pain >= 6.0:
            return "HIGH", f"Assigned HIGH category: High kinetic impact peak ({g_force:.2f}g) measured by sensors coupled with elevated pain levels ({pain:.0f}/10)."

        # --- Rule 2: MODERATE Rule-Derived Research Category ---
        reasons = []
        if g_force >= 3.0:
            reasons.append(f"moderate impact peak ({g_force:.2f}g)")
        if pain >= 4.0:
            reasons.append(f"moderate pain level ({pain:.0f}/10)")
        if movement >= 0.5:
            reasons.append("mild/severe movement limitation")
        if weight_bearing >= 0.5:
            reasons.append("partial/full weight-bearing limitation")
        if bleeding:
            reasons.append("visible bleeding")
        if direct_impact:
            reasons.append("reported direct impact to the area")

        if reasons:
            justification = "Assigned MODERATE category based on: " + ", ".join(reasons) + "."
            return "MODERATE", justification

        # --- Rule 3: LOW Rule-Derived Research Category ---
        return "LOW", f"Assigned LOW category: Normal kinetic stabilization, mild pain ({pain:.0f}/10), and no bleeding or structural crack/pop reported."
