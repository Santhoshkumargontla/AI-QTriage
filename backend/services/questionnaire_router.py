"""
questionnaire_router.py

Deterministic questionnaire template routing with optional Gemini enhancement.

Architecture:
  Vision result → Deterministic class-to-template mapping (always works)
  Vision result → Gemini structured routing → Approved template (optional)

Gemini is NEVER used to diagnose images or generate novel medical questions.
Gemini only selects which pre-approved template to load from:
  cut.json / bruise.json / swelling.json

Gemini output is validated against a strict schema before being accepted.
If Gemini is unavailable or returns invalid output, deterministic routing is used.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
# Prototype routing threshold. This is NOT a medical confidence threshold.
# It is only used for questionnaire template selection.
QUESTIONNAIRE_ROUTING_THRESHOLD = float(os.environ.get("QUESTIONNAIRE_ROUTING_THRESHOLD", "0.40"))

SUPPORTED_TEMPLATES = {"cut", "bruise", "swelling"}

TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "templates", "questionnaire_templates"
)

# Deterministic class → template mapping (includes YOLO / EfficientNet aliases)
CLASS_TO_TEMPLATE: dict[str, Optional[str]] = {
    "cut": "cut",
    "bruise": "bruise",
    "swelling": "swelling",
    "abrasion": "cut",
    "laceration": "cut",
    "wound": "cut",
    "burn": "cut",
    "other": None,  # No template for "other"
}


def load_template(template_id: str) -> Optional[dict]:
    """Load a questionnaire template JSON file by ID. Returns None if not found."""
    if template_id not in SUPPORTED_TEMPLATES:
        return None
    path = os.path.join(TEMPLATES_DIR, f"{template_id}.json")
    if not os.path.exists(path):
        logger.error(f"Template file missing: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _deterministic_route(finding: str, classification: Optional[dict] = None) -> dict:
    """
    Deterministic fallback: maps vision class to approved template.
    Returns routing result dict.
    """
    finding_lower = finding.lower().strip() if finding else ""
    template_id = CLASS_TO_TEMPLATE.get(finding_lower)

    # If classification probabilities are provided and max is below threshold,
    # mark as uncertain so the frontend can ask the user to clarify.
    is_uncertain = False
    if classification:
        probs = [
            float(v)
            for v in classification.values()
            if isinstance(v, (int, float)) and v is not None
        ]
        max_prob = max(probs) if probs else 0.0
        if max_prob < QUESTIONNAIRE_ROUTING_THRESHOLD:
            is_uncertain = True

    if template_id is None:
        return {
            "routed": False,
            "template_id": None,
            "template": None,
            "router_used": "deterministic",
            "is_uncertain": is_uncertain,
            "reason": f"No approved questionnaire template exists for finding: '{finding}'"
        }

    template = load_template(template_id)
    return {
        "routed": bool(template),
        "template_id": template_id,
        "template": template,
        "router_used": "deterministic",
        "is_uncertain": is_uncertain,
        "reason": f"Deterministic routing: '{finding}' → template '{template_id}'"
    }


def _gemini_route(finding: str, confidence: float, classification: Optional[dict]) -> Optional[dict]:
    """
    Optional Gemini-assisted routing. Gemini receives structured finding metadata only —
    NOT the image. Returns a template selection, validated against strict schema.
    Returns None if Gemini is unavailable or returns invalid output.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.debug("GEMINI_API_KEY not set — skipping Gemini routing.")
        return None

    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
    except ImportError:
        logger.debug("google-generativeai SDK not installed — skipping Gemini routing.")
        return None

    prompt = f"""You are a questionnaire routing assistant for a research prototype.
You are given the result of a computer vision pipeline (YOLO11 + EfficientNetV2).
Your ONLY job is to select the correct pre-approved questionnaire template.

Detected finding: {finding}
Confidence: {confidence:.3f}
Classification probabilities: {json.dumps(classification or {})}

Available templates:
- cut
- bruise  
- swelling

Rules:
1. You MUST select from exactly one of: cut, bruise, swelling
2. Do NOT diagnose the image
3. Do NOT generate medical questions
4. Do NOT suggest fracture, internal bleeding, or any clinical finding
5. Return ONLY valid JSON in this exact schema:
{{"template_id": "<cut|bruise|swelling>", "reason": "<one sentence>"}}

If classification is ambiguous (no class clearly dominant), still pick the most likely one.
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=128,
            )
        )
        text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        parsed = json.loads(text)

        # Strict schema validation
        if not isinstance(parsed, dict):
            raise ValueError("Response is not a dict")
        if "template_id" not in parsed:
            raise ValueError("Missing template_id")
        template_id = str(parsed["template_id"]).lower().strip()
        if template_id not in SUPPORTED_TEMPLATES:
            raise ValueError(f"Unsupported template_id returned by Gemini: {template_id}")

        template = load_template(template_id)
        return {
            "routed": bool(template),
            "template_id": template_id,
            "template": template,
            "router_used": "gemini",
            "is_uncertain": False,
            "reason": str(parsed.get("reason", "Gemini routing"))
        }

    except Exception as e:
        logger.warning(f"Gemini routing failed ({type(e).__name__}: {e}) — falling back to deterministic.")
        return None


def route_questionnaire(
    finding: str,
    confidence: float = 1.0,
    classification: Optional[dict] = None
) -> dict:
    """
    Main entry point for questionnaire routing.

    Tries Gemini first (if configured), then falls back to deterministic routing.
    Gemini receives ONLY structured metadata — never the image itself.

    Args:
        finding: Detected injury class (e.g. "swelling", "cut", "bruise")
        confidence: Model confidence score for the top finding
        classification: Full classification probability dict

    Returns:
        {
            "routed": bool,
            "template_id": str | None,
            "template": dict | None,
            "router_used": "deterministic" | "gemini",
            "is_uncertain": bool,
            "reason": str,
            "gemini_used": bool,
            "gemini_available": bool
        }
    """
    gemini_available = bool(
        (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    )

    # Try Gemini first
    result = None
    if gemini_available:
        result = _gemini_route(finding, confidence, classification)

    if result is None:
        # Fall back to deterministic
        result = _deterministic_route(finding, classification)
        result["gemini_available"] = gemini_available
        result["gemini_used"] = False
        if gemini_available and result["router_used"] == "deterministic":
            result["gemini_fallback_reason"] = "Gemini unavailable. Using standard questionnaire."
    else:
        result["gemini_available"] = gemini_available
        result["gemini_used"] = True

    return result
