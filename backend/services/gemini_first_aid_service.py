"""
Gemini First-Aid Guidance Service using official Google GenAI Python SDK (google-genai).

RESEARCH PROTOTYPE ONLY: Generates conservative, evidence-grounded basic first-aid suggestions.
Not a clinical diagnosis.

System Instructions:
- Conservative first-aid research guidance only.
- NOT a doctor. You must NOT diagnose.
- You must NOT identify fractures, internal injuries, infections, or other medical conditions from an image.
- You must NOT invent symptoms, measurements, history, model findings, or questionnaire answers.
- Use ONLY the supplied StructuredEvidence object.
- Never modify or reinterpret user questionnaire values.
- Do NOT claim YOLO detected an injury unless yolo.finding_detected=true.
- Do NOT treat EfficientNet research categories as YOLO detections.
- Do NOT override deterministic safety rules.
- Always return valid JSON matching the requested schema.
"""

import os
import json
import hashlib
import concurrent.futures
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import official Google GenAI Python SDK
try:
    from google import genai
    from google.genai import types
    GENAI_SDK_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_SDK_AVAILABLE = False


# Load environment variables from backend/.env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


def get_gemini_config() -> Dict[str, Any]:
    """Returns Gemini configuration status without exposing the API key."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    configured = bool(api_key and len(str(api_key).strip()) > 5 and GENAI_SDK_AVAILABLE)
    
    return {
        "configured": configured,
        "provider": "Google Gemini",
        "model": model_name,
        "sdk_available": GENAI_SDK_AVAILABLE,
        "fallback": "rule_based"
    }


def compute_evidence_hash(evidence: Dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 hash of the canonical StructuredEvidence object."""
    dumped = json.dumps(evidence, sort_keys=True, default=str)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]


class GeminiFirstAidService:
    """Invokes Gemini API via official google-genai SDK for first-aid guidance."""

    def __init__(self):
        self.model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def _get_api_key(self) -> Optional[str]:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def generate_guidance(
        self,
        structured_evidence: Dict[str, Any],
        timeout_seconds: float = 12.0
    ) -> Dict[str, Any]:
        """
        Invokes Gemini with canonical StructuredEvidence.
        Returns provider response metadata with fallback handling on any error/timeout.
        """
        evidence_hash = compute_evidence_hash(structured_evidence)
        generated_at = datetime.now(timezone.utc).isoformat()
        api_key = self._get_api_key()

        if not api_key:
            return {
                "provider": "rule_based_fallback",
                "status": "fallback",
                "model": self.model_name,
                "fallback_reason": "GEMINI_API_KEY not configured in backend/.env",
                "generated_at": generated_at,
                "evidence_hash": evidence_hash,
                "display_message": "AI-generated guidance unavailable. Showing rule-based research guidance."
            }

        if not GENAI_SDK_AVAILABLE:
            return {
                "provider": "rule_based_fallback",
                "status": "fallback",
                "model": self.model_name,
                "fallback_reason": "google-genai SDK not installed",
                "generated_at": generated_at,
                "evidence_hash": evidence_hash,
                "display_message": "AI-generated guidance unavailable. Showing rule-based research guidance."
            }

        # Execute call with strict ThreadPoolExecutor timeout (8s)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._call_gemini_sdk, structured_evidence, api_key)
            try:
                result = future.result(timeout=timeout_seconds)
                if result and result.get("status") == "success":
                    result["generated_at"] = generated_at
                    result["evidence_hash"] = evidence_hash
                    result["display_message"] = f"First-aid guidance generated with Gemini ({self.model_name}) from verified evidence."
                    return result
                
                fallback_reason = result.get("fallback_reason", "Gemini response validation failed") if result else "Gemini call returned empty response"
                return {
                    "provider": "rule_based_fallback",
                    "status": "fallback",
                    "model": self.model_name,
                    "fallback_reason": fallback_reason,
                    "generated_at": generated_at,
                    "evidence_hash": evidence_hash,
                    "display_message": "AI-generated guidance unavailable. Showing rule-based research guidance."
                }
            except concurrent.futures.TimeoutError:
                print(f"Gemini API request timed out (> {timeout_seconds}s). Switching to rule-based fallback.")
                return {
                    "provider": "rule_based_fallback",
                    "status": "fallback",
                    "model": self.model_name,
                    "fallback_reason": f"timeout (request exceeded {timeout_seconds}s)",
                    "generated_at": generated_at,
                    "evidence_hash": evidence_hash,
                    "display_message": "AI-generated guidance unavailable (request timed out). Showing rule-based research guidance."
                }
            except Exception as e:
                print(f"Gemini API execution error: {str(e)}")
                return {
                    "provider": "rule_based_fallback",
                    "status": "fallback",
                    "model": self.model_name,
                    "fallback_reason": f"API error: {str(e)}",
                    "generated_at": generated_at,
                    "evidence_hash": evidence_hash,
                    "display_message": "AI-generated guidance unavailable. Showing rule-based research guidance."
                }

    def _call_gemini_sdk(self, structured_evidence: Dict[str, Any], api_key: str) -> Optional[Dict[str, Any]]:
        """Makes actual call using google-genai SDK."""
        client = genai.Client(api_key=api_key)

        system_instruction = (
            "You are generating conservative first-aid research guidance for an academic software prototype.\n"
            "You are NOT a doctor. You must NOT diagnose.\n"
            "You must NOT identify fractures, internal injuries, infections, or other medical conditions from an image.\n"
            "You must NOT invent symptoms, measurements, history, model findings, or questionnaire answers.\n"
            "Use ONLY the supplied StructuredEvidence object.\n"
            "Never modify or reinterpret user-provided questionnaire values.\n"
            "Do NOT claim that YOLO detected an injury unless yolo.finding_detected=true.\n"
            "YOLO11 supported classes are ONLY: cut, bruise, wound.\n"
            "Swelling is an EfficientNet research-classifier category and is NOT a YOLO detection.\n"
            "If YOLO11 says no confident detection, preserve that fact accurately.\n"
            "Do NOT override deterministic safety rules or downgrade safety warnings.\n"
            "If urgent warning signs are present in safety_rules, clearly recommend appropriate professional medical evaluation.\n"
            "Always state clearly that this is research software and not medical advice or a diagnosis.\n"
            "Return valid JSON matching the exact requested JSON schema."
        )

        prompt = f"""Given the following verified case evidence JSON:
{json.dumps(structured_evidence, indent=2)}

Generate structured first-aid guidance in valid JSON format matching this exact schema:
{{
    "immediate_first_aid_steps": ["Step 1", "Step 2"],
    "actions_to_avoid": ["Action 1"],
    "symptoms_to_monitor": ["Symptom 1"],
    "urgent_evaluation_warning": ["Warning sign 1 if present"],
    "professional_evaluation_guidance": ["Guidance note 1"],
    "limitations": ["Research prototype limitation note"]
}}"""

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            response_mime_type="application/json"
        )

        models_to_try = [self.model_name, "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-flash-latest"]
        # Remove duplicates while preserving order
        candidate_models = list(dict.fromkeys(models_to_try))

        response = None
        used_model = self.model_name
        last_err = None

        for m_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=m_name,
                    contents=prompt,
                    config=config
                )
                if response and response.text:
                    used_model = m_name
                    break
            except Exception as e:
                last_err = e
                # Fall through immediately to next active candidate on any rate-limit or API error
                continue

        if not response or not response.text:
            return {"status": "error", "fallback_reason": f"Gemini API exception: {str(last_err)}"}

        try:
            parsed = json.loads(response.text)

            required_keys = [
                "immediate_first_aid_steps", "actions_to_avoid",
                "symptoms_to_monitor", "professional_evaluation_guidance"
            ]
            if not all(k in parsed for k in required_keys):
                return {"status": "error", "fallback_reason": "Gemini response missing required JSON keys"}

            return {
                "provider": "gemini",
                "status": "success",
                "model": used_model,
                "guidance": {
                    "immediate_first_aid_steps": [str(x) for x in parsed.get("immediate_first_aid_steps", [])],
                    "actions_to_avoid": [str(x) for x in parsed.get("actions_to_avoid", [])],
                    "symptoms_to_monitor": [str(x) for x in parsed.get("symptoms_to_monitor", [])],
                    "urgent_evaluation_warning": [str(x) for x in parsed.get("urgent_evaluation_warning", [])],
                    "professional_evaluation_guidance": [str(x) for x in parsed.get("professional_evaluation_guidance", [])],
                    "limitations": [str(x) for x in parsed.get("limitations", [
                        "AI-QTriage is an academic research prototype and does not provide clinical medical diagnosis."
                    ])]
                }
            }
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as parse_err:
            return {"status": "error", "fallback_reason": f"Gemini response JSON parse error: {str(parse_err)}"}


gemini_service = GeminiFirstAidService()
