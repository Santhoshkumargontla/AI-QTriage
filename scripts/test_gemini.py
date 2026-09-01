"""
test_gemini.py

Environment check & real API integration test for Google GenAI SDK (google-genai).
Tests:
1. Environment configuration loading (backend/.env)
2. SDK import (from google import genai)
3. Client initialization & model configuration
4. Real Gemini API request (if GEMINI_API_KEY is configured)
5. Structured JSON response parsing & schema validation
6. Never prints or exposes the actual API key.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load backend/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", ".env")
load_dotenv(env_path)

from backend.services.gemini_first_aid_service import get_gemini_config, GeminiFirstAidService, compute_evidence_hash


def test_gemini_environment():
    print("=================================================================")
    print("AI-QTriage — GOOGLE GEMINI SDK & ENVIRONMENT TEST")
    print("=================================================================")

    config = get_gemini_config()
    print(f"GEMINI CONFIGURABLE STATUS : {'YES' if config['configured'] else 'NO'}")
    print(f"GEMINI SDK INSTALLED       : {'OK (google-genai)' if config['sdk_available'] else 'FAIL'}")
    print(f"TARGET GEMINI MODEL        : {config['model']}")
    print(f"FALLBACK ENGINE            : {config['fallback']}")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print("\n-----------------------------------------------------------------")
        print("GEMINI NOT VERIFIED — GEMINI_API_KEY NOT CONFIGURED IN backend/.env")
        print("-----------------------------------------------------------------")
        print("Setup Instructions:")
        print("1. Create/edit file: backend/.env")
        print("2. Add line: GEMINI_API_KEY=your_real_gemini_api_key")
        print("3. Re-run: python scripts/test_gemini.py")
        print("=================================================================")
        return False

    print(f"API KEY DETECTED           : YES (Length: {len(api_key)} chars, Key hidden for security)")

    # Execute real test request with canonical StructuredEvidence object
    print("\n[EXECUTING REAL GEMINI API TEST REQUEST]")
    test_evidence = {
        "questionnaire": {
            "location": "Left ankle",
            "pain_level": 7,
            "cause": "sports",
            "swelling": "yes",
            "open_wound": None,
            "bleeding": None
        },
        "yolo": {
            "finding": None,
            "finding_detected": False,
            "confidence": None,
            "bounding_box": None,
            "supported_classes": ["cut", "bruise", "abrasion", "laceration"]
        },
        "research_classifier": {
            "finding": "Swelling",
            "confidence": 0.88,
            "category_type": "research_classifier"
        },
        "sensor": {
            "provided": False,
            "data": None
        },
        "experimental_models": {
            "xgboost": {"class": "MODERATE"},
            "vqc": {"class": "MODERATE"},
            "agreement": "AGREEMENT",
            "uncertainty": "LOW"
        },
        "safety_rules": {
            "guidance_level": "MODERATE",
            "warning_signs": ["User-reported pain level: 7/10"]
        }
    }

    service = GeminiFirstAidService()
    res = service.generate_guidance(test_evidence, timeout_seconds=10.0)

    print(f"\n[RESPONSE METADATA]")
    print(f"  - Provider      : {res.get('provider')}")
    print(f"  - Status        : {res.get('status')}")
    print(f"  - Model         : {res.get('model')}")
    print(f"  - Fallback Reason: {res.get('fallback_reason', 'None (Live Gemini Call Succeeded)')}")
    print(f"  - Evidence Hash : {res.get('evidence_hash')}")

    if res.get("provider") == "gemini" and res.get("status") == "success":
        guidance = res.get("guidance", {})
        print(f"\n[STRUCTURED OUTPUT VALIDATION: OK]")
        print(f"  - Immediate Steps count : {len(guidance.get('immediate_first_aid_steps', []))}")
        print(f"  - Actions to Avoid count: {len(guidance.get('actions_to_avoid', []))}")
        print(f"  - Symptoms to Monitor   : {len(guidance.get('symptoms_to_monitor', []))}")
        print(f"  - Limitations           : {guidance.get('limitations')}")

        print("\n=================================================================")
        print("GEMINI VERIFIED END-TO-END [OK]")
        print("=================================================================")
        return True
    else:
        print(f"\n-----------------------------------------------------------------")
        print(f"GEMINI NOT VERIFIED — FALLBACK ACTIVE ({res.get('fallback_reason')})")
        print("-----------------------------------------------------------------")
        print("=================================================================")
        return False


if __name__ == "__main__":
    test_gemini_environment()
