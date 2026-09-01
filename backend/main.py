import os
import sys
import json
import uuid
import shutil
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Load environment variables from backend/.env
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Set search path to include project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database.connection import get_database, init_db_indexes
from backend.config import settings
from pymongo.errors import PyMongoError

# Lazy-loaded vision wrappers — avoid reloading multi-GB weights on every /analyze call.
_yolo_detector = None
_unet_segmenter = None
_effnet_classifier = None


def _get_yolo_detector():
    global _yolo_detector
    if _yolo_detector is None:
        from ml.vision.yolo_wrapper import YOLO11Detector
        print("Loading YOLO11 detector (first use)...", flush=True)
        _yolo_detector = YOLO11Detector()
    return _yolo_detector


def _get_unet_segmenter():
    global _unet_segmenter
    if _unet_segmenter is None:
        from ml.vision.unet_wrapper import UNetSegmenter
        print("Loading U-Net segmenter (first use)...", flush=True)
        _unet_segmenter = UNetSegmenter()
    return _unet_segmenter


def _get_effnet_classifier():
    global _effnet_classifier
    if _effnet_classifier is None:
        from ml.vision.efficientnet_wrapper import EfficientNetV2Classifier
        print("Loading EfficientNet classifier (first use)...", flush=True)
        _effnet_classifier = EfficientNetV2Classifier()
    return _effnet_classifier

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting AI-QTriage backend service...")
    try:
        db = get_database()
        init_db_indexes()
        print("AI-QTriage successfully connected to MongoDB.")
    except (PyMongoError, RuntimeError, OSError) as e:
        print(f"CRITICAL: Application failed to start due to database error: {str(e)}", file=sys.stderr)
        os._exit(1)
    yield
    print("Shutting down AI-QTriage backend service...")

app = FastAPI(
    title="AI-QTriage API",
    description="Backend API for the AI-QTriage Research Prototype.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for Next.js frontend communication
raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
allow_creds = "*" not in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["http://localhost:3000"],
    allow_credentials=allow_creds,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Upload storage path
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ── Model Registry: tracks which modality configurations have a trained model ──
# Each entry declares the modalities that model was trained on.
# Only models listed here will be used for prediction.
# Do NOT add a model here unless it has actually been trained on those modalities.
MODEL_REGISTRY = [
    {
        "model_id": "xgb_full_v1",
        "model_version": "1.0",
        "modality_configuration": ["image", "questionnaire", "sensor"],
        "path": "ml/models/xgboost_best.json",
        "type": "xgboost"
    },
    {
        "model_id": "xgb_image_questionnaire_v1",
        "model_version": "1.0",
        "modality_configuration": ["image", "questionnaire"],
        "path": "ml/models/xgboost_best.json",  # Same model, sensor features zeroed by fusion layer
        "type": "xgboost",
        "reduced_modality": True,
        "reduced_modality_note": "Sensor features are absent; fusion layer masks sensor dimensions."
    },
]

# ── Questionnaire routing threshold (configurable) ──
# This is a PROTOTYPE routing threshold used only to select questionnaire templates.
# It is NOT a medical confidence threshold.
QUESTIONNAIRE_ROUTING_THRESHOLD = float(os.environ.get("QUESTIONNAIRE_ROUTING_THRESHOLD", "0.40"))

# Pydantic Schemas
class CaseCreateSchema(BaseModel):
    user_id: Optional[str] = None
    notes: Optional[str] = None

class QuestionnaireSchema(BaseModel):
    answers: Dict[str, Any]
    voice_used: Optional[bool] = False
    voice_transcript: Optional[str] = None
    template_id: Optional[str] = None
    template_version: Optional[str] = None
    answer_source: Optional[str] = "typed"  # "typed" | "voice"

class SOSDemoSchema(BaseModel):
    case_id: str
    sensor_intensity: float
    user_acknowledged: bool

class SOSDemoTriggerSchema(BaseModel):
    mode: Optional[str] = "local_demo"  # "local_demo" | "twilio_test"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy_m: Optional[float] = None
    location_label: Optional[str] = None
    maps_url: Optional[str] = None

class SOSDemoRespondSchema(BaseModel):
    user_response: str  # "safe" | "no_response"
    mode: Optional[str] = "local_demo"  # "local_demo" | "twilio_test"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy_m: Optional[float] = None
    location_label: Optional[str] = None
    maps_url: Optional[str] = None


def _normalize_sos_geo(
    *,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    accuracy_m: Optional[float] = None,
    location_label: Optional[str] = None,
    maps_url: Optional[str] = None,
    case: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge request GPS with any previously stored case GPS; fall back to injury site text."""
    case = case or {}
    stored = case.get("sos_user_location") or {}
    lat = latitude if latitude is not None else stored.get("latitude")
    lng = longitude if longitude is not None else stored.get("longitude")
    acc = accuracy_m if accuracy_m is not None else stored.get("accuracy_m")
    label = (location_label or stored.get("location_label") or "").strip() or None
    url = (maps_url or stored.get("maps_url") or "").strip() or None
    try:
        if lat is not None and lng is not None:
            lat_f = float(lat)
            lng_f = float(lng)
            if -90.0 <= lat_f <= 90.0 and -180.0 <= lng_f <= 180.0:
                if not url:
                    url = f"https://maps.google.com/?q={lat_f:.5f},{lng_f:.5f}"
                return {
                    "latitude": lat_f,
                    "longitude": lng_f,
                    "accuracy_m": float(acc) if acc is not None else None,
                    "location_label": label,
                    "maps_url": url,
                    "source": "browser_geolocation",
                }
    except (TypeError, ValueError):
        pass
    q_loc = ((case.get("questionnaire") or {}).get("answers") or {}).get("location")
    fallback = label or (str(q_loc).strip() if q_loc else None) or "Unspecified"
    return {
        "latitude": None,
        "longitude": None,
        "accuracy_m": None,
        "location_label": fallback,
        "maps_url": None,
        "source": "questionnaire_or_unspecified",
    }


class CaseResponseSchema(BaseModel):
    case_id: str
    created_at: datetime
    status: str
    image_reference: Optional[str] = None
    visible_injury: Optional[Dict[str, Any]] = None
    questionnaire: Optional[Dict[str, Any]] = None
    sensor_summary: Optional[Dict[str, Any]] = None
    sensor_available: Optional[bool] = False
    sensor_source_type: Optional[str] = "not_provided"
    xgboost_prediction: Optional[Dict[str, Any]] = None
    quantum_prediction: Optional[Dict[str, Any]] = None
    agreement_score: Optional[str] = None
    uncertainty_status: Optional[str] = None
    uncertainty_reasons: Optional[List[str]] = None
    safety_information: Optional[List[str]] = None
    rule_derived_category: Optional[str] = None
    safety_guidance_level: Optional[str] = None
    sos_status: Optional[str] = None
    sos_delivery_mode: Optional[str] = None
    sos_delivery_outcome: Optional[str] = None
    sos_twilio_sid: Optional[str] = None
    sos_delivery_status: Optional[str] = None
    sos_provider_status: Optional[str] = None
    sos_send_timestamp: Optional[str] = None
    sos_failure_reason: Optional[str] = None
    sos_twilio_error: Optional[str] = None
    sos_twilio_error_code: Optional[Any] = None
    sos_user_location: Optional[Dict[str, Any]] = None
    first_aid_guidance: Optional[Dict[str, Any]] = None
    report_reference: Optional[str] = None
    is_demo: Optional[bool] = False
    modalities_used: Optional[List[str]] = None
    model_configuration_used: Optional[str] = None
    fusion_label_source: Optional[str] = None
    clinical_claim_blocked: Optional[bool] = None
    clinical_claim: Optional[str] = None
    paired_clinical_samples: Optional[int] = None
    consistency_analysis: Optional[Dict[str, Any]] = None
    counterfactual_analysis: Optional[Dict[str, Any]] = None
    shap_explanations: Optional[Any] = None
    prediction_agreement: Optional[str] = None
    uncertainty_level: Optional[str] = None
    justification: Optional[Any] = None

@app.get("/api/health", tags=["System"])
def health_check():
    """Verify backend health and MongoDB connectivity."""
    try:
        db = get_database()
        db.command("ping")
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except (PyMongoError, RuntimeError, OSError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MongoDB is unavailable. Please verify the configured MongoDB connection. Error: {str(e)}"
        )

@app.get("/api/ai/gemini/status", tags=["AI Status"])
def get_gemini_status():
    """Return Gemini configuration and availability status without exposing API keys."""
    from backend.services.gemini_first_aid_service import get_gemini_config
    return get_gemini_config()

@app.post("/api/cases", response_model=CaseResponseSchema, status_code=status.HTTP_201_CREATED, tags=["Cases"])
def create_case(payload: CaseCreateSchema):
    """Create a new research assessment case."""
    db = get_database()
    case_id = str(uuid.uuid4())
    case_doc = {
        "case_id": case_id,
        "created_at": datetime.utcnow(),
        "status": "created",
        "image_reference": None,
        "visible_injury": None,
        "questionnaire": None,
        "sensor_summary": None,
        "sensor_available": False,
        "sensor_source_type": "not_provided",
        "xgboost_prediction": None,
        "quantum_prediction": None,
        "agreement_score": None,
        "uncertainty_status": None,
        "safety_information": [],
        "sos_status": "not_triggered",
        "report_reference": None,
        "modalities_used": [],
        "model_configuration_used": None
    }
    db.cases.insert_one(case_doc)
    return case_doc

@app.get("/api/cases/{case_id}", response_model=CaseResponseSchema, tags=["Cases"])
def get_case(case_id: str):
    """Retrieve details for a specific case."""
    db = get_database()
    case = db.cases.find_one({"case_id": case_id})
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case with ID {case_id} was not found."
        )
    if case.get("visible_injury"):
        vis = dict(case["visible_injury"])
        vis["classifier_model_status"] = _effnet_training_status()
        vis["segmentation_model_status"] = _unet_training_status()
        case["visible_injury"] = vis
    return case

@app.get("/api/cases", response_model=List[CaseResponseSchema], tags=["Cases"])
def list_cases(limit: int = 20):
    """List recent research cases."""
    db = get_database()
    cursor = db.cases.find().sort("created_at", -1).limit(limit)
    return list(cursor)

@app.post("/api/cases/{case_id}/image", tags=["Cases"])
async def upload_image(case_id: str, file: UploadFile = File(...)):
    """Upload injury image and verify basic quality."""
    from ml.vision.preprocess import verify_image_quality, ImageQualityError
    
    db = get_database()
    case = db.cases.find_one({"case_id": case_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # 1. MIME and Extension Validation
    ext = os.path.splitext(file.filename or "")[1].lower()
    allowed_exts = {".jpg", ".jpeg", ".png"}
    if ext not in allowed_exts or file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Please upload a valid JPG, JPEG, or PNG image."
        )

    # 2. File Size Limit (10 MB max)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum limit of 10 MB."
        )
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty."
        )

    # 3. Magic-byte inspection via PIL
    try:
        from PIL import Image, UnidentifiedImageError
        import io
        img = Image.open(io.BytesIO(content))
        img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as img_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupt image content: {str(img_err)}"
        )

    # Save image file locally using sanitized case_id filename
    filename = f"{case_id}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    import hashlib
    image_sha256 = hashlib.sha256(content).hexdigest()

    try:
        with open(filepath, "wb") as buffer:
            buffer.write(content)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image file: {str(e)}")

    # Check image quality using OpenCV/Pillow heuristics
    try:
        quality_metrics = verify_image_quality(filepath)
    except ImageQualityError as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except (OSError, ValueError, RuntimeError) as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during quality analysis: {str(e)}"
        )

    # Update case reference in MongoDB with image metadata and metrics
    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "image_reference": filepath,
            "image_sha256": image_sha256,
            "image_quality_metrics": quality_metrics,
            "status": "image_uploaded"
        }}
    )

    return {
        "message": "Image uploaded and verified successfully",
        "path": filepath,
        "image_sha256": image_sha256,
        "quality_metrics": quality_metrics
    }

@app.post("/api/cases/{case_id}/questionnaire", tags=["Cases"])
def submit_questionnaire(case_id: str, payload: QuestionnaireSchema):
    """Submit structured case questionnaire answers cleanly (unanswered fields saved as null)."""
    db = get_database()
    case = db.cases.find_one({"case_id": case_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    q_doc = payload.dict()
    raw_answers = q_doc.get("answers") or {}
    canonical_answers = {}
    
    for k, v in raw_answers.items():
        if v is None or str(v).strip() == "" or str(v).lower() == "not_provided":
            canonical_answers[k] = None
        elif k in ("pain_level", "pain"):
            try:
                canonical_answers["pain_level"] = int(v)
            except (ValueError, TypeError):
                canonical_answers["pain_level"] = None
        else:
            canonical_answers[k] = v

    q_doc["answers"] = canonical_answers
    q_doc.setdefault("answer_source", "typed")
    q_doc.setdefault("template_id", None)
    q_doc.setdefault("template_version", None)

    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "questionnaire": q_doc,
            "status": "questionnaire_submitted"
        }}
    )
    return {"message": "Questionnaire submitted successfully"}

@app.get("/api/cases/{case_id}/questionnaire/template", tags=["Cases"])
def get_questionnaire_template(case_id: str):
    """
    Route to the appropriate questionnaire template based on the image analysis result.
    Uses deterministic class->template mapping with optional Gemini enhancement.
    """
    from backend.services.questionnaire_router import route_questionnaire, QUESTIONNAIRE_ROUTING_THRESHOLD

    db = get_database()
    case = db.cases.find_one({"case_id": case_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    visible_injury = case.get("visible_injury") or {}
    finding = _routing_finding(visible_injury)
    confidence = _routing_confidence(visible_injury)
    classification = visible_injury.get("classification") or {}

    if not finding:
        return {
            "routed": False,
            "template_id": None,
            "template": None,
            "router_used": "none",
            "is_uncertain": False,
            "reason": "No visible finding detected yet. Please run image analysis first.",
            "routing_threshold": QUESTIONNAIRE_ROUTING_THRESHOLD
        }

    result = route_questionnaire(finding=finding, confidence=confidence, classification=classification)
    result["routing_threshold"] = QUESTIONNAIRE_ROUTING_THRESHOLD
    return result

@app.post("/api/cases/{case_id}/sensor/skip", tags=["Cases"])
def skip_sensor(case_id: str):
    """
    Explicitly marks the case as proceeding without sensor data.
    No sensor values are fabricated. Analysis will use a reduced-modality configuration.
    """
    db = get_database()
    case = db.cases.find_one({"case_id": case_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "sensor_available": False,
            "sensor_source_type": "not_provided",
            "sensor_summary": None,
            "status": "sensor_skipped"
        }}
    )
    return {
        "message": "Sensor data not provided. Analysis will use the available modalities.",
        "sensor_available": False,
        "sensor_source_type": "not_provided"
    }

@app.post("/api/cases/{case_id}/sensor", tags=["Cases"])
async def upload_sensor(case_id: str, file: UploadFile = File(...)):
    """Upload smartphone sensor log file, validate headers/sampling, and reconstruct timeline."""
    from ml.sensor.sensor_processor import process_sensor_data, SensorValidationError
    
    db = get_database()
    case = db.cases.find_one({"case_id": case_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Save sensor file locally
    sensor_filename = f"{case_id}_sensor.csv"
    sensor_path = os.path.join(UPLOAD_DIR, sensor_filename)
    
    try:
        with open(sensor_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save sensor log: {str(e)}")

    # Process and validate
    try:
        summary = process_sensor_data(sensor_path)
    except SensorValidationError as e:
        if os.path.exists(sensor_path):
            os.remove(sensor_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        if os.path.exists(sensor_path):
            os.remove(sensor_path)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing sensor logs: {str(e)}"
        )

    summary["case_id"] = case_id
    summary["source_type"] = "uploaded"
    summary["data_provenance"] = "user_provided"
    summary["scenario"] = "user_provided"
    summary["file_reference"] = sensor_path
    summary["feature_version"] = "sensor-v1"
    summary["created_at"] = datetime.utcnow().isoformat()
    summary["data_provenance_message"] = "Metrics are calculated from the supplied sensor vectors."

    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "sensor_summary": summary,
            "sensor_available": True,
            "sensor_source_type": "uploaded",
            "status": "sensor_submitted"
        }}
    )

    # Evaluate for severe kinetics triggering independent emergency warning countdown
    from backend.services.sos_service import SOSCountdownService
    sos_service = SOSCountdownService()
    sos_res = sos_service.check_and_trigger(
        case_id, 
        summary.get("peak_g_force"),
        summary.get("post_impact_stabilization_seconds"),
    )

    return {
        "message": "Sensor data uploaded and processed successfully", 
        "summary": summary,
        "sos_triggered": sos_res["sos_triggered"],
        "sos_reason": sos_res["reason"]
    }

class SimulateSensorSchema(BaseModel):
    scenario: str

@app.post("/api/cases/{case_id}/sensor/demo", tags=["Cases"])
def load_demo_sensor(case_id: str):
    """Load bundled football fall demo sensor log and process."""
    db = get_database()
    case = db.cases.find_one({"case_id": case_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    demo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample", "sensor", "football_fall.csv")
    if not os.path.exists(demo_path):
        raise HTTPException(status_code=404, detail="Bundled demo sensor log missing.")

    # Copy to case upload folder to simulate actual upload
    sensor_filename = f"{case_id}_sensor_demo.csv"
    dest_path = os.path.join(UPLOAD_DIR, sensor_filename)
    shutil.copyfile(demo_path, dest_path)

    from ml.sensor.sensor_processor import process_sensor_data, SensorValidationError
    try:
        summary = process_sensor_data(dest_path)
        summary["case_id"] = case_id
        summary["source_type"] = "demo"
        summary["data_provenance"] = "synthetic"
        summary["scenario"] = "football_fall"
        summary["file_reference"] = dest_path
        summary["feature_version"] = "sensor-v1"
        summary["created_at"] = datetime.utcnow().isoformat()
        summary["data_provenance_message"] = "Synthetic sensor vectors from the bundled demonstration dataset are processed using the same sensor pipeline."
    except SensorValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"Failed processing demo log: {str(e)}")

    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "sensor_summary": summary,
            "sensor_available": True,
            "sensor_source_type": "demo",
            "status": "sensor_submitted"
        }}
    )

    from backend.services.sos_service import SOSCountdownService
    sos_service = SOSCountdownService()
    sos_res = sos_service.check_and_trigger(
        case_id, 
        summary.get("peak_g_force"),
        summary.get("post_impact_stabilization_seconds"),
    )

    return {
        "message": "Demo sensor log loaded and processed successfully",
        "summary": summary,
        "sos_triggered": sos_res["sos_triggered"],
        "sos_reason": sos_res["reason"]
    }

@app.post("/api/cases/{case_id}/sensor/simulate", tags=["Cases"])
def simulate_sensor(case_id: str, payload: SimulateSensorSchema):
    """Generate synthetic sensor signals based on scenario and run pipeline."""
    db = get_database()
    case = db.cases.find_one({"case_id": case_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    from ml.sensor.scenarios import SUPPORTED_SCENARIOS, generate_scenario_dataframe, resolve_scenario

    scenario = resolve_scenario(payload.scenario)
    if scenario is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid scenario type. Use: " + ", ".join(SUPPORTED_SCENARIOS),
        )

    df_sim = generate_scenario_dataframe(scenario)
    
    sim_filename = f"{case_id}_sensor_sim.csv"
    sim_path = os.path.join(UPLOAD_DIR, sim_filename)
    df_sim.to_csv(sim_path, index=False)

    from ml.sensor.sensor_processor import process_sensor_data, SensorValidationError
    try:
        summary = process_sensor_data(sim_path)
        summary["case_id"] = case_id
        summary["source_type"] = "simulated"
        summary["data_provenance"] = "synthetic"
        summary["scenario"] = scenario
        summary["file_reference"] = sim_path
        summary["feature_version"] = "sensor-v1"
        summary["created_at"] = datetime.utcnow().isoformat()
        summary["data_provenance_message"] = "Synthetic sensor vectors are generated according to the selected scenario and then passed through the same preprocessing and feature-extraction pipeline."
    except SensorValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"Failed processing simulation log: {str(e)}")

    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "sensor_summary": summary,
            "sensor_available": True,
            "sensor_source_type": "simulated",
            "status": "sensor_submitted"
        }}
    )

    from backend.services.sos_service import SOSCountdownService
    sos_service = SOSCountdownService()
    sos_res = sos_service.check_and_trigger(
        case_id, 
        summary.get("peak_g_force"),
        summary.get("post_impact_stabilization_seconds"),
    )

    return {
        "message": "Sensor simulation generated and processed successfully",
        "summary": summary,
        "sos_triggered": sos_res["sos_triggered"],
        "sos_reason": sos_res["reason"]
    }

class LiveSensorPayloadInput(BaseModel):
    source_type: Optional[str] = "live"
    device_metadata: Optional[Dict[str, Any]] = None
    recording_duration_seconds: Optional[float] = None
    observed_sampling_rate_hz: Optional[float] = None
    samples: List[Dict[str, Any]]

@app.post("/api/cases/{case_id}/sensor/live/upload", tags=["Cases"])
def upload_live_sensor(case_id: str, payload: LiveSensorPayloadInput):
    """
    Accepts real-time mobile sensor data batch upload, validates raw samples server-side,
    calculates authoritative backend sampling rate, and passes data through existing sensor pipeline.
    """
    db = get_database()
    case = db.cases.find_one({"case_id": case_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    from backend.services.live_sensor_adapter import (
        validate_raw_live_samples, 
        adapt_live_samples_to_df, 
        LiveSensorValidationError
    )

    try:
        actual_duration_sec, backend_verified_rate, sanitized_samples = validate_raw_live_samples(payload.samples)
    except LiveSensorValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except (OSError, ValueError, TypeError, KeyError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed for live sensor payload: {str(e)}"
        )

    # Determine sensor availability flags
    has_accel = any(
        s.get("acceleration_x") is not None or s.get("acceleration_gravity_x") is not None 
        for s in sanitized_samples
    )
    has_gyro = any(
        s.get("rotation_alpha") is not None or s.get("rotation_rate_alpha") is not None 
        for s in sanitized_samples
    )
    has_gps = any(
        s.get("latitude") is not None
        for s in sanitized_samples
    )

    try:
        df_live = adapt_live_samples_to_df(sanitized_samples, backend_verified_rate)
    except LiveSensorValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

    live_filename = f"{case_id}_sensor_live.csv"
    live_path = os.path.join(UPLOAD_DIR, live_filename)
    df_live.to_csv(live_path, index=False)

    from ml.sensor.sensor_processor import process_sensor_data, SensorValidationError
    try:
        summary = process_sensor_data(live_path)
        summary["case_id"] = case_id
        summary["source_type"] = "live"
        summary["data_provenance"] = "real_time_device_capture"
        summary["data_provenance_badge"] = "REAL-TIME DEVICE DATA"
        summary["scenario"] = "live_device_recording"
        summary["file_reference"] = live_path
        summary["raw_recording_reference"] = live_path
        summary["feature_version"] = "sensor-v1"
        summary["created_at"] = datetime.utcnow().isoformat()
        
        # Authoritative backend calculation vs frontend estimate
        summary["recording_duration_seconds"] = round(actual_duration_sec, 2)
        summary["sample_count"] = len(sanitized_samples)
        summary["frontend_observed_sampling_rate_hz"] = payload.observed_sampling_rate_hz
        summary["backend_verified_sampling_rate_hz"] = backend_verified_rate
        summary["acceleration_unit"] = "m/s²"
        summary["timestamp_unit"] = "milliseconds"
        
        summary["sensor_availability"] = {
            "accelerometer": has_accel,
            "gyroscope": has_gyro,
            "location": has_gps
        }
        summary["device_metadata"] = payload.device_metadata or {}
        summary["data_provenance_message"] = "Metrics are calculated from real-time physical sensor samples recorded directly from the user's mobile device."
    except SensorValidationError as e:
        if os.path.exists(live_path):
            os.remove(live_path)
        raise HTTPException(status_code=422, detail=str(e))
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        if os.path.exists(live_path):
            os.remove(live_path)
        raise HTTPException(status_code=500, detail=f"Failed processing live sensor data: {str(e)}")

    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "sensor_summary": summary,
            "sensor_available": True,
            "sensor_source_type": "live",
            "status": "sensor_submitted"
        }}
    )

    from backend.services.sos_service import SOSCountdownService
    sos_service = SOSCountdownService()
    sos_res = sos_service.check_and_trigger(
        case_id, 
        summary.get("peak_g_force"),
        summary.get("post_impact_stabilization_seconds"),
    )

    return {
        "message": "Real-time device sensor data uploaded and processed successfully",
        "summary": summary,
        "sos_triggered": sos_res["sos_triggered"],
        "sos_reason": sos_res["reason"]
    }

@app.post("/api/cases/demo", response_model=CaseResponseSchema, status_code=status.HTTP_201_CREATED, tags=["Cases"])
def create_and_analyze_demo_case():
    """Create a complete end-to-end demo case using sample files, and run full analysis in one click."""
    db = get_database()
    case_id = str(uuid.uuid4())
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    sample_img_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample", "image", "football_injury.jpg")
    if not os.path.exists(sample_img_src):
        from scripts.create_sample_image import main as gen_img
        gen_img()
        
    img_extension = ".jpg"
    img_dest = os.path.join(UPLOAD_DIR, f"{case_id}{img_extension}")
    shutil.copyfile(sample_img_src, img_dest)
    
    from ml.vision.preprocess import verify_image_quality
    quality_metrics = verify_image_quality(img_dest)
    
    sample_sensor_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample", "sensor", "football_fall.csv")
    sensor_dest = os.path.join(UPLOAD_DIR, f"{case_id}_sensor_demo.csv")
    shutil.copyfile(sample_sensor_src, sensor_dest)
    
    from ml.sensor.sensor_processor import process_sensor_data
    sensor_summary = process_sensor_data(sensor_dest)
    sensor_summary["case_id"] = case_id
    sensor_summary["source_type"] = "demo"
    sensor_summary["data_provenance"] = "synthetic"
    sensor_summary["scenario"] = "football_fall"
    sensor_summary["file_reference"] = sensor_dest
    sensor_summary["feature_version"] = "sensor-v1"
    sensor_summary["created_at"] = datetime.utcnow().isoformat()
    sensor_summary["data_provenance_message"] = "Synthetic sensor vectors from the bundled demonstration dataset are processed using the same sensor pipeline."
    
    questionnaire = {
        "answers": {
            "pain_level": 7,
            "injury_mechanism": "sports",
            "movement_limitation": "moderate",
            "weight_bearing": "no",
            "visible_bleeding": "no",
            "crack_pop": "yes",
            "direct_impact": "yes"
        },
        "voice_used": False,
        "voice_transcript": None
    }
    
    case_doc = {
        "case_id": case_id,
        "created_at": datetime.utcnow(),
        "status": "sensor_submitted",
        "is_demo": True,
        "image_reference": img_dest,
        "image_quality_metrics": quality_metrics,
        "visible_injury": None,
        "questionnaire": questionnaire,
        "sensor_summary": sensor_summary,
        "sensor_available": True,
        "sensor_source_type": "demo",
        "xgboost_prediction": None,
        "quantum_prediction": None,
        "agreement_score": None,
        "uncertainty_status": None,
        "safety_information": [],
        "sos_status": "NONE",
        "report_reference": None
    }
    db.cases.insert_one(case_doc)
    
    analyze_case(case_id)
    
    analyzed_case = db.cases.find_one({"case_id": case_id})
    return analyzed_case

@app.post("/api/cases/{case_id}/sos/abort", tags=["Emergency Demo"])
def abort_sos_alert(case_id: str):
    """Abort the active SOS alert countdown."""
    from backend.services.sos_service import SOSCountdownService
    service = SOSCountdownService()
    try:
        result = service.abort_sos(case_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/cases/{case_id}/sos/status", tags=["Emergency Demo"])
def get_sos_alert_status(case_id: str):
    """Retrieve the remaining seconds or status of the SOS countdown."""
    from backend.services.sos_service import SOSCountdownService
    service = SOSCountdownService()
    try:
        result = service.get_sos_status(case_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/sos/config", tags=["Emergency Demo"])
def get_sos_config():
    """Returns safe Twilio configuration status for research UI."""
    from backend.services.twilio_service import twilio_service
    return twilio_service.get_status_info()

@app.post("/api/cases/{case_id}/sos/demo/trigger", tags=["Emergency Demo"])
def trigger_demo_sos(case_id: str, payload: Optional[SOSDemoTriggerSchema] = None):
    """
    Triggers a 10-second countdown for the DEMO SOS system.
    Supports mode: 'local_demo' or 'twilio_test'.
    NO REAL EMERGENCY SERVICES ARE CONTACTED.
    """
    db = get_database()
    case = db.cases.find_one({"case_id": case_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    mode = (payload.mode if payload and payload.mode else "local_demo").lower()
    geo = _normalize_sos_geo(
        latitude=payload.latitude if payload else None,
        longitude=payload.longitude if payload else None,
        accuracy_m=payload.accuracy_m if payload else None,
        location_label=payload.location_label if payload else None,
        maps_url=payload.maps_url if payload else None,
        case=case,
    )
    if mode == "twilio_test" and geo.get("latitude") is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Twilio SOS requires the user's GPS location (latitude/longitude). "
                "Allow browser location access or enter coordinates, then retry."
            ),
        )

    now = datetime.utcnow()

    # STALE EVENT CLEANUP: Mark any countdown events that are older than (countdown_seconds + 30s grace)
    # as "expired_stale" so they don't block creation of a genuinely new SOS test.
    stale_events = list(db.sos_events.find({"case_id": case_id, "sos_status": "countdown"}))
    for stale_ev in stale_events:
        try:
            created_str = stale_ev.get("created_at", "")
            created_at = datetime.fromisoformat(created_str) if created_str else None
            countdown_secs = float(stale_ev.get("countdown_seconds", 10))
            grace_seconds = countdown_secs + 30.0  # allow 30s extra for polling delays
            if created_at and (now - created_at).total_seconds() > grace_seconds:
                db.sos_events.update_one(
                    {"event_id": stale_ev["event_id"]},
                    {"$set": {
                        "sos_status": "expired_stale",
                        "expired_at": now.isoformat()
                    }}
                )
        except (ValueError, TypeError, OverflowError) as e:
            print(f"SOS stale-event cleanup skipped: {e}")

    # Duplicate protection: return existing ACTIVE countdown if present (within grace window)
    existing = db.sos_events.find_one(
        {"case_id": case_id, "sos_status": "countdown"},
        sort=[("created_at", -1)]
    )
    if existing:
        existing.pop("_id", None)
        return {
            "message": "An active SOS countdown already exists for this case.",
            "event": existing,
            "duplicate": True
        }

    from backend.services.twilio_service import twilio_service
    twilio_cfg, twilio_msg = twilio_service.is_configured()

    event_id = str(uuid.uuid4())
    event_doc = {
        "event_id": event_id,
        "case_id": case_id,
        "sos_status": "countdown",
        "trigger_source": "demo",
        "delivery_mode": mode,
        "twilio_enabled": twilio_service.enabled,
        "twilio_configured": twilio_cfg,
        "demo_only": True,
        "countdown_seconds": 10,
        "user_response": None,
        "user_location": geo,
        "created_at": now.isoformat(),
        "warning": "NO REAL EMERGENCY SERVICES WERE CONTACTED."
    }
    db.sos_events.insert_one(event_doc)

    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "sos_status": "countdown",
            "sos_event_id": event_id,
            "sos_trigger_source": "demo",
            "sos_delivery_mode": mode,
            "sos_trigger_time": now.isoformat(),
            "sos_countdown_seconds": 10,
            "sos_user_location": geo,
        }}
    )

    event_doc.pop("_id", None)
    return {
        "message": f"SOS countdown started ({'Twilio Test Mode' if mode == 'twilio_test' else 'Local Demo'}). Simulation only.",
        "event": event_doc,
        "user_location": geo,
        "duplicate": False
    }

@app.post("/api/cases/{case_id}/sos/demo/respond", tags=["Emergency Demo"])
def respond_demo_sos(case_id: str, payload: SOSDemoRespondSchema):
    """
    Record the user's response to the DEMO SOS countdown.
    safe → cancelled. no_response → LOCAL_SIMULATION or Twilio outcome.
    SMS_SENT is never written. A SID is required for TWILIO_REQUEST_QUEUED.
    """
    db = get_database()
    case = db.cases.find_one({"case_id": case_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    user_response = payload.user_response.lower().strip()
    if user_response not in ("safe", "no_response"):
        raise HTTPException(status_code=400, detail="user_response must be 'safe' or 'no_response'")

    mode = (payload.mode or case.get("sos_delivery_mode") or "local_demo").lower()
    from backend.services.twilio_service import twilio_service
    from backend.services.sos_delivery import (
        LOCAL_SIMULATION,
        persist_local_simulation,
        persist_twilio_result,
        utc_now_iso,
    )

    active_event = db.sos_events.find_one({"case_id": case_id, "sos_status": "countdown"})

    if user_response == "safe":
        new_status = "cancelled"
        msg = "✓ Demo SOS cancelled. You indicated you are safe."
        if active_event:
            db.sos_events.update_one(
                {"event_id": active_event["event_id"]},
                {"$set": {
                    "sos_status": "cancelled",
                    "user_response": "safe",
                    "resolved_at": datetime.utcnow().isoformat()
                }}
            )
        db.cases.update_one(
            {"case_id": case_id},
            {"$set": {"sos_status": "cancelled"}}
        )
        return {
            "sos_status": "cancelled",
            "delivery_mode": mode,
            "message": msg,
            "twilio_result": None,
            "warning": "NO REAL EMERGENCY SERVICES WERE CONTACTED."
        }
    else:
        if not active_event:
            latest_event = db.sos_events.find_one({"case_id": case_id}, sort=[("created_at", -1)])
            new_status = latest_event.get("sos_status") if latest_event else LOCAL_SIMULATION
            sid = latest_event.get("twilio_message_sid") if latest_event else None
            return {
                "sos_status": new_status,
                "delivery_outcome": new_status,
                "delivery_mode": mode,
                "message": "Demo SOS has already been triggered or processed.",
                "twilio_message_sid": sid,
                "provider_status": (latest_event or {}).get("provider_status") or (latest_event or {}).get("delivery_status"),
                "timestamp": (latest_event or {}).get("send_timestamp"),
                "failure_reason": (latest_event or {}).get("failure_reason") or (latest_event or {}).get("twilio_error_message"),
                "twilio_result": {
                    "success": bool(sid),
                    "twilio_message_sid": sid,
                    "delivery_status": (latest_event or {}).get("provider_status") or (latest_event or {}).get("delivery_status"),
                } if sid else None,
                "warning": "NO REAL EMERGENCY SERVICES WERE CONTACTED."
            }

        claimed_event = db.sos_events.find_one_and_update(
            {"case_id": case_id, "sos_status": "countdown"},
            {"$set": {
                "sos_status": "sending",
                "sending_started_at": utc_now_iso()
            }},
            return_document=True
        )

        if not claimed_event:
            latest_event = db.sos_events.find_one({"case_id": case_id}, sort=[("created_at", -1)])
            new_status = latest_event.get("sos_status") if latest_event else LOCAL_SIMULATION
            return {
                "sos_status": new_status,
                "delivery_outcome": new_status,
                "delivery_mode": mode,
                "message": "Demo SOS has already been triggered or processed.",
                "twilio_result": None,
                "warning": "NO REAL EMERGENCY SERVICES WERE CONTACTED."
            }

        db.cases.update_one(
            {"case_id": case_id},
            {"$set": {"sos_status": "sending"}}
        )

        twilio_result = None

        if mode == "twilio_test":
            vi = case.get("visible_injury") or {}
            q_data = (case.get("questionnaire") or {}).get("answers") or {}
            geo = _normalize_sos_geo(
                latitude=payload.latitude,
                longitude=payload.longitude,
                accuracy_m=payload.accuracy_m,
                location_label=payload.location_label,
                maps_url=payload.maps_url,
                case=case,
            )
            if geo.get("latitude") is None:
                db.sos_events.update_one(
                    {"event_id": claimed_event["event_id"]},
                    {"$set": {"sos_status": "countdown", "sending_started_at": None}},
                )
                db.cases.update_one(
                    {"case_id": case_id},
                    {"$set": {"sos_status": "countdown"}},
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Twilio SOS requires the user's GPS location (latitude/longitude). "
                        "Allow browser location access or enter coordinates, then retry."
                    ),
                )
            db.cases.update_one(
                {"case_id": case_id},
                {"$set": {"sos_user_location": geo}},
            )
            db.sos_events.update_one(
                {"event_id": claimed_event["event_id"]},
                {"$set": {"user_location": geo}},
            )
            body_part = q_data.get("location")
            location_fallback = geo.get("location_label") or body_part or "GPS unavailable"
            if vi.get("yolo_finding_detected") and vi.get("yolo_finding"):
                yolo_finding = vi["yolo_finding"]
                classifier_category = vi.get("classifier_finding")
            else:
                yolo_finding = None
                classifier_category = vi.get("classifier_finding") or vi.get("finding")

            twilio_res = twilio_service.send_test_sos_message(
                case_id=case_id,
                yolo_finding=yolo_finding,
                classifier_category=classifier_category,
                user_location=location_fallback,
                latitude=geo.get("latitude"),
                longitude=geo.get("longitude"),
                maps_url=geo.get("maps_url"),
                sos_event_id=claimed_event["event_id"],
                trigger_time=claimed_event.get("created_at", "")[:19]
            )
            twilio_result = twilio_res
            applied = persist_twilio_result(
                db, case_id, claimed_event["event_id"], twilio_res,
                extra_event={"user_response": "no_response", "user_location": geo},
            )
            new_status = applied["status"]
            if new_status == "TWILIO_REQUEST_QUEUED":
                msg = "Twilio API accepted the request and returned a message SID. Not proof of handset delivery."
            elif new_status == "TWILIO_NOT_CONFIGURED":
                msg = applied.get("failure_reason") or "Twilio is not configured. No SMS was sent."
            else:
                msg = f"Twilio request failed: {applied.get('failure_reason') or 'unknown error'}"
        else:
            applied = persist_local_simulation(db, case_id, claimed_event["event_id"])
            new_status = LOCAL_SIMULATION
            msg = "LOCAL_SIMULATION recorded. No SMS was sent."
            twilio_result = None

    return {
        "sos_status": new_status,
        "delivery_outcome": new_status,
        "delivery_mode": mode,
        "message": msg,
        "twilio_message_sid": (twilio_result or {}).get("twilio_message_sid") if twilio_result else None,
        "provider_status": (twilio_result or {}).get("provider_status") or (twilio_result or {}).get("delivery_status") if twilio_result else None,
        "timestamp": (twilio_result or {}).get("timestamp") if twilio_result else None,
        "failure_reason": (twilio_result or {}).get("failure_reason") if twilio_result else None,
        "twilio_result": twilio_result,
        "warning": "NO REAL EMERGENCY SERVICES WERE CONTACTED."
    }




def _routing_finding(visible_injury: dict) -> str:
    """Prefer a YOLO class when detected. Do not route questionnaires from an untrustworthy classifier."""
    vi = visible_injury or {}
    if vi.get("yolo_finding_detected") and vi.get("yolo_finding"):
        name = str(vi["yolo_finding"]).strip()
        if name.lower() in ("cut", "bruise", "abrasion"):
            return name
        return ""
    status = str(vi.get("classifier_model_status") or "")
    if "NOT_TRUSTWORTHY" in status:
        return ""
    if vi.get("classifier_is_confident") is False:
        return ""
    clf = vi.get("classifier_finding") or ""
    text = str(clf).strip()
    if not text or text.lower() in (
        "unknown", "none", "null", "normal", "other", "ood_reject", "ood-reject"
    ):
        return ""
    return text


def _routing_confidence(visible_injury: dict) -> float:
    """Parse a confidence without crashing on explicit None (YOLO miss stores confidence=None)."""
    vi = visible_injury or {}
    status = str(vi.get("classifier_model_status") or "")
    skip_classifier = (
        "NOT_TRUSTWORTHY" in status
        or vi.get("classifier_is_confident") is False
    )
    keys = (
        ("yolo_confidence", "confidence")
        if skip_classifier
        else ("yolo_confidence", "classifier_probability", "confidence")
    )
    for key in keys:
        val = vi.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def _comparison_api_payload(cmp: dict) -> dict:
    """Build the Research Results comparison payload from a live evaluation artifact."""
    if "xgboost" not in cmp and isinstance(cmp.get("comparison"), dict):
        cmp = cmp["comparison"]
    vqc = cmp.get("vqc") or {}
    xgb = cmp.get("xgboost") or {}
    sample_count = cmp.get("held_out_test_samples") or xgb.get("test_sample_count") or vqc.get("test_sample_count")
    return {
        "status": "evaluated",
        "sample_count": sample_count,
        "disclaimer": "SYNTHETIC held-out split. VQC is EXPERIMENTAL_ONLY and excluded from main decisions.",
        "quantum_simulator_notice": "PennyLane default.qubit is a classical simulator. No quantum advantage claimed.",
        "metrics_source": cmp.get("metrics_source") or "live_held_out_predictions",
        "canonical_artifact": "data/results/canonical_held_out_evaluation.json",
        "class_order": cmp.get("class_order") or ["LOW", "MODERATE", "HIGH"],
        "classical_xgb": {
            "xgb_correct": xgb.get("correct_predictions"),
            "accuracy": xgb.get("accuracy"),
            "precision": xgb.get("macro_precision"),
            "recall": xgb.get("macro_recall"),
            "macro_f1": xgb.get("macro_f1"),
            "mcc": xgb.get("mcc"),
            "ece": xgb.get("ece"),
            "brier_score": xgb.get("brier_score"),
            "confusion_matrix": xgb.get("confusion_matrix"),
        },
        "quantum_vqc": {
            "vqc_correct": vqc.get("correct_predictions"),
            "accuracy": vqc.get("accuracy"),
            "precision": vqc.get("macro_precision"),
            "recall": vqc.get("macro_recall"),
            "macro_f1": vqc.get("macro_f1"),
            "mcc": vqc.get("mcc"),
            "ece": vqc.get("ece"),
            "brier_score": vqc.get("brier_score"),
            "confusion_matrix": vqc.get("confusion_matrix"),
        },
        "selective_classification": cmp.get("selective_classification") or {
            "coverage": None,
            "accuracy_at_coverage": None,
            "reason": "Selective-classification metrics were not written to the canonical held-out artifact.",
            "status": "not_available",
        },
        "vqc_outperforms_xgboost": cmp.get("vqc_outperforms_xgboost"),
        "recommendation": cmp.get("recommendation"),
        "used_in_main_decision": False,
        "interpretation": "Same held-out synthetic split for XGBoost and VQC. VQC does not drive case decisions.",
    }


def require_model_artifacts():
    """Fail closed when production/research artifacts are missing. Never auto-train random models."""
    from ml.models.canonical_paths import YOLO_CANONICAL, EFFNET_CANONICAL, UNET_CANONICAL, XGB_CANONICAL, exists
    missing = []
    for rel in (XGB_CANONICAL, YOLO_CANONICAL, EFFNET_CANONICAL, UNET_CANONICAL):
        if not exists(rel):
            missing.append(rel)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MODEL_ARTIFACT_MISSING: " + ", ".join(missing)
        )


def _classifier_yolo_coverage(parsed: dict, yolo_names) -> str:
    """Semantic overlap between classifier winner and YOLO model.names. Never 'AVAILABLE' just because YOLO fired."""
    names = {str(n).lower() for n in (yolo_names or [])}
    winner = (parsed.get("winner") or "")
    if not parsed.get("is_confident") or not winner:
        return "NOT APPLICABLE — classifier output withheld"
    if winner.lower() in names:
        return "AVAILABLE"
    return "NOT AVAILABLE"


def _effnet_training_status() -> str:
    from ml.models.canonical_paths import EFFNET_METADATA
    return _model_meta_status(EFFNET_METADATA, "UNKNOWN")


def _unet_training_status() -> str:
    from ml.models.canonical_paths import UNET_METADATA
    return _model_meta_status(UNET_METADATA, "UNKNOWN")


def _compose_full_image_mask(mask, orig_h: int, orig_w: int, bbox=None):
    """Paste an ROI mask into a full-image canvas. Never stretch an ROI mask over the original photo."""
    import cv2
    import numpy as np
    full = np.zeros((int(orig_h), int(orig_w)), dtype=np.uint8)
    if mask is None:
        return full
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = (m > 0).astype(np.uint8)
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(int(orig_w), x2)
        y2 = min(int(orig_h), y2)
        if x2 > x1 and y2 > y1:
            rh, rw = y2 - y1, x2 - x1
            if m.shape[0] != rh or m.shape[1] != rw:
                m = cv2.resize(m, (rw, rh), interpolation=cv2.INTER_NEAREST)
            full[y1:y2, x1:x2] = m
            return full
    if m.shape[0] == int(orig_h) and m.shape[1] == int(orig_w):
        return m
    return full

def _gradcam_payload(classifier, image_rgb, parsed, case_id: str, orig_w: int, orig_h: int) -> dict:
    """Generate Grad-CAM only for VALID classifier output. Never writes a misleading overlay."""
    from ml.explainability.grad_cam import maybe_generate_gradcam
    overlay, meta = maybe_generate_gradcam(classifier, image_rgb, parsed)
    overlay_path = None
    overlay_url = None
    overlay_filename = f"{case_id}_overlay.jpg"
    overlay_disk = os.path.join(UPLOAD_DIR, overlay_filename)
    if meta.get("overlay_generated") and overlay is not None:
        import cv2
        overlay_path = overlay_disk
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlay_url = f"/uploads/{overlay_filename}"
    elif os.path.exists(overlay_disk):
        # Avoid stale Grad-CAM files looking trustworthy after a withhold.
        try:
            os.remove(overlay_disk)
        except OSError:
            pass
    return {
        "overlay_path": overlay_path,
        "overlay_url": overlay_url,
        "overlay_width": orig_w,
        "overlay_height": orig_h,
        "gradcam_source_model": meta.get("source_model"),
        "gradcam_predicted_class": meta.get("predicted_class"),
        "gradcam_confidence": meta.get("confidence"),
        "gradcam_model_status": meta.get("model_status"),
        "gradcam_explanation_status": meta.get("explanation_status"),
        "gradcam_label": meta.get("gradcam_label"),
        "gradcam_explanation": meta.get("gradcam_explanation"),
        "gradcam_reliability": meta.get("gradcam_reliability"),
        "gradcam_overlay_generated": meta.get("overlay_generated"),
        "gradcam_withheld_reason": meta.get("withheld_reason"),
    }


def _apply_vision_image_provenance(vision_results: dict, case: dict, sensor_is_demo: bool) -> None:
    """Label image provenance from the image source, not from demo/synthetic sensors.

    A user-uploaded photo must not be rewritten as a synthetic demonstration image
    merely because demo sensor CSV was attached to the same case.
    """
    if not vision_results:
        return
    if case.get("is_demo"):
        vision_results["source_type"] = "demo"
        vision_results["data_provenance"] = "synthetic"
        vision_results["clinical_validity"] = False
        vision_results["display_message"] = (
            "Synthetic demonstration image — not a real patient image."
        )
        return
    vision_results["source_type"] = "uploaded"
    vision_results["data_provenance"] = "user_provided"
    vision_results["clinical_validity"] = False
    vision_results["sensor_data_is_demo"] = bool(sensor_is_demo)
    if sensor_is_demo:
        vision_results["display_message"] = (
            "User-uploaded image. Sensor data for this case is demo/synthetic; "
            "modalities are not clinically paired. Research/demo only."
        )
    else:
        vision_results["display_message"] = (
            "User-uploaded image analyzed for research/demo only — not clinical diagnosis."
        )


@app.post("/api/cases/{case_id}/analyze", tags=["Cases"])
def analyze_case(case_id: str):
    """Trigger the real multimodal classical-quantum hybrid analysis pipeline."""
    db = get_database()
    case = db.cases.find_one({"case_id": case_id})
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # 1. Ensure models are trained and checkpoints exist
    from ml.classifiers.xgboost_classifier import XGBoostClassifier
    from ml.classifiers.vqc_classifier import VQCClassifier
    from ml.fusion.feature_fusion import MultimodalFeatureFusion
    from ml.fusion.rules_engine import RulesEngine
    from ml.explainability.evidence_consistency import EvidenceConsistencyAnalyzer
    from ml.models.canonical_paths import XGB_CANONICAL, VQC_DIR

    require_model_artifacts()
    xgb_path = XGB_CANONICAL
    vqc_dir = VQC_DIR

    # 2. Gather Modality Inputs
    # 2a. Vision
    img_ref = case.get("image_reference")
    vision_results = {}
    if img_ref and os.path.exists(img_ref):
        try:
            # Run actual image pipeline!
            from ml.vision.preprocess import preprocess_image_for_inference
            from ml.vision.unet_wrapper import interpret_segmentation
            from ml.vision.efficientnet_wrapper import interpret_prediction
            
            yolo_det = _get_yolo_detector()
            unet_seg = _get_unet_segmenter()
            effnet_clf = _get_effnet_classifier()
            yolo_class_names = list(yolo_det.class_list) if yolo_det.class_list else []

            if yolo_det.model is None or yolo_det.status == "MODEL_ARTIFACT_MISSING":
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"MODEL_ARTIFACT_MISSING: {yolo_det.model_path}"
                )
            detections = yolo_det.detect(img_ref)
                
            if detections:
                best_det = max(detections, key=lambda d: float(d["confidence"]))
                finding_class = best_det["finding"]
                confidence = best_det["confidence"]
                bbox = best_det["bounding_box"]

                # Read original image for Grad-CAM (preserve original dimensions)
                import cv2 as _cv2_orig
                img_bgr_orig = _cv2_orig.imread(img_ref)
                img_rgb_orig = _cv2_orig.cvtColor(img_bgr_orig, _cv2_orig.COLOR_BGR2RGB)
                orig_h_real, orig_w_real = img_rgb_orig.shape[:2]

                # Letterbox metadata is retained for the API contract; U-Net uses original pixels.
                _, _, meta = preprocess_image_for_inference(img_ref)
                
                bbox_orig = bbox

                # Segment the original photograph ROI. Do not run U-Net on the
                # letterboxed 224 tensor — that would map a padded-space mask
                # into original-image coordinates.
                mask, pixel_count, affected_ratio, debug_info = unet_seg.segment(img_rgb_orig, bbox_orig)
                parsed_seg = interpret_segmentation(mask, pixel_count, affected_ratio, debug_info)
                
                # Crop ROI from original image for classification
                x1o, y1o, x2o, y2o = [int(v) for v in bbox_orig]
                x1o = max(0, x1o); y1o = max(0, y1o)
                x2o = min(orig_w_real, x2o); y2o = min(orig_h_real, y2o)
                roi = img_rgb_orig[y1o:y2o, x1o:x2o] if (y2o > y1o and x2o > x1o) else img_rgb_orig
                
                # Classify using ROI — withhold confident injury labels on invalid input
                probs = effnet_clf.predict(roi)
                parsed = interpret_prediction(probs)

                cam = _gradcam_payload(effnet_clf, img_rgb_orig, parsed, case_id, orig_w_real, orig_h_real)
                
                import cv2
                image_filename = os.path.basename(img_ref) 
                mask_filename = f"{case_id}_mask.png"
                mask_path = os.path.join(UPLOAD_DIR, mask_filename)
                full_mask = _compose_full_image_mask(mask, orig_h_real, orig_w_real, bbox_orig)
                cv2.imwrite(mask_path, full_mask * 255)
                
                YOLO_SUPPORTED_D = set(yolo_class_names)
                yolo_info_live = yolo_det.get_info()
                seg_is_reliable = parsed_seg["is_reliable"]
                vision_results = {
                    # YOLO canonical fields — confirmed detection
                    "yolo_finding": finding_class.capitalize(),
                    "yolo_finding_detected": True,
                    "yolo_supported_classes": sorted(list(YOLO_SUPPORTED_D)),
                    "yolo_validated_classes": yolo_info_live.get("validated_classes") or [],
                    "yolo_unsupported_classes": yolo_info_live.get("unsupported_classes") or [],
                    "yolo_class_support": yolo_info_live.get("class_support") or {},
                    "yolo_class_support_status": yolo_det.class_status(finding_class),
                    "yolo_promotion_status": yolo_info_live.get("promotion_status"),
                    "yolo_dataset_provenance": yolo_info_live.get("dataset_provenance"),
                    "yolo_confidence": confidence,
                    "yolo_bounding_box": bbox_orig,

                    # EfficientNet classifier result (separate from YOLO detection)
                    "classifier_finding": parsed["winner"] or parsed.get("abstention_class"),
                    "classifier_probability": parsed["max_prob"],
                    "classifier_status": parsed["status"],
                    "classifier_model_status": _effnet_training_status(),
                    "classifier_reason": parsed["reason"],
                    "classifier_is_confident": parsed["is_confident"],
                    "classifier_is_injury_finding": parsed.get("is_injury_finding", parsed["is_confident"]),
                    "classifier_abstention_class": parsed.get("abstention_class"),
                    "classifier_category_type": "research_classifier",
                    "classifier_yolo_coverage": _classifier_yolo_coverage(parsed, yolo_class_names),
                    "segmentation_model_status": _unet_training_status(),

                    # Legacy fields for backward compat
                    "finding": finding_class.capitalize(),
                    "finding_detected": True,
                    "confidence": confidence,
                    "bounding_box": bbox_orig,
                    "bounding_box_model": bbox_orig,
                    "affected_ratio": affected_ratio if seg_is_reliable else None,
                    "affected_area_ratio": affected_ratio if seg_is_reliable else None,
                    "segmentation_available": seg_is_reliable,
                    "segmentation_reliable": seg_is_reliable,
                    "segmentation_reason": parsed_seg["display_message"] or "No reliable segmentation mask available.",
                    "classification": parsed["class_probs"],
                    **cam,
                    "mask_path": mask_path,
                    "mask_url": f"/uploads/{mask_filename}",
                    "image_url": f"/uploads/{image_filename}",
                    "original_width": orig_w_real,
                    "original_height": orig_h_real,
                    "preprocessing_meta": meta,
                    "segmentation_debug": debug_info,
                    "segmentation_status": parsed_seg["status"],
                    "segmentation_message": parsed_seg["display_message"],
                    "segmentation_trust": parsed_seg["status"],
                    "model_version": best_det.get("model_version", "v1.0.0"),
                    "model_status": best_det.get("model_status", "UNKNOWN"),
                    "model_path": best_det.get("model_path", "")
                }
            else:
                # YOLO found nothing — do NOT create a fallback bounding box
                # Read original image
                import cv2 as _cv2_orig2
                img_bgr_orig2 = _cv2_orig2.imread(img_ref)
                img_rgb_orig2 = _cv2_orig2.cvtColor(img_bgr_orig2, _cv2_orig2.COLOR_BGR2RGB)
                orig_h_real2, orig_w_real2 = img_rgb_orig2.shape[:2]

                tensor, img_rgb, meta = preprocess_image_for_inference(img_ref)
                probs = effnet_clf.predict(img_rgb)
                parsed = interpret_prediction(probs)
                effnet_is_confident = parsed["is_confident"]
                effnet_max_prob = parsed["max_prob"]
                winning_class = parsed["winner"]
                winning_prob = parsed["max_prob"]
                is_low_confidence = parsed["low_confidence"]
                confidence_status_name = "Confident" if effnet_is_confident else "Withheld"
                clean_probs = parsed["class_probs"]

                YOLO_SUPPORTED = set(yolo_class_names)
                classifier_class_lower = (winning_class or "").lower()
                if not effnet_is_confident or not winning_class:
                    yolo_coverage = "NOT APPLICABLE — classifier output withheld"
                elif classifier_class_lower in YOLO_SUPPORTED:
                    yolo_coverage = "AVAILABLE"
                else:
                    yolo_coverage = "NOT AVAILABLE"

                # Segment the original photograph (not the letterboxed 224 tensor)
                mask, pixel_count, affected_ratio, debug_info = unet_seg.segment(img_rgb_orig2)
                parsed_seg = interpret_segmentation(mask, pixel_count, affected_ratio, debug_info)
                seg_is_reliable = parsed_seg["is_reliable"]

                cam = _gradcam_payload(effnet_clf, img_rgb_orig2, parsed, case_id, orig_w_real2, orig_h_real2)

                import cv2
                image_filename = os.path.basename(img_ref)
                mask_filename = f"{case_id}_mask.png"
                mask_path = os.path.join(UPLOAD_DIR, mask_filename)
                full_mask = _compose_full_image_mask(mask, orig_h_real2, orig_w_real2, None)
                cv2.imwrite(mask_path, full_mask * 255)

                vision_results = {
                    # YOLO canonical fields — explicitly null (YOLO detected nothing)
                    "yolo_finding": None,
                    "yolo_finding_detected": False,
                    "yolo_supported_classes": sorted(list(YOLO_SUPPORTED)),
                    "yolo_validated_classes": (yolo_det.get_info() or {}).get("validated_classes") or [],
                    "yolo_unsupported_classes": (yolo_det.get_info() or {}).get("unsupported_classes") or [],
                    "yolo_class_support": (yolo_det.get_info() or {}).get("class_support") or {},
                    "yolo_class_support_status": None,
                    "yolo_promotion_status": (yolo_det.get_info() or {}).get("promotion_status"),
                    "yolo_dataset_provenance": (yolo_det.get_info() or {}).get("dataset_provenance"),
                    "yolo_confidence": None,
                    "yolo_bounding_box": None,

                    # EfficientNet research classifier fields — labeled as classifier, NOT detector
                    "classifier_finding": winning_class or parsed.get("abstention_class"),
                    "classifier_probability": winning_prob if winning_class else parsed.get("max_prob"),
                    "classifier_status": parsed["status"],
                    "classifier_model_status": _effnet_training_status(),
                    "classifier_reason": parsed["reason"],
                    "classifier_category_type": "research_classifier",
                    "classifier_yolo_coverage": yolo_coverage,
                    "segmentation_model_status": _unet_training_status(),
                    "classifier_low_confidence": is_low_confidence,
                    "classifier_confidence_status": confidence_status_name,
                    "classifier_is_confident": effnet_is_confident,
                    "classifier_is_injury_finding": parsed.get("is_injury_finding", effnet_is_confident),
                    "classifier_abstention_class": parsed.get("abstention_class"),
                    "classifier_max_prob": effnet_max_prob,

                    # Legacy detection field stays empty on a YOLO miss.
                    # Do not copy EfficientNet "Swelling" into `finding`.
                    "finding": None,
                    "raw_finding": winning_class,
                    "finding_detected": False,
                    "detection_message": (
                        f"Visible finding not confidently detected by YOLO11 "
                        f"(supported: {', '.join(sorted(YOLO_SUPPORTED))}). "
                        + (
                            f"Research classifier category: {winning_class}."
                            if effnet_is_confident and winning_class
                            else f"Research classifier output withheld ({parsed['status']}: {parsed['reason']})."
                        )
                    ),
                    # Confidence/bbox remain null — no YOLO detection
                    "confidence": None,
                    "bounding_box": None,
                    "affected_ratio": affected_ratio if seg_is_reliable else None,
                    "affected_area_ratio": affected_ratio if seg_is_reliable else None,
                    "segmentation_available": seg_is_reliable,
                    "segmentation_reliable": seg_is_reliable,
                    "segmentation_reason": parsed_seg["display_message"] or "No reliable segmentation mask available.",
                    "classification": clean_probs,
                    **cam,
                    "mask_path": mask_path,
                    "mask_url": f"/uploads/{mask_filename}",
                    "image_url": f"/uploads/{image_filename}",
                    "original_width": orig_w_real2,
                    "original_height": orig_h_real2,
                    "preprocessing_meta": meta,
                    "segmentation_debug": debug_info,
                    "segmentation_status": parsed_seg["status"],
                    "segmentation_message": parsed_seg["display_message"],
                    "denominator_label": "full image",
                    "segmentation_trust": parsed_seg["status"],
                }

        except HTTPException:
            raise
        except Exception as e:
            # Vision pipeline failed — record error honestly, do NOT inject hardcoded values
            import traceback
            pipeline_error = traceback.format_exc()
            print(f"Vision pipeline FAILED: {str(e)}")
            print(pipeline_error)

            # Attempt to read actual image dimensions even in fallback
            actual_w, actual_h = None, None
            try:
                import cv2 as _cv2_dim
                _img_dim = _cv2_dim.imread(img_ref)
                if _img_dim is not None:
                    actual_h, actual_w = _img_dim.shape[:2]
            except (OSError, ValueError) as e:
                print(f"Could not read image dimensions after vision failure: {e}")

            vision_results = {
                "finding": "Unknown",
                "finding_detected": False,
                "detection_message": f"Vision pipeline execution failed. Error: {str(e)[:200]}",
                "confidence": None,                         # NOT hardcoded
                "bounding_box": None,                       # NO hardcoded fallback box
                "affected_ratio": None,                     # NOT hardcoded
                "classification": None,                     # NOT hardcoded
                "overlay_url": None,
                "overlay_width": actual_w,
                "overlay_height": actual_h,
                "mask_url": None,
                "image_url": f"/uploads/{os.path.basename(img_ref)}",
                "original_width": actual_w,
                "original_height": actual_h,
                "segmentation_status": "Not confidently detected",
                "segmentation_message": f"Vision pipeline failed — segmentation not available. Error: {str(e)[:150]}",
                "gradcam_source_model": "EfficientNetV2",
                "gradcam_predicted_class": None,
                "gradcam_confidence": None,
                "gradcam_model_status": "MODEL_UNAVAILABLE",
                "gradcam_explanation_status": "WITHHELD",
                "gradcam_label": "MODEL VISUALIZATION",
                "gradcam_explanation": "NOT CLINICAL EXPLANATION",
                "gradcam_reliability": "NOT_CLINICAL_EXPLANATION",
                "gradcam_overlay_generated": False,
                "gradcam_withheld_reason": "vision_pipeline_failed",

                "pipeline_error": str(e)[:300],
                "denominator_label": "full image"
            }
    
    # 2b. Questionnaire
    q_data = case.get("questionnaire") or {}
    
    # 2c. Sensor — track availability and do NOT fabricate missing values
    sensor_available = case.get("sensor_available", False)
    s_data = case.get("sensor_summary") or {}

    # Determine which modalities are active
    modalities_used = []
    if vision_results:
        modalities_used.append("image")
    if q_data:
        modalities_used.append("questionnaire")
    if sensor_available and s_data:
        modalities_used.append("sensor")

    # Select the correct model configuration from registry
    modality_set = set(modalities_used)
    selected_config = None
    for entry in MODEL_REGISTRY:
        if set(entry["modality_configuration"]) == modality_set:
            selected_config = entry
            break
    # If exact match not found, try subset (e.g., image+questionnaire when sensor absent)
    if selected_config is None:
        for entry in MODEL_REGISTRY:
            if set(entry["modality_configuration"]).issubset(modality_set):
                selected_config = entry
                break

    model_config_note = None
    if selected_config is None:
        selected_config = {
            "model_id": "xgb_option_a_available_modalities",
            "model_version": "1.0",
            "modality_configuration": modalities_used,
            "path": "ml/models/xgboost_best.json",
            "type": "xgboost",
            "reduced_modality": True,
            "reduced_modality_note": "No exact MODEL_REGISTRY row. Same 23-d XGBoost with Option A zeros for absent modalities.",
        }
        model_config_note = selected_config["reduced_modality_note"]
        print(f"NOTICE: Using Option A XGBoost for modalities {modality_set} (no exact registry row).")

    # 3. Fuse Modalities into a Feature Vector
    # When sensor is not available, pass empty sensor dict so fusion layer
    # uses its own zero-masking for the sensor dimensions (not fabricated values).
    fusion = MultimodalFeatureFusion()
    case_data = {
        "vision_analysis": {
            "classification": vision_results.get("classification"),
            "segmentation": {"affected_ratio": vision_results.get("affected_ratio", 0.0)}
        } if vision_results else {},
        "questionnaire": q_data,
        "sensor_summary": {
            "peak_g_force": s_data.get("peak_g_force"),
            "pre_impact_delta_v": s_data.get("pre_impact_delta_v"),
            "post_impact_stabilization_seconds": s_data.get("post_impact_stabilization_seconds"),
            "optical_lux_drop": s_data.get("optical_lux_drop"),
        } if (sensor_available and s_data) else {},
    }
    
    _, vector, names = fusion.fuse_features(case_data)

    # 4. Evaluate Heuristics
    rules = RulesEngine()
    rule_label, justifications = rules.evaluate_rules(vector, names)

    # 5. Run Classical Classifier (XGBoost) & Explainers (SHAP)
    # Always load the canonical artifact. Never auto-train a replacement.
    try:
        xgb_model = XGBoostClassifier(XGB_CANONICAL)
        if (not xgb_model.is_trained) or xgb_model.status == "MODEL_ARTIFACT_MISSING":
            raise RuntimeError(f"MODEL_ARTIFACT_MISSING: {XGB_CANONICAL}")
        pred_xgb, probs_xgb = xgb_model.predict(vector)
        shap_exps = xgb_model.explain_prediction(vector, pred_xgb)
    except (RuntimeError, FileNotFoundError, OSError) as e:
        if "MODEL_ARTIFACT_MISSING" in str(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )
        raise

    # 6. Experimental VQC (PennyLane simulator). Isolated from the main decision.
    # Failures must not fabricate a class/probabilities and must not break analyze.
    from ml.classifiers.vqc_classifier import EXPERIMENTAL_ONLY, MODEL_UNAVAILABLE
    label_map = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
    xgb_class_str = label_map[pred_xgb]
    vqc_status = MODEL_UNAVAILABLE
    vqc_error = None
    vqc_model = None
    pred_vqc = None
    scores_vqc = None
    vqc_class_str = None
    try:
        vqc_model = VQCClassifier(vqc_dir)
        pred_vqc, scores_vqc = vqc_model.predict(vector)
        vqc_class_str = label_map[int(pred_vqc)]
        vqc_status = EXPERIMENTAL_ONLY
    except (RuntimeError, FileNotFoundError, OSError, ValueError, KeyError, TypeError) as e:
        pred_vqc = None
        scores_vqc = None
        vqc_class_str = None
        vqc_model = None
        vqc_status = MODEL_UNAVAILABLE
        vqc_error = f"{type(e).__name__}: {e}"[:300]
    
    seg_reliable = bool(vision_results.get("segmentation_reliable", False)) if vision_results else False
    sensor_is_demo = s_data.get("source_type") in ["demo", "simulated"]
    is_demo = bool(case.get("is_demo", False) or sensor_is_demo)
    _apply_vision_image_provenance(vision_results, case, sensor_is_demo)
    if vision_results and img_ref and os.path.exists(img_ref):
        import hashlib as _hashlib_vision
        with open(img_ref, "rb") as _img_fh:
            vision_results["image_sha256"] = _hashlib_vision.sha256(_img_fh.read()).hexdigest()
        vision_results["image_reference"] = img_ref

    # 7. Run Consistency and Counterfactual Sweeps
    consistency_analyzer = EvidenceConsistencyAnalyzer()
    consistency_res = consistency_analyzer.calculate_consistency(
        vector, 
        names,
        xgb_prediction_class=xgb_class_str,
        vqc_prediction_class=None,
        segmentation_reliable=seg_reliable,
        is_demo=is_demo
    )
    try:
        counterfactuals = consistency_analyzer.analyze_counterfactuals(vector, names, xgb_model, vqc_model)
    except (RuntimeError, ValueError, TypeError, KeyError, OSError) as e:
        counterfactuals = {
            "explanation": "Counterfactual sweep skipped after VQC/experimental failure.",
            "error": f"{type(e).__name__}: {e}"[:200],
            "baseline_predictions": {"classical_xgb": xgb_class_str, "quantum_vqc": None},
        }

    # 8. Save all calculations back to MongoDB
    agreement_score_str = "VQC_EXCLUDED_FROM_DECISION"
    max_xgb_prob = float(max(probs_xgb))
    
    uncertainty_reasons = []
    if not seg_reliable:
        uncertainty_reasons.append("Segmentation output was unavailable or below the reliability threshold.")
    if is_demo:
        uncertainty_reasons.append("This case uses synthetic or demo data.")
    if max_xgb_prob < 0.70:
        uncertainty_reasons.append(f"XGBoost model confidence is moderate ({max_xgb_prob * 100:.1f}%).")

    if len(uncertainty_reasons) >= 2:
        uncertainty_level_str = "HIGH UNCERTAINTY"
    elif len(uncertainty_reasons) == 1:
        uncertainty_level_str = "MODERATE UNCERTAINTY"
    else:
        uncertainty_level_str = "LOW UNCERTAINTY"

    xgb_info = xgb_model.get_info()
    xgboost_out = {
        "class": xgb_class_str,
        "probability": float(probs_xgb[pred_xgb]),
        "model_version": xgb_info.get("model_version") or selected_config["model_version"],
        "model_id": selected_config["model_id"],
        "n_features": 23,
        "data_provenance": xgb_info.get("data_provenance") or "SYNTHETIC",
        "data_provenance_detail": xgb_info.get("data_provenance_detail") or "synthetic_multimodal_fusion",
        "status": xgb_info.get("metadata_status") or xgb_info.get("status"),
        "artifact_path": XGB_CANONICAL.replace("\\", "/"),
        "label_source": "SYNTHETIC_RULE_LABELS",
        "paired_clinical_samples": 0,
        "clinical_claim_blocked": True,
        "clinical_claim": "BLOCKED_NO_PAIRED_CLINICAL_LABELS",
    }
    pca_features = None
    if vqc_model is not None and pred_vqc is not None:
        try:
            import numpy as np
            x_in = np.asarray(vector, dtype=np.float64).reshape(1, -1)
            x_scaled = vqc_model.scaler.transform(x_in)
            pca_features = np.clip(vqc_model.pca.transform(x_scaled)[0], -3.0, 3.0).astype(float).tolist()
        except (ValueError, TypeError, AttributeError) as e:
            print(f"VQC PCA projection unavailable: {e}")
            pca_features = None
    quantum_out = {
        "class": vqc_class_str,
        "score": scores_vqc,
        "model_version": "v1.4.0",
        "model_id": "vqc_sim_v1",
        "status": vqc_status,
        "experimental": True,
        "experimental_only": True,
        "used_in_main_decision": False,
        "data_provenance": "SYNTHETIC",
        "error": vqc_error,
        "pca_features": pca_features,
        "pca_note": (
            "Four PCA components from the train-fitted VQC scaler/PCA on this case vector. "
            "Not stored as clinical evidence."
            if pca_features is not None
            else "PCA components were not stored for this case (VQC unavailable or projection failed)."
        ),
    }

    from backend.services.first_aid_service import first_aid_service
    first_aid_guidance = first_aid_service.generate_first_aid_guidance(
        questionnaire_answers=(case.get("questionnaire") or {}).get("answers") or {},
        sensor_summary=case.get("sensor_summary"),
        visible_injury=vision_results,
        rule_derived_category=rule_label,
        xgboost_pred=xgboost_out,
        quantum_pred=quantum_out
    )

    db.cases.update_one(
        {"case_id": case_id},
        {"$set": {
            "visible_injury": vision_results,
            "xgboost_prediction": xgboost_out,
            "quantum_prediction": quantum_out,
            "agreement_score": agreement_score_str,
            "uncertainty_status": uncertainty_level_str,
            "uncertainty_level": uncertainty_level_str,
            "uncertainty_reasons": uncertainty_reasons,
            "rule_derived_category": rule_label,
            "safety_guidance_level": rule_label,
            "justification": justifications,
            "safety_information": [justifications],
            "first_aid_guidance": first_aid_guidance,
            "consistency_analysis": consistency_res,
            "counterfactual_analysis": counterfactuals,
            "shap_explanations": shap_exps,
            "status": "analyzed",
            "modalities_used": modalities_used,
            "model_configuration_used": selected_config["model_id"],
            "fusion_label_source": "SYNTHETIC_RULE_LABELS",
            "clinical_claim_blocked": True,
            "clinical_claim": "BLOCKED_NO_PAIRED_CLINICAL_LABELS",
            "paired_clinical_samples": 0,
        }}
    )


    return {
        "message": "Multimodal analysis completed successfully",
        "xgboost": xgboost_out,
        "quantum": quantum_out,
        "consistency": consistency_res,
        "modalities_used": modalities_used,
        "model_configuration_used": selected_config["model_id"],
        "clinical_claim_blocked": True,
        "fusion_label_source": "SYNTHETIC_RULE_LABELS",
        "paired_clinical_samples": 0,
        "clinical_claim": "BLOCKED_NO_PAIRED_CLINICAL_LABELS",
    }


@app.get("/api/cases/{case_id}/report", tags=["Cases"])
def get_report(case_id: str):
    """Retrieve the compiled research report structure."""
    from backend.services.report_service import ResearchReportGenerator
    generator = ResearchReportGenerator()
    try:
        report_data = generator.compile_report_data(case_id)
        return report_data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/cases/{case_id}/report/json", tags=["Cases"])
def download_json_report(case_id: str):
    """Download the JSON research report file."""
    from backend.services.report_service import ResearchReportGenerator
    from fastapi.responses import JSONResponse
    generator = ResearchReportGenerator()
    try:
        report_data = generator.compile_report_data(case_id)
        return JSONResponse(
            content=report_data,
            headers={"Content-Disposition": f"attachment; filename=report_{case_id}.json"}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/cases/{case_id}/report/pdf", tags=["Cases"])
def download_pdf_report(case_id: str):
    """Download the PDF research report file."""
    from backend.services.report_service import ResearchReportGenerator
    from fastapi.responses import Response
    generator = ResearchReportGenerator()
    try:
        pdf_bytes = generator.generate_pdf_bytes(case_id)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=report_{case_id}.pdf"}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/sos/demo", tags=["Emergency Demo"])
def trigger_sos_demo(payload: SOSDemoSchema):
    """Log an emergency simulation event (no real calls made)."""
    db = get_database()
    event_id = str(uuid.uuid4())
    event_doc = {
        "event_id": event_id,
        "case_id": payload.case_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sensor_intensity": payload.sensor_intensity,
        "user_acknowledged": payload.user_acknowledged,
        "alert_sent": False,
        "delivery_outcome": "cancelled" if payload.user_acknowledged else "LOCAL_SIMULATION",
        "status": "cancelled" if payload.user_acknowledged else "LOCAL_SIMULATION",
        "twilio_message_sid": None,
        "warning": "NO REAL EMERGENCY SERVICES WERE CONTACTED. No SMS was sent.",
    }
    
    db.sos_events.insert_one(event_doc)
    event_doc.pop("_id", None)
    
    # Update case document as well
    db.cases.update_one(
        {"case_id": payload.case_id},
        {"$set": {"sos_status": event_doc["status"]}}
    )
    
    return event_doc


def _model_meta_status(metadata_path: str, fallback: str) -> str:
    """Read honest training status from metadata JSON. File existence is not training proof."""
    from ml.models.canonical_paths import exists, resolve_existing
    located = resolve_existing(metadata_path)
    if not exists(located):
        return fallback
    try:
        with open(located, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return str(meta.get("status") or fallback)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not read model metadata {metadata_path}: {e}")
        return fallback


def _effnet_classifier_categories() -> list:
    """Advertise classes from the loaded checkpoint sidecar, not a hardcoded taxonomy."""
    from ml.models.canonical_paths import ROOT, exists
    sidecar = os.path.join(ROOT, "ml", "models", "vision", "efficientnetv2_injury_best_classes.json")
    if exists(sidecar):
        try:
            with open(sidecar, encoding="utf-8") as handle:
                names = json.load(handle)
            if isinstance(names, list) and names:
                return [str(n).lower() for n in names]
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return ["cut", "bruise", "normal", "ood_reject"]


@app.get("/api/models", tags=["Research Models"])
def get_models():
    """Retrieve metadata about the currently deployed models dynamically from disk status."""
    from ml.vision.yolo_wrapper import YOLO11Detector, UNTRAINED_CLASS
    from ml.models.canonical_paths import (
        YOLO_CANONICAL,
        EFFNET_CANONICAL,
        UNET_CANONICAL,
        XGB_CANONICAL,
        VQC_DIR,
        VQC_WEIGHTS,
        EFFNET_METADATA,
        UNET_METADATA,
        sha256_file,
        exists,
        posix,
        abs_path,
    )
    from ml.classifiers.xgboost_classifier import (
        MODEL_ARTIFACT_MISSING,
        load_xgboost_metadata,
        provenance_class_from_metadata,
    )

    yolo_det = _get_yolo_detector()
    yolo_info = yolo_det.get_info()
    yolo_status = yolo_info["status"]
    yolo_loaded = yolo_det.model is not None
    yolo_path = yolo_info["model_path"] or YOLO_CANONICAL
    yolo_classes = list(yolo_info.get("classes") or yolo_info.get("supported_classes") or [])
    yolo_version = yolo_info.get("version") or yolo_info.get("model_version")
    yolo_provenance = yolo_info.get("training_data")
    yolo_task = yolo_info.get("task")
    yolo_sha = yolo_info.get("artifact_sha256")
    if yolo_loaded and exists(yolo_path) and not yolo_sha:
        yolo_sha = sha256_file(yolo_path)

    yolo_inference_ok = False
    if yolo_loaded:
        try:
            import numpy as np
            import cv2
            dummy_test_path = abs_path(os.path.join("data", "datasets", "yolo_injury", "dummy_test.jpg"))
            if not os.path.exists(dummy_test_path):
                os.makedirs(os.path.dirname(dummy_test_path), exist_ok=True)
                cv2.imwrite(dummy_test_path, np.zeros((64, 64, 3), dtype=np.uint8))
            yolo_det.detect(dummy_test_path)
            yolo_inference_ok = True
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            yolo_status = f"LOADED BUT INFERENCE FAILED: {str(e)}"
            yolo_inference_ok = False

    yolo_exists = yolo_loaded and yolo_inference_ok
    untrained_example = {
        name: UNTRAINED_CLASS
        for name in ("wound", "laceration", "swelling")
        if name not in set(yolo_classes)
    }

    effnet_exists = exists(EFFNET_CANONICAL)
    unet_exists = exists(UNET_CANONICAL)
    effnet_sha = sha256_file(EFFNET_CANONICAL) if effnet_exists else None
    unet_sha = sha256_file(UNET_CANONICAL) if unet_exists else None
    effnet_meta = {}
    unet_meta = {}
    if exists(EFFNET_METADATA):
        try:
            with open(abs_path(EFFNET_METADATA), encoding="utf-8") as handle:
                effnet_meta = json.load(handle) or {}
        except (OSError, json.JSONDecodeError):
            effnet_meta = {}
    if exists(UNET_METADATA):
        try:
            with open(abs_path(UNET_METADATA), encoding="utf-8") as handle:
                unet_meta = json.load(handle) or {}
        except (OSError, json.JSONDecodeError):
            unet_meta = {}
    xgb_meta = load_xgboost_metadata()
    xgb_exists = exists(XGB_CANONICAL)
    xgb_sha = sha256_file(XGB_CANONICAL) if xgb_exists else None
    xgb_provenance = provenance_class_from_metadata(xgb_meta)
    xgb_status = (xgb_meta.get("status") or "MODEL_LOADS") if xgb_exists else MODEL_ARTIFACT_MISSING
    vqc_npz = VQC_WEIGHTS
    vqc_exists = exists(vqc_npz)
    vqc_meta_path = os.path.join(VQC_DIR, "vqc_metadata.json")
    vqc_meta = {}
    if exists(vqc_meta_path):
        try:
            with open(vqc_meta_path, encoding="utf-8") as handle:
                vqc_meta = json.load(handle)
        except (OSError, json.JSONDecodeError):
            vqc_meta = {}
    vqc_status = (vqc_meta.get("status") or "EXPERIMENTAL_ONLY") if vqc_exists else "MODEL_UNAVAILABLE"

    return [
        {
            "model_name": "YOLO11",
            "model_version": yolo_version,
            "version": yolo_version,
            "status": yolo_status,
            "weights_loaded": yolo_exists,
            "model_path": yolo_path,
            "canonical_path": posix(YOLO_CANONICAL),
            "artifact_path": posix(yolo_path),
            "artifact_sha256": yolo_sha,
            "task": yolo_task,
            "classes": yolo_classes,
            "supported_classes": yolo_classes,
            "yolo_supported_classes": yolo_classes,
            "yolo11_supported_classes": yolo_classes,
            "validated_classes": yolo_info.get("validated_classes") or [],
            "unsupported_classes": yolo_info.get("unsupported_classes") or [],
            "class_support": yolo_info.get("class_support") or {},
            "promotion_status": yolo_info.get("promotion_status"),
            "dataset_provenance": yolo_info.get("dataset_provenance"),
            "untrained_classes": {**(yolo_info.get("untrained_classes") or {}), **untrained_example},
            "untrained_class_status": UNTRAINED_CLASS,
            "swelling_yolo_coverage": UNTRAINED_CLASS,
            "swelling_note": (
                "Swelling is UNTRAINED_CLASS for YOLO11 (not in model.names). "
                "The active EfficientNet kaggle-v1 head (8 classes) also does not include swelling "
                "(no honest labeled swelling training data). User-reported swelling in the "
                "questionnaire is separate from classifier predictions."
            ),
            "wound_note": (
                (
                    "Wound is in the promoted expanded-skin-v1 model.names with honest training boxes "
                    f"(classes: {', '.join(yolo_classes)})."
                )
                if "wound" in set(yolo_classes)
                else (
                    "Wound is not in the active YOLO model.names. Status=UNTRAINED_CLASS."
                )
            ),
            "training_dataset": yolo_provenance,
            "clinically_validated": False,
            "runtime_status": yolo_status
        },
        {
            "model_name": "EfficientNetV2",
            "model_version": effnet_meta.get("version") or "v1.3.0",
            "version": effnet_meta.get("version") or "v1.3.0",
            "status": _model_meta_status(EFFNET_METADATA, "MODEL_LOADS" if effnet_exists else "MODEL_UNAVAILABLE"),
            "training_status": _model_meta_status(EFFNET_METADATA, "MODEL_LOADS" if effnet_exists else "MODEL_UNAVAILABLE"),
            "weights_loaded": effnet_exists,
            "canonical_path": posix(EFFNET_CANONICAL),
            "artifact_path": posix(EFFNET_CANONICAL),
            "model_path": abs_path(EFFNET_CANONICAL) if effnet_exists else None,
            "artifact_sha256": effnet_sha,
            "classes": _effnet_classifier_categories(),
            "classifier_categories": _effnet_classifier_categories(),
            "category_type": "research_classifier",
            "status_note": "File existence is not TRAINED_AND_EVALUATED. Status comes from training metadata when present.",
            "note": (
                "EfficientNetV2 kaggle-v1 classes: abrasion, bruise, burn, cut, laceration, "
                "wound, normal, ood_reject. normal and ood_reject are abstention classes, not "
                "diagnoses. swelling is not in the active classifier head (no labeled swelling "
                "data). These categories are NOT equivalent to YOLO11 detections. Mixed real-photo "
                "Kaggle sources — research demo only, not clinical."
            )
        },
        {
            "model_name": "U-Net",
            "model_version": unet_meta.get("version") or "v1.0.0",
            "version": unet_meta.get("version") or "v1.0.0",
            "status": _model_meta_status(UNET_METADATA, "MODEL_LOADS" if unet_exists else "MODEL_UNAVAILABLE"),
            "training_status": _model_meta_status(UNET_METADATA, "MODEL_LOADS" if unet_exists else "MODEL_UNAVAILABLE"),
            "weights_loaded": unet_exists,
            "canonical_path": posix(UNET_CANONICAL),
            "artifact_path": posix(UNET_CANONICAL),
            "model_path": abs_path(UNET_CANONICAL) if unet_exists else None,
            "artifact_sha256": unet_sha,
            "status_note": "File existence is not TRAINED_AND_EVALUATED. Status comes from training metadata when present. Overlay gates withhold blank/OOD masks; they do not certify photographs.",
        },
        {
            "model_name": "XGBoost",
            "model_version": xgb_meta.get("version") if xgb_exists else None,
            "status": xgb_status,
            "weights_loaded": xgb_exists,
            "schema_features": 23,
            "canonical_path": posix(XGB_CANONICAL),
            "artifact_path": posix(XGB_CANONICAL),
            "artifact_sha256": xgb_sha,
            "data_provenance": xgb_provenance,
            "data_provenance_detail": xgb_meta.get("data_provenance"),
            "training_dataset": xgb_meta.get("data_provenance"),
            "status_note": (
                "Missing canonical artifact is MODEL_ARTIFACT_MISSING. "
                "File existence is not TRAINED_AND_EVALUATED. Status comes from xgboost_metadata.json."
            ),
        },
        {
            "model_name": "VQC",
            "model_version": vqc_meta.get("version") if vqc_exists else None,
            "status": vqc_status,
            "weights_loaded": vqc_exists,
            "canonical_path": posix(vqc_npz),
            "artifact_path": posix(vqc_npz),
            "artifact_sha256": sha256_file(vqc_npz) if vqc_exists else None,
            "data_provenance": vqc_meta.get("data_provenance_class") or "SYNTHETIC",
            "used_in_main_decision": False,
            "experimental_only": True,
            "recommendation": vqc_meta.get("recommendation"),
            "status_note": "EXPERIMENTAL_ONLY. Isolated from SOS, first-aid, and main case decisions. Missing artifact is MODEL_UNAVAILABLE, not a fabricated class.",
        }
    ]

@app.get("/api/models/registry", tags=["Models"])
def get_model_registry():
    """Retrieve full model registry from ml/models/model_registry.json.

    Each entry is enriched with display_* fields derived from the same metrics
    object (no hand-edited numbers) so the Research page does not invent N/A
    when the metric key name differs across models.
    """
    from ml.models.canonical_paths import REGISTRY_PATH, exists, resolve_existing
    registry_path = resolve_existing(REGISTRY_PATH)
    if exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            if isinstance(registry, dict):
                return {key: _enrich_registry_entry(value) for key, value in registry.items()}
            return registry
        except (OSError, json.JSONDecodeError) as e:
            return {"error": f"Failed to load model registry: {str(e)}"}
    return {"message": "Model registry not yet initialized."}


def _enrich_registry_entry(entry: dict) -> dict:
    """Attach unambiguous display fields from canonical registry metrics."""
    if not isinstance(entry, dict):
        return entry
    out = dict(entry)
    metrics = out.get("metrics") if isinstance(out.get("metrics"), dict) else {}
    train_n = metrics.get("train_samples")
    val_n = metrics.get("val_samples")
    test_n = metrics.get("test_samples")
    if train_n is None and out.get("sample_count") is not None:
        train_n = out.get("sample_count")
    # U-Net metadata stores split sizes under metrics.val.n / metrics.test.n
    if val_n is None and isinstance(metrics.get("val"), dict):
        val_n = metrics["val"].get("n")
    if test_n is None and isinstance(metrics.get("test"), dict):
        test_n = metrics["test"].get("n")
    if train_n is None and isinstance(metrics.get("train"), dict):
        train_n = metrics["train"].get("n")

    held_out = (
        metrics.get("correct_predictions")
        or metrics.get("vqc_correct_predictions")
        or metrics.get("overall_correct_predictions")
    )
    if not held_out and metrics.get("vqc_accuracy") is not None and test_n:
        held_out = f"{int(round(float(metrics['vqc_accuracy']) * int(test_n)))} / {int(test_n)}"
    if not held_out and metrics.get("overall_accuracy") is not None and test_n:
        held_out = f"{int(round(float(metrics['overall_accuracy']) * int(test_n)))} / {int(test_n)}"
    if not held_out and metrics.get("accuracy") is not None and test_n and "mAP50" not in metrics:
        # Sensor-style accuracy with test_samples
        held_out = metrics.get("correct_predictions") or (
            f"{int(round(float(metrics['accuracy']) * int(test_n)))} / {int(test_n)}"
            if test_n
            else None
        )

    map50 = metrics.get("mAP50")
    if map50 is None and isinstance(metrics.get("independent_test"), dict):
        map50 = metrics["independent_test"].get("mAP50")
    map5095 = metrics.get("mAP50-95")
    if map5095 is None and isinstance(metrics.get("independent_test"), dict):
        map5095 = metrics["independent_test"].get("mAP50-95")

    dice = None
    if isinstance(metrics.get("test"), dict) and metrics["test"].get("mean_dice") is not None:
        dice = metrics["test"]["mean_dice"]
    elif metrics.get("mean_dice_score") is not None:
        dice = metrics.get("mean_dice_score")

    status = str(out.get("status") or out.get("training_status") or "")
    if held_out:
        metric_display = str(held_out)
    elif map50 is not None:
        metric_display = f"mAP50: {float(map50):.4f}"
        if map5095 is not None:
            metric_display += f" | mAP50-95: {float(map5095):.4f}"
    elif dice is not None:
        metric_display = f"Dice(test): {float(dice):.4f}"
    elif "READY_FOR_RESEARCH_DEMO" in status:
        version = str(out.get("version") or "research-demo")
        test_block = metrics.get("test") if isinstance(metrics, dict) else None
        if isinstance(test_block, dict) and test_block.get("macro_f1") is not None:
            metric_display = f"macro-F1(test): {float(test_block['macro_f1']):.4f}"
        elif isinstance(test_block, dict) and test_block.get("accuracy") is not None:
            metric_display = f"accuracy(test): {float(test_block['accuracy']):.4f}"
        else:
            metric_display = f"READY_FOR_RESEARCH_DEMO ({version})"
    elif "NOT_TRUSTWORTHY" in status:
        metric_display = "NOT_TRUSTWORTHY - OOD collapse (gates withhold)"
    else:
        metric_display = "N/A"

    sample_parts = []
    if train_n is not None:
        sample_parts.append(f"train {train_n}")
    if val_n is not None:
        sample_parts.append(f"val {val_n}")
    if test_n is not None:
        sample_parts.append(f"test {test_n}")
    sample_display = " | ".join(sample_parts) if sample_parts else (
        str(out.get("sample_count")) if out.get("sample_count") is not None else "N/A"
    )

    # Keep metrics.correct_predictions aliased for VQC so older clients also see held-out.
    if metrics and not metrics.get("correct_predictions") and metrics.get("vqc_correct_predictions"):
        metrics = dict(metrics)
        metrics["correct_predictions"] = metrics["vqc_correct_predictions"]
        out["metrics"] = metrics

    out["display_sample_count"] = sample_display
    out["display_sample_count_note"] = (
        "Counts are split sizes from training metadata (train/val/test). "
        "Held-out metric uses the test split only."
    )
    out["display_held_out_metric"] = metric_display
    out["train_samples"] = train_n
    out["val_samples"] = val_n
    out["test_samples"] = test_n
    if map50 is not None:
        out["display_map50"] = round(float(map50), 4)
    if map5095 is not None:
        out["display_map50_95"] = round(float(map5095), 4)
    return out

@app.get("/api/evaluation", tags=["Evaluation"])
def get_evaluation():
    """Get active training evaluation parameters from data/results/evaluation_results.json."""
    from ml.models.canonical_paths import EVAL_RESULTS, exists, resolve_existing
    eval_json_path = resolve_existing(EVAL_RESULTS)
    if exists(eval_json_path):
        try:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Failed to read evaluation results JSON: {e}")
            raise HTTPException(status_code=500, detail=f"Invalid evaluation data format: {str(e)}")

    return {
        "status": "not_evaluated",
        "message": "Evaluation unavailable — model training or test set evaluation has not been completed."
    }

@app.get("/api/evaluation/comparison", tags=["Evaluation"])
def get_model_comparisons():
    """Research Results reads the canonical held-out artifact. Never auto-trains. Never uses hardcoded metrics."""
    from ml.models.canonical_paths import EVAL_COMPARE, EVAL_HELD_OUT, EVAL_RESULTS, exists, resolve_existing
    held_out_path = resolve_existing(EVAL_HELD_OUT)
    compare_path = resolve_existing(EVAL_COMPARE)
    eval_json_path = resolve_existing(EVAL_RESULTS)

    if exists(held_out_path):
        try:
            with open(held_out_path, encoding="utf-8") as handle:
                return _comparison_api_payload(json.load(handle))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Failed to read canonical held-out evaluation JSON: {e}")

    if exists(compare_path):
        try:
            with open(compare_path, encoding="utf-8") as handle:
                return _comparison_api_payload(json.load(handle))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Failed to read VQC comparison JSON: {e}")

    if exists(eval_json_path):
        try:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                mc = data.get("metrics_comparison", {})
                sc = data.get("selective_classification", {})
                return {
                    "status": "evaluated",
                    "sample_count": data.get("test_sample_count"),
                    "disclaimer": data.get("disclaimer", "Experimental Research Classification Performance"),
                    "quantum_simulator_notice": data.get("quantum_simulator_notice"),
                    "metrics_source": data.get("metrics_source"),
                    "canonical_artifact": "data/results/evaluation_results.json",
                    "classical_xgb": {
                        "xgb_correct": mc.get("xgb_correct"),
                        "accuracy": mc.get("xgb_accuracy"),
                        "precision": mc.get("xgb_precision"),
                        "recall": mc.get("xgb_recall"),
                        "macro_f1": mc.get("xgb_macro_f1"),
                        "mcc": mc.get("xgb_mcc"),
                        "ece": mc.get("xgb_ece"),
                        "brier_score": mc.get("xgb_brier"),
                        "confusion_matrix": mc.get("xgb_confusion_matrix"),
                    },
                    "quantum_vqc": {
                        "vqc_correct": mc.get("vqc_correct"),
                        "accuracy": mc.get("vqc_accuracy"),
                        "precision": mc.get("vqc_precision"),
                        "recall": mc.get("vqc_recall"),
                        "macro_f1": mc.get("vqc_macro_f1"),
                        "mcc": mc.get("vqc_mcc"),
                        "ece": mc.get("vqc_ece"),
                        "brier_score": mc.get("vqc_brier"),
                        "confusion_matrix": mc.get("vqc_confusion_matrix"),
                    },
                    "selective_classification": sc,
                    "used_in_main_decision": False,
                    "interpretation": "Experimental 4-Qubit VQC outputs vs XGBoost classical baseline. Isolated from main decision."
                }
        except (json.JSONDecodeError, OSError) as e:
            print(f"Failed to read evaluation comparison JSON: {e}")
            raise HTTPException(status_code=500, detail=f"Invalid evaluation data format: {str(e)}")

    return {
        "status": "not_evaluated",
        "sample_count": 0,
        "message": "Evaluation unavailable — run 'python -m ml.evaluation.run_all' to write canonical held-out metrics. This endpoint does not auto-train.",
        "used_in_main_decision": False,
    }

@app.get("/api/evaluation/ablation", tags=["Evaluation"])
def get_ablation_benchmarks():
    """Retrieve ablation studies and perturbations from evaluation artifacts."""
    from ml.models.canonical_paths import EVAL_RESULTS, exists, resolve_existing
    eval_json_path = resolve_existing(EVAL_RESULTS)
    if exists(eval_json_path):
        try:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "status": "evaluated",
                    "sample_count": data.get("test_sample_count"),
                    "ablation_study": data.get("ablation_study", [])
                }
        except (json.JSONDecodeError, OSError) as e:
            print(f"Failed to read evaluation ablation JSON: {e}")
            raise HTTPException(status_code=500, detail=f"Invalid evaluation data format: {str(e)}")

    return {
        "status": "not_evaluated",
        "sample_count": 0,
        "message": "Ablation metrics unavailable — model training or evaluation incomplete."
    }

