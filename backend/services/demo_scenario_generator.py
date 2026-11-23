"""
demo_scenario_generator.py

Generates diverse, randomized, clinically plausible demonstration cases for AI-QTriage.
Provides diverse combinations of:
- Injury images (wound, cut, bruise, swelling, abrasions, burns)
- Questionnaire responses (varying pain 1-10, mechanisms, mobility, deformity, sensations)
- Sensor telemetry (varying kinetic impacts 1.2g - 8.5g, or omitted for optional sensor demos)
Produces different multimodal fusion vectors, different XGBoost probabilities, and different Gemini safety/SOS outputs on each run.
"""

import os
import shutil
import random
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_DIR = os.path.join(BASE_DIR, "data", "sample")
SAMPLE_SENSOR_SRC = os.path.join(SAMPLE_DIR, "sensor", "football_fall.csv")


def generate_synthetic_image(dest_path: str, injury_type: str = "wound") -> str:
    """Dynamically renders a high-quality synthetic injury image to ensure visual variety."""
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
        
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    
    # Random base skin tone
    skin_b = random.randint(120, 150)
    skin_g = random.randint(145, 175)
    skin_r = random.randint(200, 235)
    img[:, :] = [skin_b, skin_g, skin_r]
    
    center_x = random.randint(140, 180)
    center_y = random.randint(140, 180)
    
    if injury_type == "bruise":
        # Multi-layered contusion with purple/yellow hues
        cv2.circle(img, (center_x, center_y), random.randint(55, 75), (90, 130, 150), -1)
        cv2.circle(img, (center_x, center_y), random.randint(35, 50), (120, 60, 90), -1)
        cv2.circle(img, (center_x, center_y), random.randint(15, 25), (70, 30, 60), -1)
    elif injury_type == "swelling":
        # Diffuse joint enlargement with shaded contours
        cv2.circle(img, (center_x, center_y), random.randint(70, 90), (100, 115, 175), -1)
        cv2.ellipse(img, (center_x, center_y), (60, 40), random.randint(0, 180), 0, 360, (85, 100, 155), -1)
    elif injury_type == "cut":
        # Sharp linear laceration with erythematous margins
        pt1 = (center_x - random.randint(40, 70), center_y - random.randint(20, 40))
        pt2 = (center_x + random.randint(40, 70), center_y + random.randint(20, 40))
        cv2.line(img, pt1, pt2, (40, 30, 190), random.randint(4, 7))
        cv2.circle(img, ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2), 15, (50, 40, 200), -1)
    else:  # laceration / open wound / burn
        cv2.ellipse(img, (center_x, center_y), (random.randint(45, 65), random.randint(20, 35)), random.randint(0, 360), 0, 360, (30, 20, 180), -1)
        cv2.circle(img, (center_x, center_y), random.randint(12, 20), (15, 10, 130), -1)

    # Texture & grain noise so it satisfies contrast & non-blur constraints
    noise = np.random.randint(0, 50, (320, 320, 3), dtype=np.int16)
    img_textured = np.clip(img.astype(np.int16) + noise - 25, 0, 255).astype(np.uint8)
    
    # Border & research label
    cv2.rectangle(img_textured, (6, 6), (314, 314), (20, 20, 20), 2)
    label_txt = f"AI-QTriage Demo: {injury_type.upper()}"
    cv2.putText(img_textured, label_txt, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

    cv2.imwrite(dest_path, img_textured)
    return dest_path


def create_scaled_sensor_csv(dest_path: str, target_peak_g: float) -> str:
    """Scales baseline football fall CSV to achieve exact target peak acceleration while preserving valid 50Hz structure."""
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
        
    if os.path.exists(SAMPLE_SENSOR_SRC):
        df = pd.read_csv(SAMPLE_SENSOR_SRC)
        baseline_peak_g = 5.17
        scale = float(target_peak_g / baseline_peak_g)
        
        accel_cols = [c for c in ["accelerometer_x", "accelerometer_y", "accelerometer_z", "accel_x", "accel_y", "accel_z"] if c in df.columns]
        for c in accel_cols:
            df[c] = np.round(df[c] * scale, 2)
            
        gyro_cols = [c for c in ["gyroscope_x", "gyroscope_y", "gyroscope_z", "gyro_x", "gyro_y", "gyro_z"] if c in df.columns]
        for c in gyro_cols:
            df[c] = np.round(df[c] * min(2.0, max(0.5, scale)), 2)
            
        df.to_csv(dest_path, index=False)
        return dest_path
    return SAMPLE_SENSOR_SRC


SCENARIOS = [
    {
        "name": "High-Impact Sports Knee Collision",
        "injury_type": "wound",
        "preset_img": os.path.join(SAMPLE_DIR, "image", "football_injury.jpg"),
        "location": "knee",
        "mechanism": "sports",
        "pain_range": (7, 9),
        "movement": "severe",
        "weight_bearing": "no",
        "visible_bleeding": "no",
        "crack_pop": "yes",
        "direct_impact": "yes",
        "deformity": "yes",
        "target_g_range": (5.8, 8.4),
        "include_sensor": True,
    },
    {
        "name": "Lateral Ankle Inversion Sprain & Swelling",
        "injury_type": "swelling",
        "preset_img": os.path.join(SAMPLE_DIR, "qa", "qa_swelling_offcenter.jpg"),
        "location": "ankle",
        "mechanism": "fall",
        "pain_range": (4, 6),
        "movement": "moderate",
        "weight_bearing": "partial",
        "visible_bleeding": "no",
        "crack_pop": "no",
        "direct_impact": "no",
        "deformity": "no",
        "target_g_range": (2.8, 3.8),
        "include_sensor": True,
    },
    {
        "name": "Upper Arm Deep Contusion & Hematoma",
        "injury_type": "bruise",
        "preset_img": os.path.join(SAMPLE_DIR, "web_verify", "upper_arm_bruise_wikimedia.jpg"),
        "location": "upper arm",
        "mechanism": "impact",
        "pain_range": (4, 6),
        "movement": "mild",
        "weight_bearing": "yes",
        "visible_bleeding": "no",
        "crack_pop": "no",
        "direct_impact": "yes",
        "deformity": "no",
        "target_g_range": (2.2, 3.4),
        "include_sensor": True,
    },
    {
        "name": "Minor Superficial Forearm Abrasion (Sensor-Free Triage)",
        "injury_type": "cut",
        "preset_img": os.path.join(SAMPLE_DIR, "web_verify", "roboflow_demo_bruise_or_abrasion.jpg"),
        "location": "forearm",
        "mechanism": "sharp",
        "pain_range": (1, 3),
        "movement": "none",
        "weight_bearing": "yes",
        "visible_bleeding": "mild",
        "crack_pop": "no",
        "direct_impact": "no",
        "deformity": "no",
        "include_sensor": False,  # Demonstrates optional sensor handling
    },
    {
        "name": "High-Velocity Bicycle Fall Trauma",
        "injury_type": "wound",
        "preset_img": None,  # Will use dynamic synthetic generator
        "location": "shoulder",
        "mechanism": "fall",
        "pain_range": (8, 10),
        "movement": "severe",
        "weight_bearing": "no",
        "visible_bleeding": "active",
        "crack_pop": "yes",
        "direct_impact": "yes",
        "deformity": "yes",
        "target_g_range": (6.5, 9.2),
        "include_sensor": True,
    },
    {
        "name": "Mild Household Blunt Impact (Low Risk)",
        "injury_type": "bruise",
        "preset_img": None,
        "location": "wrist",
        "mechanism": "impact",
        "pain_range": (2, 4),
        "movement": "mild",
        "weight_bearing": "yes",
        "visible_bleeding": "no",
        "crack_pop": "no",
        "direct_impact": "yes",
        "deformity": "no",
        "target_g_range": (1.4, 2.1),
        "include_sensor": True,
    }
]


def generate_random_demo_case_data(case_id: str, upload_dir: str) -> Dict[str, Any]:
    """
    Selects a randomized scenario, prepares images and sensor logs, and returns
    complete initialized case payload with diverse multimodal values.
    """
    from ml.vision.preprocess import verify_image_quality
    from ml.sensor.sensor_processor import process_sensor_data

    scenario = random.choice(SCENARIOS)
    
    # 1. Image preparation
    img_dest = os.path.join(upload_dir, f"{case_id}.jpg")
    use_preset = scenario.get("preset_img") and os.path.exists(scenario["preset_img"]) and (random.random() > 0.40)
    
    if use_preset:
        shutil.copyfile(scenario["preset_img"], img_dest)
    else:
        generate_synthetic_image(img_dest, scenario.get("injury_type", "wound"))
        
    quality_metrics = verify_image_quality(img_dest)

    # 2. Sensor preparation (optional or included based on scenario)
    sensor_summary = None
    sensor_available = False
    sensor_source_type = "not_provided"
    
    if scenario.get("include_sensor", True):
        g_min, g_max = scenario.get("target_g_range", (2.0, 5.0))
        target_peak_g = round(random.uniform(g_min, g_max), 2)
        
        sensor_dest = os.path.join(upload_dir, f"{case_id}_sensor_demo.csv")
        create_scaled_sensor_csv(sensor_dest, target_peak_g)
        
        try:
            sensor_summary = process_sensor_data(sensor_dest)
            sensor_summary["case_id"] = case_id
            sensor_summary["source_type"] = "demo"
            sensor_summary["data_provenance"] = "synthetic"
            sensor_summary["scenario"] = scenario["name"]
            sensor_summary["file_reference"] = sensor_dest
            sensor_summary["feature_version"] = "sensor-v1"
            sensor_summary["created_at"] = datetime.now(timezone.utc).isoformat()
            sensor_summary["data_provenance_message"] = f"Synthetic demonstration sensor log for '{scenario['name']}' ({sensor_summary.get('peak_g_force')}g measured)."
            sensor_available = True
            sensor_source_type = "demo"
        except Exception as e:
            print(f"Demo sensor processing warning: {e}")
            sensor_summary = None
            sensor_available = False

    # 3. Questionnaire preparation
    p_min, p_max = scenario.get("pain_range", (4, 7))
    pain_level = random.randint(p_min, p_max)
    
    questionnaire = {
        "answers": {
            "pain_level": pain_level,
            "injury_mechanism": scenario.get("mechanism", "sports"),
            "movement_limitation": scenario.get("movement", "moderate"),
            "weight_bearing": scenario.get("weight_bearing", "no"),
            "visible_bleeding": scenario.get("visible_bleeding", "no"),
            "crack_pop": scenario.get("crack_pop", "no"),
            "direct_impact": scenario.get("direct_impact", "yes"),
            "location": scenario.get("location", "injured limb"),
            "deformity": scenario.get("deformity", "no"),
            "swelling": "yes" if scenario.get("injury_type") == "swelling" or pain_level >= 5 else "no"
        },
        "scenario_name": scenario["name"],
        "voice_used": False,
        "voice_transcript": None
    }

    return {
        "scenario_name": scenario["name"],
        "image_reference": img_dest,
        "image_quality_metrics": quality_metrics,
        "sensor_summary": sensor_summary,
        "sensor_available": sensor_available,
        "sensor_source_type": sensor_source_type,
        "questionnaire": questionnaire,
    }
