import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "ai_qtriage"
    port: int = 8000
    host: str = "0.0.0.0"

    sos_display_timezone: str = "Asia/Kolkata"

    # YOLO keep-threshold. Evidence: THRESHOLD_SWEEP_REPORT.json. Default 0.25,
    # not 0.10. Conservative profile is 0.30. Per-class env vars are optional.
    yolo_conf_threshold: float = 0.25
    yolo_low_confidence_flag: float = 0.40
    yolo_conf_threshold_cut: Optional[float] = None
    yolo_conf_threshold_bruise: Optional[float] = None
    yolo_conf_threshold_wound: Optional[float] = None

    # EfficientNet keep-gate (kaggle-v1 8-class head). Uniform/blank inputs are
    # withheld by quality gates; temperature + min confidence tune remaining softmax.
    effnet_min_confidence: float = 0.80
    effnet_temperature: float = 1.5

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
