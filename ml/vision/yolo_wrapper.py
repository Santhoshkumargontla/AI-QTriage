import hashlib
import os

from ultralytics import YOLO

from ml.models.canonical_paths import YOLO_CANONICAL, YOLO_RUNTIME, resolve_existing

UNTRAINED_CLASS = "UNTRAINED_CLASS"
UNSUPPORTED_CLASS = "UNSUPPORTED"

# Evidence: ml/models/yolo_threshold_eval/THRESHOLD_SWEEP_REPORT.json
# Val/test recall is unchanged from 0.10 through 0.50; 0.10 only adds FPs.
# Do not default to 0.10. 0.25 is the research-demo default; 0.30 is conservative.
DEFAULT_YOLO_INFER_CONF = 0.25
DEFAULT_YOLO_LOW_CONF_FLAG = 0.40
CONSERVATIVE_YOLO_INFER_CONF = 0.30
YOLO_CLASS_SUPPORT_SIDECAR = os.path.join("ml", "models", "vision", "yolo11_class_support.json")
YOLO_METADATA_SIDECAR = os.path.join("ml", "models", "vision", "yolo11_metadata.json")
ENV_YOLO_INFER_CONF = "YOLO_CONF_THRESHOLD"
ENV_YOLO_LOW_CONF_FLAG = "YOLO_LOW_CONFIDENCE_FLAG"
ENV_CLASS_THRESHOLDS = {
    "cut": "YOLO_CONF_THRESHOLD_CUT",
    "bruise": "YOLO_CONF_THRESHOLD_BRUISE",
    "wound": "YOLO_CONF_THRESHOLD_WOUND",
}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yolo_metadata() -> dict:
    located = resolve_existing(YOLO_METADATA_SIDECAR)
    if not located or not os.path.exists(located):
        return {}
    try:
        import json
        with open(located, encoding="utf-8") as handle:
            return json.load(handle) or {}
    except (OSError, ValueError):
        return {}


def load_yolo_class_support() -> dict:
    """Honest per-class training support from sidecar (not inferred from aggregate mAP)."""
    located = resolve_existing(YOLO_CLASS_SUPPORT_SIDECAR)
    if not located or not os.path.exists(located):
        return {}
    try:
        import json
        with open(located, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _names_in_index_order(names) -> list:
    if not names:
        return []
    if isinstance(names, dict):
        return [str(names[key]).lower() for key in sorted(names.keys())]
    return [str(name).lower() for name in names]


def _validate_conf(name: str, value: float) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {numeric}")
    return numeric


def _read_float_env(name: str):
    raw = os.environ.get(name)
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    return _validate_conf(name, float(text))


def _settings_values():
    try:
        from backend.config import Settings
        return Settings()
    except ImportError:
        return None


def resolve_yolo_infer_conf(explicit=None) -> float:
    """Global box-keep threshold. Env YOLO_CONF_THRESHOLD overrides the default."""
    if explicit is not None:
        return _validate_conf("infer_conf", explicit)
    env = _read_float_env(ENV_YOLO_INFER_CONF)
    if env is not None:
        return env
    settings = _settings_values()
    if settings is not None and settings.yolo_conf_threshold is not None:
        return _validate_conf(ENV_YOLO_INFER_CONF, settings.yolo_conf_threshold)
    return DEFAULT_YOLO_INFER_CONF


def resolve_yolo_low_conf_flag(explicit=None) -> float:
    """Display flag only. Does not drop boxes. Env YOLO_LOW_CONFIDENCE_FLAG."""
    if explicit is not None:
        return _validate_conf("low_confidence_flag", explicit)
    env = _read_float_env(ENV_YOLO_LOW_CONF_FLAG)
    if env is not None:
        return env
    settings = _settings_values()
    if settings is not None and settings.yolo_low_confidence_flag is not None:
        return _validate_conf(ENV_YOLO_LOW_CONF_FLAG, settings.yolo_low_confidence_flag)
    return DEFAULT_YOLO_LOW_CONF_FLAG


def resolve_yolo_class_thresholds() -> dict:
    """Optional per-class keep thresholds. Unset classes use the global infer conf."""
    out = {}
    settings = _settings_values()
    settings_map = {}
    if settings is not None:
        settings_map = {
            "cut": settings.yolo_conf_threshold_cut,
            "bruise": settings.yolo_conf_threshold_bruise,
            "wound": settings.yolo_conf_threshold_wound,
        }
    for class_name, env_name in ENV_CLASS_THRESHOLDS.items():
        value = _read_float_env(env_name)
        if value is None:
            raw = settings_map.get(class_name)
            if raw is None:
                continue
            value = _validate_conf(env_name, raw)
        out[class_name] = value
    return out


class YOLO11Detector:
    """Runtime YOLO11 detector. Loads YOLO_CANONICAL only unless an explicit path is passed."""

    def __init__(self, model_path: str = None, conf_threshold: float = None, infer_conf: float = None):
        self.model = None
        self.infer_conf = resolve_yolo_infer_conf(infer_conf)
        # conf_threshold is the low-confidence display flag, not the keep threshold.
        self.conf_threshold = resolve_yolo_low_conf_flag(conf_threshold)
        self.class_thresholds = resolve_yolo_class_thresholds()
        self.status = "MODEL_ARTIFACT_MISSING"
        self.is_trained = False
        self.model_name = "YOLO11 Detection"
        self.model_version = "expanded-skin-v1"
        self.training_data = "canonical_yolo11_checkpoint"
        self.supported_classes = set()
        self.class_list = []
        self.clinically_validated = False
        self.untrained_classes = {}
        self.class_support = {}
        self.artifact_sha256 = None
        self.task = None

        if model_path is None:
            model_path = YOLO_RUNTIME
        self.model_path = model_path
        load_path = resolve_existing(model_path)

        if not load_path or not os.path.exists(load_path):
            self.status = "MODEL_ARTIFACT_MISSING"
            return

        try:
            self.load_model(load_path)
            self.model_path = model_path
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Error loading YOLO weights from {load_path}: {exc}")
            self.status = "MODEL_ARTIFACT_MISSING"
            self.model = None

    def _sync_classes_from_model(self):
        if self.model is None:
            self.supported_classes = set()
            self.class_list = []
            self.untrained_classes = {}
            self.class_support = {}
            return
        self.class_list = _names_in_index_order(self.model.names)
        self.supported_classes = set(self.class_list)
        support_doc = load_yolo_class_support()
        classes_meta = (support_doc.get("classes") or {}) if support_doc else {}
        self.class_support = {}
        self.untrained_classes = {}
        for name in self.class_list:
            meta = classes_meta.get(name) or {}
            status = str(meta.get("status") or "UNKNOWN")
            train_n = int(meta.get("train_boxes") or 0)
            row = {
                "status": status,
                "train_boxes": train_n,
                "val_boxes": int(meta.get("val_boxes") or 0),
                "test_boxes": int(meta.get("test_boxes") or 0),
                "note": meta.get("note"),
                "in_model_names": True,
            }
            self.class_support[name] = row
            # Architecture may list the class, but zero honest labels => unsupported.
            if status == UNSUPPORTED_CLASS or train_n <= 0:
                self.untrained_classes[name] = UNSUPPORTED_CLASS
        # Classes advertised elsewhere but absent from model.names
        for absent in ("fracture", "swelling", "Normal", "OOD_Reject"):
            if absent not in self.supported_classes:
                self.untrained_classes[absent] = UNTRAINED_CLASS
        # Older 3-class checkpoints may lack burn/wound/laceration
        for maybe in ("burn", "wound", "laceration", "abrasion"):
            if maybe not in self.supported_classes:
                self.untrained_classes[maybe] = UNTRAINED_CLASS

    def class_status(self, class_name: str) -> str:
        if class_name is None:
            return UNTRAINED_CLASS
        name = str(class_name).lower().strip()
        support = (self.class_support or {}).get(name) or {}
        if support.get("status") == UNSUPPORTED_CLASS or int(support.get("train_boxes") or 0) <= 0:
            if name in self.supported_classes:
                return UNSUPPORTED_CLASS
        if name in self.supported_classes:
            return name
        return UNTRAINED_CLASS

    def load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MODEL_ARTIFACT_MISSING: {model_path}")
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.artifact_sha256 = _sha256_file(model_path)
        self._sync_classes_from_model()
        meta = load_yolo_metadata()
        if meta.get("version"):
            self.model_version = str(meta["version"])
        if meta.get("metrics", {}).get("dataset_name") or meta.get("training_dataset"):
            self.training_data = str(
                (meta.get("metrics") or {}).get("dataset_name")
                or meta.get("training_dataset")
            )
        task = getattr(self.model, "task", None)
        self.task = task
        if isinstance(task, str) and task != "detect":
            self.status = "NOT_TRUSTWORTHY"
            self.is_trained = False
            raise RuntimeError(f"YOLO checkpoint task is '{task}', expected 'detect'")
        self.status = "INFERENCE_EXECUTES"
        self.is_trained = True
        print(
            f"YOLO11 loaded from {model_path} names={self.class_list} "
            f"task={task} sha256={self.artifact_sha256}"
        )

    def get_info(self) -> dict:
        names = {}
        task = self.task
        ckpt = self.model_path
        if self.model is not None:
            names = dict(self.model.names) if isinstance(self.model.names, dict) else list(self.model.names)
            task = getattr(self.model, "task", None)
            ckpt = getattr(self.model, "ckpt_path", None) or self.model_path
        return {
            "model_name": self.model_name,
            "active_model": self.model_name,
            "model_version": self.model_version,
            "version": self.model_version,
            "model_path": self.model_path,
            "canonical_path": YOLO_CANONICAL,
            "ckpt_path": str(ckpt) if ckpt else None,
            "task": task,
            "artifact_sha256": self.artifact_sha256,
            "model_names": names,
            "classes": list(self.class_list),
            "training_data": self.training_data,
            "supported_classes": list(self.class_list),
            "class_support": dict(getattr(self, "class_support", {}) or {}),
            "validated_classes": [
                name
                for name, meta in (getattr(self, "class_support", {}) or {}).items()
                if str(meta.get("status")) not in {UNSUPPORTED_CLASS, "UNKNOWN"}
                and int(meta.get("train_boxes") or 0) > 0
            ],
            "unsupported_classes": [
                name
                for name, meta in (getattr(self, "class_support", {}) or {}).items()
                if str(meta.get("status")) == UNSUPPORTED_CLASS or int(meta.get("train_boxes") or 0) <= 0
            ],
            "untrained_classes": dict(self.untrained_classes),
            "unsupported_class_status": UNTRAINED_CLASS,
            "promotion_status": (load_yolo_class_support() or {}).get("promotion_status"),
            "dataset_provenance": (load_yolo_class_support() or {}).get("dataset_provenance"),
            "status": self.status,
            "is_trained": self.is_trained,
            "clinically_validated": self.clinically_validated,
            "infer_conf": self.infer_conf,
            "low_confidence_flag": self.conf_threshold,
            "class_thresholds": dict(self.class_thresholds),
            "threshold_env": ENV_YOLO_INFER_CONF,
            "recommended_research_demo_threshold": DEFAULT_YOLO_INFER_CONF,
            "recommended_conservative_threshold": CONSERVATIVE_YOLO_INFER_CONF,
        }

    def _class_keep_threshold(self, class_name: str, global_conf: float) -> float:
        if class_name in self.class_thresholds:
            return self.class_thresholds[class_name]
        return global_conf

    def detect(self, image_path: str, conf: float = None) -> list:
        if self.model is None:
            raise RuntimeError(f"MODEL_ARTIFACT_MISSING: {self.model_path or YOLO_CANONICAL}")

        global_conf = resolve_yolo_infer_conf(conf) if conf is not None else self.infer_conf
        run_conf = global_conf
        if self.class_thresholds:
            run_conf = min([global_conf, *self.class_thresholds.values()])

        results = self.model(image_path, conf=run_conf, verbose=False)
        findings = []
        if len(results) == 0:
            return findings

        result = results[0]
        boxes = result.boxes
        orig_shape = getattr(result, "orig_shape", None)
        if orig_shape is not None and len(orig_shape) == 2:
            orig_h, orig_w = int(orig_shape[0]), int(orig_shape[1])
        else:
            orig_h, orig_w = 640, 640

        for box in boxes:
            class_id = int(box.cls[0].item())
            raw_name = self.model.names.get(class_id, str(class_id)) if isinstance(self.model.names, dict) else str(class_id)
            name = str(raw_name).lower()
            if name not in self.supported_classes:
                continue
            conf_v = float(box.conf[0].item())
            keep_thr = self._class_keep_threshold(name, global_conf)
            if conf_v < keep_thr:
                continue
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            # Clamp to original image bounds (defensive — Ultralytics already uses orig space).
            x1 = float(max(0.0, min(float(xyxy[0]), float(orig_w - 1))))
            y1 = float(max(0.0, min(float(xyxy[1]), float(orig_h - 1))))
            x2 = float(max(0.0, min(float(xyxy[2]), float(orig_w))))
            y2 = float(max(0.0, min(float(xyxy[3]), float(orig_h))))
            if x2 <= x1 or y2 <= y1:
                continue
            findings.append({
                "finding": name,
                "confidence": round(conf_v, 4),
                "bounding_box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "low_confidence": conf_v < self.conf_threshold,
                "keep_threshold": keep_thr,
                "class_support_status": self.class_status(name),
                "image_width": orig_w,
                "image_height": orig_h,
                "coordinate_space": "original_image_xyxy",
                "model_version": self.model_version,
                "model_status": self.status,
                "model_path": self.model_path,
                "artifact_sha256": self.artifact_sha256,
            })
        return findings
