import os
import json
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import timm

from ml.vision.input_quality import (
    STATUS_OOD,
    STATUS_UNAVAILABLE,
    STATUS_VALID,
    assess_input_quality,
    resolve_min_confidence,
    resolve_temperature,
    softmax_entropy,
)


class EfficientNetV2Classifier:
    """Wrapper for EfficientNetV2 visible injury classifier."""

    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Placeholder until load_model() replaces with checkpoint taxonomy.
        # Active kaggle-v1 canonical head is 8-class (no swelling).
        self.classes = [
            "abrasion",
            "bruise",
            "burn",
            "cut",
            "laceration",
            "wound",
            "normal",
            "ood_reject",
        ]
        self.model_path = model_path
        self.min_confidence = resolve_min_confidence()
        self.temperature = resolve_temperature()

        self.model = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k",
            pretrained=False,
            num_classes=len(self.classes),
        )
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = False

        from ml.models.canonical_paths import EFFNET_CANONICAL, resolve_existing

        if model_path:
            located = resolve_existing(model_path)
            if os.path.exists(located):
                self.load_model(located)
            return

        canonical = resolve_existing(EFFNET_CANONICAL)
        if os.path.exists(canonical):
            self.load_model(canonical)

    def load_model(self, model_path: str):
        """Loads state dictionary weights. Rebuilds head to match checkpoint class count."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classifier weights not found at: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        n_classes = None
        for key, tensor in state_dict.items():
            if str(key).endswith("classifier.weight"):
                n_classes = int(tensor.shape[0])
                break
        if n_classes is None:
            n_classes = len(self.classes)
        self.classes = self._resolve_class_names(model_path, n_classes)
        self.model = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k",
            pretrained=False,
            num_classes=n_classes,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        self.model_path = model_path
        print(f"EfficientNetV2 classifier successfully loaded from {model_path} classes={self.classes}")

    def _resolve_class_names(self, model_path: str, n_classes: int) -> list:
        candidates = [
            os.path.splitext(model_path)[0] + "_classes.json",
            os.path.join(os.path.dirname(model_path), "efficientnet_classes.json"),
            os.path.join("ml", "models", "vision", "efficientnetv2_injury_best_classes.json"),
        ]
        from ml.models.canonical_paths import exists, resolve_existing
        for path in candidates:
            located = resolve_existing(path)
            if not exists(located):
                continue
            path = located
            try:
                with open(path, encoding="utf-8") as handle:
                    names = json.load(handle)
                if isinstance(names, list) and len(names) == n_classes:
                    return [str(n).lower() for n in names]
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        if n_classes == 3:
            return ["cut", "bruise", "other"]
        if n_classes == 4:
            return ["cut", "bruise", "normal", "ood_reject"]
        if n_classes == 8:
            return [
                "abrasion",
                "bruise",
                "burn",
                "cut",
                "laceration",
                "wound",
                "normal",
                "ood_reject",
            ]
        return [f"class_{i}" for i in range(n_classes)]

    def _empty_class_dict(self) -> dict:
        return {c.capitalize(): None for c in self.classes}

    def _withheld(self, status: str, reason: str, quality: dict = None) -> dict:
        result = self._empty_class_dict()
        result["__is_confident"] = False
        result["__max_prob"] = None
        result["__winner"] = None
        result["__low_confidence"] = True
        result["__status"] = status
        result["__reason"] = reason
        result["__entropy"] = None
        result["__margin"] = None
        result["__temperature"] = self.temperature
        result["__min_confidence"] = self.min_confidence
        result["__quality"] = (quality or {}).get("metrics", {})
        return result

    def _forward_logits(self, image_rgb: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(image_rgb, (224, 224))
        if img_resized.dtype != np.uint8:
            if np.issubdtype(img_resized.dtype, np.floating) and float(np.nanmax(img_resized)) <= 1.5:
                img_resized = np.clip(img_resized * 255.0, 0, 255).astype(np.uint8)
            else:
                img_resized = np.clip(img_resized, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
        return outputs.squeeze(0).cpu()

    def predict_raw(self, image_rgb: np.ndarray, temperature: float = 1.0) -> dict:
        """Softmax without quality gates. For audits only — do not use as a user-facing prediction."""
        if not self.is_loaded:
            return self._withheld(STATUS_UNAVAILABLE, "classifier_weights_not_loaded")
        logits = self._forward_logits(image_rgb)
        t = max(float(temperature), 1e-3)
        probabilities = F.softmax(logits / t, dim=0).numpy()
        winner_idx = int(probabilities.argmax())
        ordered = np.sort(probabilities)
        return {
            "probs": {self.classes[i].capitalize(): float(probabilities[i]) for i in range(len(self.classes))},
            "max_prob": float(probabilities.max()),
            "winner": self.classes[winner_idx].capitalize(),
            "entropy": softmax_entropy(probabilities),
            "margin": float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else float(ordered[-1]),
            "temperature": t,
        }

    def predict(self, image_rgb: np.ndarray, min_confidence: float = None) -> dict:
        """
        Runs prediction on the image with input-quality gates and confidence rejection.

        Status values:
          VALID, LOW_QUALITY_INPUT, OUT_OF_DISTRIBUTION, MODEL_UNAVAILABLE

        Invalid/uniform/OOD inputs do not return a confident injury class.
        Softmax on those inputs is withheld (class values are None).
        """
        if not self.is_loaded:
            return self._withheld(STATUS_UNAVAILABLE, "classifier_weights_not_loaded")

        quality = assess_input_quality(image_rgb)
        if quality["status"] != STATUS_VALID:
            return self._withheld(quality["status"], quality["reason"], quality)

        min_conf = resolve_min_confidence(min_confidence)
        logits = self._forward_logits(image_rgb)
        probabilities = F.softmax(logits / self.temperature, dim=0).numpy()
        max_prob = float(probabilities.max())
        winner_idx = int(probabilities.argmax())
        winner_raw = str(self.classes[winner_idx])
        # Display names: keep ood_reject readable; title-case others.
        winner = "OOD_Reject" if winner_raw.lower() == "ood_reject" else winner_raw.replace("_", " ").title()
        entropy = softmax_entropy(probabilities)
        ordered = np.sort(probabilities)
        margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else float(ordered[-1])

        # Closed-set head is overconfident. Require both score and margin.
        entropy_cap = float(np.log(len(self.classes))) * 0.55
        score_ok = (
            max_prob >= min_conf
            and margin >= 0.20
            and entropy <= entropy_cap
        )
        # normal / ood_reject are abstention classes — never injury findings.
        non_injury = winner_raw.lower() in {"normal", "ood_reject", "other"}
        is_confident = bool(score_ok) and not non_injury

        def _label(name: str) -> str:
            return "OOD_Reject" if str(name).lower() == "ood_reject" else str(name).replace("_", " ").title()

        result = {
            _label(self.classes[i]): float(probabilities[i]) if score_ok else None
            for i in range(len(self.classes))
        }
        result["__is_confident"] = bool(is_confident)
        result["__max_prob"] = max_prob
        result["__winner"] = winner if is_confident else (winner if score_ok and non_injury else None)
        result["__is_injury_finding"] = bool(is_confident)
        result["__abstention_class"] = winner if score_ok and non_injury else None
        result["__low_confidence"] = not is_confident
        if score_ok and non_injury:
            result["__status"] = STATUS_VALID
            result["__reason"] = f"abstention_class_{winner_raw.lower()}"
        elif is_confident:
            result["__status"] = STATUS_VALID
            result["__reason"] = "passed_quality_and_confidence_gates"
        else:
            result["__status"] = STATUS_OOD
            result["__reason"] = "softmax_not_trusted_closed_set_overconfidence"
        result["__entropy"] = round(entropy, 6)
        result["__margin"] = round(margin, 6)
        result["__temperature"] = self.temperature
        result["__min_confidence"] = min_conf
        result["__quality"] = quality.get("metrics", {})
        result["__raw_winner"] = winner
        result["__raw_max_prob"] = max_prob
        return result


def interpret_prediction(probs: dict) -> dict:
    """Normalize wrapper output for API callers. Never promotes a non-VALID winner.

    Abstention classes (Normal / OOD_Reject) may be VALID status but are never
    injury findings (is_confident / is_injury_finding stay False).
    """
    status = probs.get("__status") or STATUS_UNAVAILABLE
    injury = bool(probs.get("__is_confident")) and status == STATUS_VALID
    abstention = probs.get("__abstention_class") if status == STATUS_VALID else None
    winner = probs.get("__winner") if injury else None
    max_prob = probs.get("__max_prob")
    return {
        "status": status,
        "reason": probs.get("__reason"),
        "winner": winner,
        "max_prob": float(max_prob) if injury and isinstance(max_prob, (int, float)) else (
            float(max_prob) if abstention and isinstance(max_prob, (int, float)) else None
        ),
        "is_confident": injury,
        "is_injury_finding": injury,
        "abstention_class": abstention,
        "low_confidence": not injury,
        "class_probs": {k: v for k, v in probs.items() if not str(k).startswith("__")},
    }
