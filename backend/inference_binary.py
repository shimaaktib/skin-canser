"""Stage 1 inference: calibrated EfficientNet-B3 Binary screening with 8-view TTA."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config_binary import (
    STAGE1_CALIBRATION_METHOD,
    STAGE1_CLASSES,
    STAGE1_RECOMMENDATIONS,
    STAGE1_THRESHOLD,
    STAGE1_TTA_VIEWS,
)
from model_binary import load_binary_artifacts
from preprocessing_binary import preprocess_image

logger = logging.getLogger("inference_binary")


def _raw_prob_to_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _build_calibrator(checkpoint: dict[str, Any]) -> tuple[str, Callable[[float], float]]:
    method = str(checkpoint["calibration_method"])

    if method == "Isotonic Regression":
        x_values = np.asarray(checkpoint["calibrator_x_thresholds"], dtype=np.float64)
        y_values = np.asarray(checkpoint["calibrator_y_thresholds"], dtype=np.float64)

        def apply(probability: float) -> float:
            return float(np.interp(float(probability), x_values, y_values))

        return method, apply

    if method == "Temperature Scaling":
        temperature = float(checkpoint["calibrator_temperature"])

        def apply(probability: float) -> float:
            logit = _raw_prob_to_logit(np.asarray(probability, dtype=np.float64))
            return float(1.0 / (1.0 + np.exp(-logit / max(temperature, 1e-3))))

        return method, apply

    if method == "Platt":
        coefficient = float(checkpoint["calibrator_coef"][0][0])
        intercept = float(checkpoint["calibrator_intercept"][0])

        def apply(probability: float) -> float:
            raw = coefficient * float(probability) + intercept
            return float(1.0 / (1.0 + np.exp(-raw)))

        return method, apply

    raise ValueError(f"Unsupported Binary calibration method: {method}")


class Stage1Predictor:
    """Loads Stage 1 once and applies the checkpoint’s exact TTA/calibration/threshold flow."""

    def __init__(self, device: torch.device):
        self.device = device
        self.model, self.checkpoint = load_binary_artifacts(device)
        self.threshold = float(self.checkpoint["best_threshold"])
        self.calibration_method, self.apply_calibration = _build_calibrator(self.checkpoint)
        self.tta_views = int(self.checkpoint["config"].get("tta_views", STAGE1_TTA_VIEWS))

        if abs(self.threshold - STAGE1_THRESHOLD) > 1e-12:
            logger.warning(
                "Binary checkpoint threshold %.6f differs from config default %.6f; using checkpoint value.",
                self.threshold,
                STAGE1_THRESHOLD,
            )
        if self.calibration_method != STAGE1_CALIBRATION_METHOD:
            logger.warning(
                "Binary checkpoint calibration method %r differs from config default %r; using checkpoint value.",
                self.calibration_method,
                STAGE1_CALIBRATION_METHOD,
            )
        if self.tta_views != STAGE1_TTA_VIEWS:
            raise RuntimeError(
                f"Binary checkpoint requests {self.tta_views} TTA views, but deployment supports "
                f"{STAGE1_TTA_VIEWS}."
            )
        logger.info(
            "Stage 1 Binary model loaded on %s (calibration=%s, threshold=%.3f, TTA=%d)",
            device,
            self.calibration_method,
            self.threshold,
            self.tta_views,
        )

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        views = preprocess_image(image)
        batch = torch.stack(views).to(self.device)
        logits = self.model(batch)
        raw_probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        raw_probability_malignant = float(raw_probabilities.mean())
        probability_malignant = self.apply_calibration(raw_probability_malignant)
        probability_malignant = float(np.clip(probability_malignant, 0.0, 1.0))
        probability_benign = 1.0 - probability_malignant

        is_malignant = probability_malignant >= self.threshold
        label = STAGE1_CLASSES[1] if is_malignant else STAGE1_CLASSES[0]
        confidence = probability_malignant if is_malignant else probability_benign

        return {
            "prediction": label,
            "confidence": round(float(confidence), 4),
            "probability_benign": round(float(probability_benign), 4),
            "probability_malignant": round(float(probability_malignant), 4),
            "raw_probability_malignant": round(raw_probability_malignant, 4),
            "threshold_used": round(self.threshold, 4),
            "calibration_method": self.calibration_method,
            "tta_views": self.tta_views,
            "is_malignant": bool(is_malignant),
            "recommendation": STAGE1_RECOMMENDATIONS[label],
        }
