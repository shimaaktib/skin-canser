"""Stage 1 Binary EfficientNet-B3 architecture and checkpoint loader."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision

from config_binary import (
    STAGE1_CLASSES,
    STAGE1_CHECKPOINT_FORMAT_VERSION,
    STAGE1_MODEL_PATH,
)


class EfficientNetB3BinaryClassifier(nn.Module):
    """The exact architecture used by Skin_Lesion_Binary_FINAL_Ready.ipynb."""

    def __init__(self) -> None:
        super().__init__()
        full_model = torchvision.models.efficientnet_b3(weights=None)
        self.features = full_model.features
        self.avgpool = full_model.avgpool
        in_features = full_model.classifier[1].in_features
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x).squeeze(1)


def _validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    required = {"model_state", "config", "best_threshold", "calibration_method"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"Binary checkpoint is missing required keys: {missing}")

    class_names = checkpoint.get("class_names", STAGE1_CLASSES)
    if list(class_names) != list(STAGE1_CLASSES):
        raise RuntimeError(
            f"Binary checkpoint class_names {class_names} do not match {STAGE1_CLASSES}."
        )

    if checkpoint.get("checkpoint_format_version") != STAGE1_CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            "Unsupported Binary checkpoint format: "
            f"{checkpoint.get('checkpoint_format_version')!r}; expected "
            f"{STAGE1_CHECKPOINT_FORMAT_VERSION}."
        )

    threshold = float(checkpoint["best_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Binary checkpoint threshold must be in (0, 1), got {threshold}.")

    method = str(checkpoint["calibration_method"])
    if method == "Isotonic Regression":
        x_values = checkpoint.get("calibrator_x_thresholds")
        y_values = checkpoint.get("calibrator_y_thresholds")
        if not isinstance(x_values, list) or not isinstance(y_values, list):
            raise KeyError("Isotonic calibration arrays are missing from the Binary checkpoint.")
        if len(x_values) < 2 or len(x_values) != len(y_values):
            raise ValueError("Serialized Isotonic calibration arrays are malformed.")
        if any(float(right) <= float(left) for left, right in zip(x_values, x_values[1:])):
            raise ValueError("Serialized Isotonic x-thresholds must be strictly increasing.")
    elif method == "Temperature Scaling":
        temperature = float(checkpoint.get("calibrator_temperature", 0.0))
        if temperature <= 0:
            raise ValueError("Serialized temperature must be greater than zero.")
    elif method == "Platt":
        if "calibrator_coef" not in checkpoint or "calibrator_intercept" not in checkpoint:
            raise KeyError("Serialized Platt calibration parameters are missing.")
    else:
        raise ValueError(f"Unsupported Binary calibration method: {method}")


def load_binary_artifacts(
    device: torch.device,
    checkpoint_path=STAGE1_MODEL_PATH,
) -> tuple[EfficientNetB3BinaryClassifier, dict[str, Any]]:
    """Load the Binary model and its serialized calibration/threshold artifacts."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Binary checkpoint must contain a dictionary.")
    _validate_checkpoint(checkpoint)

    model = EfficientNetB3BinaryClassifier()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def load_binary_model(device: torch.device) -> EfficientNetB3BinaryClassifier:
    """Backward-compatible model-only loader."""
    model, _ = load_binary_artifacts(device)
    return model
