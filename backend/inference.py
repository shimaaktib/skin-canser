"""
inference.py
Stage 2 inference: EfficientNet-B3 multiclass classifier with 3-view TTA and softmax + argmax,
matching Multiclass_EfficientNetB3_Stage2_V1.ipynb (Sections 13 & 15) exactly - no calibration
and no decision threshold, per the notebook's own explicit design decision (a 3-class softmax
output has no single scalar "positive class probability" to calibrate or threshold against).
"""
import logging

import torch
import torch.nn.functional as F
from PIL import Image

from config import STAGE2_CLASSES, STAGE2_CLASS_FULL_NAMES, STAGE2_RECOMMENDATIONS
from model import load_multiclass_model
from preprocessing import preprocess_image

logger = logging.getLogger("inference")


class Stage2Predictor:
    """Loads the Stage 2 model once at startup and reuses it for every request."""

    def __init__(self, device: torch.device):
        self.device = device
        self.model = load_multiclass_model(device)
        logger.info("Stage 2 (multiclass) model loaded on %s", device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        views = preprocess_image(image)
        batch = torch.stack(views).to(self.device)          # (3, C, H, W)
        logits = self.model(batch)                            # (3, n_classes)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        mean_probs = probs.mean(axis=0)                        # TTA average, (n_classes,)

        pred_idx = int(mean_probs.argmax())
        pred_class = STAGE2_CLASSES[pred_idx]
        pred_full_name = STAGE2_CLASS_FULL_NAMES[pred_class]

        probabilities = {cls: round(float(mean_probs[i]), 4) for i, cls in enumerate(STAGE2_CLASSES)}

        return {
            "prediction": pred_full_name,
            "prediction_code": pred_class,
            "confidence": round(float(mean_probs[pred_idx]), 4),
            "probabilities": probabilities,
            "recommendation": STAGE2_RECOMMENDATIONS[pred_class],
        }
