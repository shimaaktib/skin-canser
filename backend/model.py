"""
model.py
Stage 2 architecture - EfficientNet-B3 multiclass (3-way) classifier.

Extracted exactly from Multiclass_EfficientNetB3_Stage2_V1.ipynb (Section 9):
backbone = torchvision.models.efficientnet_b3, classifier head replaced with
Dropout(0.3) -> Linear(in_features, 3), forward() returns the full (B, 3) logit tensor
(no squeeze - this is a softmax target, not a sigmoid one). best_model_m.pth's `model_state`
was produced by this exact class - layer names and shapes must match exactly.
"""
import torch
import torch.nn as nn
import torchvision

from config import STAGE2_MODEL_PATH, STAGE2_CLASSES


class EfficientNetB3MulticlassClassifier(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        # weights=None: fine-tuned weights are loaded from the checkpoint right after, so no
        # ImageNet-pretrained download is needed at inference/server startup time.
        self.backbone = torchvision.models.efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, n_classes),
        )

    def forward(self, x):
        return self.backbone(x)  # (B, n_classes) logits


def load_multiclass_model(device: torch.device) -> EfficientNetB3MulticlassClassifier:
    """Instantiates the Stage 2 architecture and loads best_model_m.pth's weights onto it."""
    if not STAGE2_MODEL_PATH.exists():
        raise FileNotFoundError(f"Stage 2 checkpoint not found at {STAGE2_MODEL_PATH}")

    checkpoint = torch.load(STAGE2_MODEL_PATH, map_location=device, weights_only=False)

    # The checkpoint carries its own `class_order` (mel/bcc/akiec) - refuse to load rather than
    # silently risk a mismatched label mapping if config.py and the checkpoint ever disagree.
    saved_class_order = checkpoint.get("class_order", STAGE2_CLASSES)
    if list(saved_class_order) != list(STAGE2_CLASSES):
        raise RuntimeError(
            f"Checkpoint class_order {saved_class_order} does not match config.STAGE2_CLASSES "
            f"{STAGE2_CLASSES}. Refusing to load with a possibly mismatched label mapping."
        )

    model = EfficientNetB3MulticlassClassifier(n_classes=len(STAGE2_CLASSES))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model
