"""Stage 1 (Binary: Benign vs Malignant) deployment configuration.

The values in this module are derived from the supplied ``Skin_Lesion_Binary_FINAL_Ready``
notebook and the finalized ``best_model.pth`` checkpoint.  The checkpoint remains the source of
truth at load time; these constants provide stable defaults and make the API configuration
inspectable without loading model weights.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STAGE1_MODEL_PATH = MODELS_DIR / "best_model.pth"

STAGE1_RESIZE_SIZE = 320
STAGE1_IMG_SIZE = 300
STAGE1_NORM_MEAN = [0.485, 0.456, 0.406]
STAGE1_NORM_STD = [0.229, 0.224, 0.225]
STAGE1_INTERPOLATION = "bicubic"
STAGE1_CLASSES = ["Benign", "Malignant"]

# Frozen validation-only decision rule serialized in the supplied checkpoint.
STAGE1_THRESHOLD = 0.365
STAGE1_CALIBRATED = True
STAGE1_CALIBRATION_METHOD = "Isotonic Regression"
STAGE1_CHECKPOINT_FORMAT_VERSION = 2
STAGE1_USE_TTA = True
STAGE1_TTA_VIEWS = 8

# Kept as a compatibility export for callers that imported the old config mapping.  The
# authoritative mapping used by the API is centralized in interpretation.py.
STAGE1_RECOMMENDATIONS = {
    "Benign": (
        "The lesion is classified as benign, with no malignant features detected. Routine "
        "skin self-checks are recommended; consult a dermatologist if you notice changes in "
        "size, shape, color, or symptoms over time."
    ),
    "Malignant": (
        "The lesion shows features suggestive of malignancy. It has been automatically routed "
        "to subtype analysis below. Please consult a dermatologist promptly."
    ),
}
