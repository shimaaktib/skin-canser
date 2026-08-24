"""
config.py
General application configuration and Stage 2 (Malignant Subtype) model configuration.

Stage 2 preprocessing/model parameters below are extracted directly from the training
checkpoint (best_model_m.pth's saved `config` dict) and from
Multiclass_EfficientNetB3_Stage2_V1.ipynb (Section 9) - resolved at training time from
torchvision's official `EfficientNet_B3_Weights.IMAGENET1K_V1` transform metadata, never
hand-picked. Do not change these without re-checking the checkpoint.
"""
import os
import torch
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# --------------------------------------------------------------------------------------
# Dataset metadata (powers GET /api/dashboard) - HAM10000_metadata.csv columns:
# lesion_id, image_id, dx, dx_type, age, sex, localization
# --------------------------------------------------------------------------------------
DATASET_METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"

# --------------------------------------------------------------------------------------
# Frontend (optional - see app.py)
# Defaults to the sibling `frontend/` directory used by the local dev layout and the
# Dockerfile (see README.md). Override with the FRONTEND_DIR env var if you place the
# built frontend somewhere else on the same filesystem, or leave it missing entirely if
# this backend is deployed as a standalone API with the frontend hosted elsewhere.
# --------------------------------------------------------------------------------------
FRONTEND_DIR = Path(os.environ.get("FRONTEND_DIR", str(BASE_DIR.parent / "frontend")))

# --------------------------------------------------------------------------------------
# CORS
# Comma-separated list of browser origins allowed to call this API, e.g.
# "https://myapp.com,https://www.myapp.com". Defaults to "*" (any origin), which is fine
# for local development or a fully public research demo with no auth/cookies, but should
# be restricted to your real frontend origin(s) for a production deployment.
# --------------------------------------------------------------------------------------
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if _raw_origins == "*" else [o.strip() for o in _raw_origins.split(",") if o.strip()]

# --------------------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------------
# App metadata
# --------------------------------------------------------------------------------------
APP_TITLE = "شامة"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "A hierarchical AI system for skin lesion analysis. Stage 1 screens for malignancy; "
    "malignant lesions are automatically routed to Stage 2 for subtype classification."
)

# --------------------------------------------------------------------------------------
# Upload constraints
# --------------------------------------------------------------------------------------
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
MAX_UPLOAD_SIZE_MB = 10

# --------------------------------------------------------------------------------------
# Stage 2 - Malignant Subtype Classifier (EfficientNet-B3, 3-class)
# --------------------------------------------------------------------------------------
STAGE2_MODEL_PATH = MODELS_DIR / "best_model_m.pth"

STAGE2_RESIZE_SIZE = 320
STAGE2_IMG_SIZE = 300
STAGE2_NORM_MEAN = [0.485, 0.456, 0.406]
STAGE2_NORM_STD = [0.229, 0.224, 0.225]
STAGE2_INTERPOLATION = "bicubic"

# Class order fixed by training and re-verified against the checkpoint's own saved
# `class_order` at load time (see model.py) - mel=0, bcc=1, akiec=2.
STAGE2_CLASSES = ["mel", "bcc", "akiec"]
STAGE2_CLASS_FULL_NAMES = {
    "mel": "Melanoma",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis / Intraepithelial Carcinoma",
}

STAGE2_USE_TTA = True  # 3-view TTA: identity, horizontal flip, vertical flip
STAGE2_CALIBRATED = False  # per training notebook's explicit design decision - raw softmax, argmax

STAGE2_RECOMMENDATIONS = {
    "mel": (
        "The lesion shows features consistent with melanoma, the most serious form of skin "
        "cancer. Urgent evaluation by a dermatologist or oncologist is strongly recommended."
    ),
    "bcc": (
        "The lesion shows features consistent with basal cell carcinoma. This subtype is "
        "typically slow-growing and highly treatable when caught early - prompt dermatologist "
        "evaluation is recommended."
    ),
    "akiec": (
        "The lesion shows features consistent with actinic keratosis or early intraepithelial "
        "carcinoma. Dermatologist evaluation is recommended to determine the appropriate "
        "treatment."
    ),
}
