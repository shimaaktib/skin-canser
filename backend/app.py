# app.py

"""
app.py
FastAPI application - Two-Stage Skin Cancer AI System (backend API).

Pipeline:
Every uploaded image goes through Stage 1 (binary) first.
If Stage 1 predicts Benign, that is the final result.
If Stage 1 predicts Malignant, Stage 2 (subtype) runs automatically
and its prediction becomes the final result.

Supabase integration:
- Uploads analyzed images to Supabase Storage.
- Saves analysis metadata/results in the `analyses` table.
- Provides History endpoints.
"""

import io
import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from config import (
    APP_TITLE,
    APP_VERSION,
    APP_DESCRIPTION,
    ALLOWED_CONTENT_TYPES,
    MAX_UPLOAD_SIZE_MB,
    DEVICE,
    DATASET_METADATA_PATH,
    ALLOWED_ORIGINS,
    FRONTEND_DIR,
)

from inference_binary import Stage1Predictor
from inference import Stage2Predictor
from dataset_stats import compute_dataset_stats
from interpretation import final_interpretation
from supabase_client import supabase, SUPABASE_BUCKET


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("app")


# --------------------------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------------------------

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

stage1_predictor: Optional[Stage1Predictor] = None
stage2_predictor: Optional[Stage2Predictor] = None

dataset_stats_cache: Optional[dict] = None


# --------------------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------------------

@app.on_event("startup")
def load_models() -> None:
    global stage1_predictor
    global stage2_predictor
    global dataset_stats_cache

    logger.info("Loading models on device: %s", DEVICE)

    try:
        stage1_predictor = Stage1Predictor(DEVICE)
    except Exception:
        stage1_predictor = None
        logger.exception("Stage 1 Binary model failed to load")

    try:
        stage2_predictor = Stage2Predictor(DEVICE)
    except Exception:
        stage2_predictor = None
        logger.exception("Stage 2 malignant model failed to load")

    if stage1_predictor is not None and stage2_predictor is not None:
        logger.info("Both stages loaded successfully. Ready to serve /predict.")
    else:
        logger.error("One or more model stages failed to load; affected requests will return a safe 503.")

    dataset_stats_cache = compute_dataset_stats(DATASET_METADATA_PATH)

    if dataset_stats_cache is None:
        logger.warning(
            "GET /api/dashboard will report 503 until HAM10000_metadata.csv is placed at %s.",
            DATASET_METADATA_PATH,
        )
    else:
        logger.info("Dataset stats ready. Serving /api/dashboard.")


# --------------------------------------------------------------------------------------
# Helper: check Supabase
# --------------------------------------------------------------------------------------

def require_supabase():
    """
    Make sure Supabase is configured before using database/storage features.
    """

    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Supabase is not configured. "
                "Please check SUPABASE_URL and "
                "SUPABASE_SERVICE_ROLE_KEY in the backend .env file."
            ),
        )

    return supabase


# --------------------------------------------------------------------------------------
# Helper: upload image to Supabase Storage
# --------------------------------------------------------------------------------------

def upload_image_to_supabase(
    raw_bytes: bytes,
    filename: str,
    content_type: str
) -> str:

    client = require_supabase()

    # Get file extension
    extension = Path(filename).suffix.lower()

    if not extension:
        extension = ".jpg"

    # Generate a unique filename.
    # This prevents two users from uploading files with the same name
    # and overwriting each other.
    unique_filename = (
        f"{uuid4().hex}{extension}"
    )

    image_path = unique_filename

    try:

        client.storage \
            .from_(SUPABASE_BUCKET) \
            .upload(
                path=image_path,
                file=raw_bytes,
                file_options={
                    "content-type": content_type,
                    "upsert": "false",
                },
            )

        logger.info(
            "Image uploaded to Supabase Storage: %s",
            image_path
        )

        return image_path

    except Exception as exc:

        logger.exception(
            "Failed to upload image to Supabase Storage: %s",
            exc
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to save image to Supabase Storage."
        )


# --------------------------------------------------------------------------------------
# Helper: create signed image URL
# --------------------------------------------------------------------------------------

def create_image_signed_url(
    image_path: Optional[str],
    expires_in: int = 3600
) -> Optional[str]:

    if not image_path:
        return None

    client = require_supabase()

    try:

        result = (
            client.storage
            .from_(SUPABASE_BUCKET)
            .create_signed_url(
                image_path,
                expires_in
            )
        )

        # Supabase Python client versions may return a dict
        # containing signedURL / signedUrl.
        if isinstance(result, dict):

            return (
                result.get("signedURL")
                or result.get("signedUrl")
                or result.get("signed_url")
            )

        return None

    except Exception as exc:

        logger.warning(
            "Could not create signed URL for %s: %s",
            image_path,
            exc
        )

        return None


# --------------------------------------------------------------------------------------
# Optional persistence
# --------------------------------------------------------------------------------------

def persist_analysis(
    raw_bytes: bytes,
    filename: str,
    content_type: str,
    result: str,
    confidence: float,
):
    """Persist an analysis when Supabase is configured; otherwise return a local-only result."""
    if supabase is None:
        logger.info("Supabase is not configured; returning inference result without persistence.")
        return None, None

    image_path = upload_image_to_supabase(
        raw_bytes=raw_bytes,
        filename=filename,
        content_type=content_type,
    )
    client = require_supabase()
    analysis_data = {
        "image_path": image_path,
        "image_name": filename,
        "result": result,
        "confidence": confidence,
    }

    try:
        db_response = client.table("analyses").insert(analysis_data).execute()
    except Exception as exc:
        logger.exception("Failed to save analysis to database: %s", exc)
        try:
            client.storage.from_(SUPABASE_BUCKET).remove([image_path])
        except Exception:
            logger.warning("Could not remove orphaned image: %s", image_path)
        raise HTTPException(status_code=500, detail="Failed to save analysis to database.")

    saved_analysis = db_response.data[0] if db_response.data else None
    return image_path, saved_analysis


# --------------------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------------------

@app.get("/health")
def health():

    models_loaded = {
        "stage1_binary": stage1_predictor is not None,
        "stage2_multiclass": stage2_predictor is not None,
    }

    return {
        "status": (
            "ok"
            if all(models_loaded.values())
            else "degraded"
        ),
        "device": str(DEVICE),
        "models_loaded": models_loaded,
        "supabase_configured": supabase is not None,
    }


# --------------------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------------------

@app.get("/api/dashboard")
def dashboard():

    """
    Dataset analytics computed from HAM10000_metadata.csv:
    totals, class/gender/age distributions,
    diagnosis type breakdown, and body location breakdown.
    """

    if dataset_stats_cache is None:

        raise HTTPException(
            status_code=503,
            detail=(
                f"Dataset metadata not available. "
                f"Place HAM10000_metadata.csv at "
                f"{DATASET_METADATA_PATH} and restart the server."
            ),
        )

    return dataset_stats_cache


# --------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if (
        stage1_predictor is None
        or stage2_predictor is None
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Models are still loading. "
                "Please try again shortly."
            ),
        )

    # ------------------------------------------------------------------
    # Validate file type
    # ------------------------------------------------------------------

    if file.content_type not in ALLOWED_CONTENT_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: "
                f"{file.content_type}"
            ),
        )

    # ------------------------------------------------------------------
    # Read image
    # ------------------------------------------------------------------

    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:

        raise HTTPException(
            status_code=400,
            detail=(
                f"File exceeds the "
                f"{MAX_UPLOAD_SIZE_MB}MB limit."
            ),
        )

    try:

        image = Image.open(
            io.BytesIO(raw_bytes)
        )

        image.load()

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file is not a valid image."
            ),
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read the uploaded image."
            ),
        )

    # ------------------------------------------------------------------
    # Stage 1: Binary screening
    # ------------------------------------------------------------------

    stage1_result = stage1_predictor.predict(image)

    # ------------------------------------------------------------------
    # BENIGN
    # ------------------------------------------------------------------

    if not stage1_result["is_malignant"]:

        final_prediction = stage1_result["prediction"]
        final_confidence = stage1_result["confidence"]

        image_path, saved_analysis = persist_analysis(
            raw_bytes=raw_bytes,
            filename=file.filename or "uploaded_image.jpg",
            content_type=file.content_type,
            result=final_prediction,
            confidence=final_confidence,
        )

        interpretation = final_interpretation(stage1_result["prediction"])

        return {
            "id": (
                saved_analysis.get("id")
                if saved_analysis
                else None
            ),

            "image_name": file.filename,

            "image_path": image_path,

            "stage1": {
                "prediction": stage1_result["prediction"],
                "confidence": stage1_result["confidence"],
                "probability_benign": stage1_result[
                    "probability_benign"
                ],
                "probability_malignant": stage1_result[
                    "probability_malignant"
                ],
                "raw_probability_malignant": stage1_result["raw_probability_malignant"],
                "threshold_used": stage1_result["threshold_used"],
                "calibration_method": stage1_result["calibration_method"],
                "tta_views": stage1_result["tta_views"],
            },

            "stage2": None,

            "final_prediction": final_prediction,

            "final_confidence": final_confidence,

            "interpretation": interpretation,

            "recommendation": interpretation["recommendation"],

            "created_at": (
                saved_analysis.get("created_at")
                if saved_analysis
                else None
            ),
        }

    # ------------------------------------------------------------------
    # MALIGNANT → Stage 2
    # ------------------------------------------------------------------

    stage2_result = stage2_predictor.predict(image)

    final_prediction = stage2_result["prediction"]
    final_confidence = stage2_result["confidence"]
    interpretation = final_interpretation(stage1_result["prediction"], stage2_result)

    image_path, saved_analysis = persist_analysis(
        raw_bytes=raw_bytes,
        filename=file.filename or "uploaded_image.jpg",
        content_type=file.content_type,
        result=final_prediction,
        confidence=final_confidence,
    )

    return {
        "id": (
            saved_analysis.get("id")
            if saved_analysis
            else None
        ),

        "image_name": file.filename,

        "image_path": image_path,

        "stage1": {
            "prediction": stage1_result["prediction"],
            "confidence": stage1_result["confidence"],
            "probability_benign": stage1_result["probability_benign"],
            "probability_malignant": stage1_result["probability_malignant"],
            "raw_probability_malignant": stage1_result["raw_probability_malignant"],
            "threshold_used": stage1_result["threshold_used"],
            "calibration_method": stage1_result["calibration_method"],
            "tta_views": stage1_result["tta_views"],
        },

        "stage2": {
            "prediction": stage2_result["prediction"],
            "prediction_code": stage2_result["prediction_code"],
            "confidence": stage2_result["confidence"],
            "probabilities": stage2_result["probabilities"],
            "routed_from_binary": True,
            "routing_message": "Stage 2 ran automatically after Stage 1 classified the lesion as malignant.",
        },

        "final_prediction": final_prediction,

        "final_confidence": final_confidence,

        "interpretation": interpretation,

        "recommendation": interpretation["recommendation"],

        "created_at": (
            saved_analysis.get("created_at")
            if saved_analysis
            else None
        ),
    }


# --------------------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------------------

@app.get("/history")
def get_history(
    page: int = 1,
    limit: int = 10
):

    """
    Return saved analyses ordered from newest to oldest.
    """

    if page < 1:

        raise HTTPException(
            status_code=400,
            detail="Page must be greater than or equal to 1."
        )

    if limit < 1 or limit > 100:

        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 100."
        )

    client = require_supabase()

    start = (page - 1) * limit
    end = start + limit - 1

    try:

        response = (
            client
            .table("analyses")
            .select("*")
            .order("created_at", desc=True)
            .range(start, end)
            .execute()
        )

    except Exception as exc:

        logger.exception(
            "Failed to retrieve history: %s",
            exc
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analysis history."
        )

    history = []

    for item in response.data:

        image_path = item.get("image_path")

        item["image_url"] = create_image_signed_url(
            image_path
        )

        history.append(item)

    return {
        "data": history,
        "page": page,
        "limit": limit,
        "count": len(history),
    }


# --------------------------------------------------------------------------------------
# Single History Item
# --------------------------------------------------------------------------------------

@app.get("/history/{analysis_id}")
def get_history_item(
    analysis_id: int
):

    """
    Return one analysis by ID.
    """

    client = require_supabase()

    try:

        response = (
            client
            .table("analyses")
            .select("*")
            .eq("id", analysis_id)
            .single()
            .execute()
        )

    except Exception as exc:

        logger.exception(
            "Failed to retrieve analysis %s: %s",
            analysis_id,
            exc
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analysis."
        )

    if not response.data:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found."
        )

    analysis = response.data

    analysis["image_url"] = create_image_signed_url(
        analysis.get("image_path")
    )

    return analysis


# --------------------------------------------------------------------------------------
# Delete History Item
# --------------------------------------------------------------------------------------

@app.delete("/history/{analysis_id}")
def delete_history_item(
    analysis_id: int
):

    """
    Delete one analysis from the database
    and its corresponding image from Supabase Storage.
    """

    client = require_supabase()

    # --------------------------------------------------------------
    # Find analysis first
    # --------------------------------------------------------------

    try:

        response = (
            client
            .table("analyses")
            .select("id,image_path")
            .eq("id", analysis_id)
            .single()
            .execute()
        )

    except Exception as exc:

        logger.exception(
            "Failed to find analysis %s: %s",
            analysis_id,
            exc
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to find analysis."
        )

    if not response.data:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found."
        )

    image_path = response.data.get(
        "image_path"
    )

    # --------------------------------------------------------------
    # Delete image from Storage
    # --------------------------------------------------------------

    if image_path:

        try:

            client.storage \
                .from_(SUPABASE_BUCKET) \
                .remove([image_path])

        except Exception as exc:

            logger.warning(
                "Failed to delete image %s from Storage: %s",
                image_path,
                exc
            )

    # --------------------------------------------------------------
    # Delete database record
    # --------------------------------------------------------------

    try:

        client \
            .table("analyses") \
            .delete() \
            .eq("id", analysis_id) \
            .execute()

    except Exception as exc:

        logger.exception(
            "Failed to delete analysis %s: %s",
            analysis_id,
            exc
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to delete analysis."
        )

    return {
        "success": True,
        "message": "Analysis deleted successfully.",
        "id": analysis_id,
    }


# --------------------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------------------

if FRONTEND_DIR.exists():

    @app.get(
        "/",
        response_class=FileResponse,
        include_in_schema=False
    )
    def serve_index():

        return FileResponse(
            FRONTEND_DIR / "index.html"
        )


    @app.get(
        "/dashboard",
        response_class=FileResponse,
        include_in_schema=False
    )
    def serve_dashboard_page():

        return FileResponse(
            FRONTEND_DIR / "dashboard.html"
        )


    app.mount(
        "/static",
        StaticFiles(
            directory=FRONTEND_DIR / "static"
        ),
        name="static"
    )

    logger.info(
        "Frontend found at %s - serving it alongside the API "
        "(single-service mode).",
        FRONTEND_DIR
    )

else:

    logger.info(
        "No frontend directory at %s - running in API-only mode.",
        FRONTEND_DIR
    )

