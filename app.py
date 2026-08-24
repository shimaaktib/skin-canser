from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# The existing inference modules live under backend/. Add that directory explicitly so this
# root-level Streamlit entrypoint works locally and on Streamlit Community Cloud.
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import torch
    from PIL import Image, UnidentifiedImageError

    from inference import Stage2Predictor
    from inference_binary import Stage1Predictor
    from interpretation import final_interpretation
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in broken deployments
    missing_name = exc.name or "a required package"
    st.error(
        f"The deployment is missing a required dependency: {missing_name}. "
        "Install the dependencies from requirements.txt and restart the app."
    )
    st.stop()


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("streamlit_app")

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_predictors() -> tuple[Stage1Predictor, Stage2Predictor]:
    """Load both frozen predictors once per Streamlit process."""
    if not (BACKEND_DIR / "models" / "best_model.pth").exists():
        raise FileNotFoundError(
            "The Stage 1 checkpoint is missing: backend/models/best_model.pth"
        )
    if not (BACKEND_DIR / "models" / "best_model_m.pth").exists():
        raise FileNotFoundError(
            "The Stage 2 checkpoint is missing: backend/models/best_model_m.pth"
        )

    stage1 = Stage1Predictor(DEVICE)
    stage2 = Stage2Predictor(DEVICE)
    return stage1, stage2


def read_uploaded_image(uploaded_file: Any) -> Image.Image:
    """Validate and decode an uploaded image without exposing internal tracebacks."""
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError("The image exceeds the 10 MB upload limit.")

    suffix = Path(uploaded_file.name or "").suffix.lower().lstrip(".")
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported image format. Use JPG, JPEG, PNG, or WEBP.")

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid readable image.") from exc
    except Exception as exc:
        raise ValueError("The uploaded image could not be decoded safely.") from exc

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("The image dimensions are too large for safe processing.")

    return image.convert("RGB")


def show_prediction(result: dict[str, Any]) -> None:
    """Render the hierarchical prediction and preserved interpretation layer."""
    stage1 = result["stage1"]
    stage2 = result.get("stage2")
    interpretation = result["interpretation"]

    st.subheader("Screening result")
    result_col, confidence_col = st.columns(2)
    with result_col:
        st.metric("Final prediction", result["final_prediction"])
    with confidence_col:
        st.metric("Final confidence", f"{float(result['final_confidence']) * 100:.1f}%")

    st.write(
        f"**Stage 1 binary screen:** {stage1['prediction']} "
        f"({float(stage1['confidence']) * 100:.1f}% confidence)"
    )
    st.caption(
        f"Stage 1 uses {stage1['tta_views']}-view test-time augmentation, "
        f"{stage1['calibration_method']} calibration, and threshold "
        f"{stage1['threshold_used']:.3f}."
    )

    if stage2 is not None:
        st.write(
            f"**Stage 2 malignant subtype:** {stage2['prediction']} "
            f"({float(stage2['confidence']) * 100:.1f}% confidence)"
        )
        st.caption("Stage 2 was run automatically because Stage 1 classified the image as malignant.")

        probability_rows = [
            {"Subtype": code, "Probability": f"{float(probability) * 100:.1f}%"}
            for code, probability in stage2["probabilities"].items()
        ]
        st.table(probability_rows)

    st.subheader("Interpretation")
    interpretation_col, severity_col = st.columns(2)
    with interpretation_col:
        st.write(f"**{interpretation['grade']}**")
        st.write(interpretation["summary"])
    with severity_col:
        st.write(f"**Severity:** {interpretation['severity']}")
        st.write(f"**Level:** {interpretation['severity_level']}")

    st.info(interpretation["recommendation"])


def main() -> None:
    st.set_page_config(
        page_title="Dermoscopic AI — Skin Lesion Analysis",
        page_icon=":material/health_and_safety:",
        layout="centered",
    )

    st.title("Dermoscopic AI — Skin Lesion Analysis")
    st.write("AI-assisted skin lesion screening")
    st.warning(
        "Research/demo use only. This application provides an AI-generated screening estimate, "
        "not a medical diagnosis. A qualified clinician must confirm any consequential finding."
    )

    uploaded_file = st.file_uploader(
        "Upload a skin-lesion image",
        type=sorted(ALLOWED_EXTENSIONS),
        help="Accepted formats: JPG, JPEG, PNG, and WEBP. Maximum size: 10 MB.",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    analyze = st.button("Analyze image", type="primary", disabled=uploaded_file is None)
    if not analyze:
        st.caption(f"Inference device: {DEVICE}. The models are loaded only when analysis starts.")
        return

    try:
        image = read_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        with st.spinner("Loading the frozen models and analyzing the image…"):
            stage1_predictor, stage2_predictor = load_predictors()
            stage1_result = stage1_predictor.predict(image)
            stage2_result = stage2_predictor.predict(image) if stage1_result["is_malignant"] else None
            final_prediction = (
                stage1_result["prediction"]
                if stage2_result is None
                else stage2_result["prediction"]
            )
            final_confidence = (
                stage1_result["confidence"]
                if stage2_result is None
                else stage2_result["confidence"]
            )
            interpretation = final_interpretation(stage1_result["prediction"], stage2_result)

        show_prediction(
            {
                "stage1": stage1_result,
                "stage2": stage2_result,
                "final_prediction": final_prediction,
                "final_confidence": final_confidence,
                "interpretation": interpretation,
            }
        )
    except FileNotFoundError as exc:
        logger.exception("A required checkpoint is missing")
        st.error(f"Model loading failed: {exc}")
    except RuntimeError as exc:
        logger.exception("The model could not be loaded or inference failed")
        st.error(f"The AI model could not complete this analysis: {exc}")
    except Exception:
        logger.exception("Unexpected inference failure")
        st.error(
            "Analysis failed unexpectedly. Verify that the model files and dependencies are present, "
            "then restart the application."
        )


if __name__ == "__main__":
    main()
