"""Deterministic, editable interpretation rules for the research screening UI.

This module intentionally contains no model calls and no generated medical advice.  The displayed
screening grade, severity label, and recommendation are fixed mappings from the classification
result and are not a separate clinical model.
"""

from typing import Any, Mapping, Optional


INTERPRETATIONS = {
    "Benign": {
        "grade": "Grade 1",
        "severity": "Low screening concern",
        "severity_level": "low",
        "recommendation": (
            "The lesion is classified as benign, with no malignant features detected. Routine "
            "skin self-checks are recommended; consult a dermatologist if you notice changes in "
            "size, shape, color, or symptoms over time."
        ),
        "summary": "No malignant features detected by the binary screening model.",
    },
    "mel": {
        "grade": "Grade 4",
        "severity": "High screening concern",
        "severity_level": "high",
        "recommendation": (
            "The lesion shows features consistent with melanoma, the most serious form of skin "
            "cancer. Urgent evaluation by a dermatologist or oncologist is strongly recommended."
        ),
        "summary": "Stage 1 identified the lesion as malignant and Stage 2 classified the subtype as melanoma.",
    },
    "bcc": {
        "grade": "Grade 3",
        "severity": "High screening concern",
        "severity_level": "high",
        "recommendation": (
            "The lesion shows features consistent with basal cell carcinoma. This subtype is "
            "typically slow-growing and highly treatable when caught early - prompt dermatologist "
            "evaluation is recommended."
        ),
        "summary": "Stage 1 identified the lesion as malignant and Stage 2 classified the subtype as basal cell carcinoma.",
    },
    "akiec": {
        "grade": "Grade 2",
        "severity": "Moderate screening concern",
        "severity_level": "moderate",
        "recommendation": (
            "The lesion shows features consistent with actinic keratosis or early intraepithelial "
            "carcinoma. Dermatologist evaluation is recommended to determine the appropriate "
            "treatment."
        ),
        "summary": "Stage 1 identified the lesion as malignant and Stage 2 classified the subtype as actinic keratosis or intraepithelial carcinoma.",
    },
}


def _copy_interpretation(key: str) -> dict[str, str]:
    try:
        return dict(INTERPRETATIONS[key])
    except KeyError as exc:
        raise ValueError(f"No deterministic interpretation configured for class: {key}") from exc


def final_interpretation(
    stage1_prediction: str,
    stage2_result: Optional[Mapping[str, Any]] = None,
) -> dict[str, str]:
    """Return the fixed interpretation for the hierarchical pipeline result.

    A benign Stage 1 result maps directly to ``Benign``.  A malignant Stage 1 result must have a
    Stage 2 subtype; refusing an incomplete malignant result prevents disconnected outputs.
    """
    if stage1_prediction == "Benign":
        return _copy_interpretation("Benign")

    if stage1_prediction != "Malignant":
        raise ValueError(f"Unsupported Stage 1 prediction: {stage1_prediction}")

    if not stage2_result or not stage2_result.get("prediction_code"):
        raise ValueError("A malignant Stage 1 result requires a Stage 2 subtype result.")

    return _copy_interpretation(str(stage2_result["prediction_code"]))
