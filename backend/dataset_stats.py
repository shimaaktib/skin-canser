"""
dataset_stats.py
Computes summary statistics from the HAM10000 metadata CSV for the GET /dashboard endpoint.

Read once at application startup and cached in memory in app.py (the CSV never changes at
runtime, so there is no reason to re-parse it on every request) - mirrors the pattern already
used for the two model predictors.
"""
import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("dataset_stats")

# Full human-readable names for the 7 official HAM10000 diagnosis codes.
DX_FULL_NAMES = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}

# Decade-wide age buckets, e.g. "0-10", "10-20", ... "80-90".
AGE_BIN_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def _bin_age(age: float, edges: list) -> Optional[int]:
    """Return the index of the bin `age` falls into, or None if out of range."""
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if lo <= age < hi:
            return i
        if i == len(edges) - 2 and age == hi:  # inclusive right edge on the last bin
            return i
    return None


def compute_dataset_stats(csv_path: Path) -> Optional[dict]:
    """
    Parse the HAM10000 metadata CSV and return a dashboard-ready stats dict.

    Returns None if the file is missing, so the /dashboard endpoint can report that clearly
    instead of the app crashing on startup when the dataset hasn't been placed yet.
    """
    if not csv_path.exists():
        logger.warning(
            "Dataset metadata CSV not found at %s - GET /dashboard will report it as unavailable.",
            csv_path,
        )
        return None

    class_counts: dict = {}
    gender_counts: dict = {}
    location_counts: dict = {}
    ages: list = []
    total = 0
    rows_missing_age = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1

            dx = (row.get("dx") or "").strip() or "unknown"
            class_counts[dx] = class_counts.get(dx, 0) + 1

            sex = (row.get("sex") or "").strip() or "unknown"
            gender_counts[sex] = gender_counts.get(sex, 0) + 1

            loc = (row.get("localization") or "").strip() or "unknown"
            location_counts[loc] = location_counts.get(loc, 0) + 1

            age_raw = (row.get("age") or "").strip()
            if age_raw:
                try:
                    ages.append(float(age_raw))
                except ValueError:
                    rows_missing_age += 1
            else:
                rows_missing_age += 1

    # Age histogram (decade buckets)
    bin_labels = [f"{AGE_BIN_EDGES[i]}-{AGE_BIN_EDGES[i + 1]}" for i in range(len(AGE_BIN_EDGES) - 1)]
    bin_counts = [0] * len(bin_labels)
    for age in ages:
        idx = _bin_age(age, AGE_BIN_EDGES)
        if idx is not None:
            bin_counts[idx] += 1

    diagnosis_types = {
        DX_FULL_NAMES.get(code, code.upper()): count for code, count in class_counts.items()
    }

    average_age = round(sum(ages) / len(ages), 1) if ages else None

    stats = {
        "total_images": total,
        "average_age": average_age,
        "records_with_age": len(ages),
        "records_missing_age": rows_missing_age,
        "class_distribution": class_counts,
        "diagnosis_types": diagnosis_types,
        "gender_distribution": gender_counts,
        "body_locations": location_counts,
        "age_distribution": {
            "bins": bin_labels,
            "counts": bin_counts,
        },
    }

    logger.info(
        "Dataset stats computed: %s images, %s diagnosis classes, %s locations.",
        total, len(class_counts), len(location_counts),
    )
    return stats
