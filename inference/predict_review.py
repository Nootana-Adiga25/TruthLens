"""
TruthLens - Calibrated Review Prediction

Production inference pipeline:
Review text
    ↓
Word TF-IDF
    +
Character TF-IDF
    +
Linguistic features
    +
Verified purchase
    ↓
Logistic Regression
    ↓
Probability calibration
    ↓
Deception probability
    ↓
Prediction + Risk level
"""

import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack

from features.linguistic_features import extract_linguistic_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "training",
    "models"
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "calibrated_logistic.joblib"
)

WORD_TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "calibrated_word_tfidf.joblib"
)

CHAR_TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "calibrated_char_tfidf.joblib"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "calibrated_scaler.joblib"
)

VERIFIED_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "calibrated_verified_encoder.joblib"
)

CALIBRATION_PATH = os.path.join(
    MODEL_DIR,
    "probability_calibration.joblib"
)


print("Loading TruthLens calibrated model...")

model = joblib.load(MODEL_PATH)

word_tfidf = joblib.load(
    WORD_TFIDF_PATH
)

char_tfidf = joblib.load(
    CHAR_TFIDF_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

verified_encoder = joblib.load(
    VERIFIED_ENCODER_PATH
)

calibration = joblib.load(
    CALIBRATION_PATH
)

A = calibration["A"]
B = calibration["B"]

print("Calibrated model loaded successfully.")


# ============================================================
# RISK THRESHOLDS
# ============================================================

CLASSIFICATION_THRESHOLD = 0.35

LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70


# ============================================================
# PROBABILITY CALIBRATION
# ============================================================

def calibrate_probability(base_probability):
    """
    Convert the model's raw probability into
    calibrated deception probability.
    """

    calibrated = 1.0 / (
        1.0 +
        np.exp(
            -(
                A * base_probability
                + B
            )
        )
    )

    return float(calibrated)


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability):
    """
    Convert calibrated probability into a
    user-facing risk level.
    """

    if probability < LOW_RISK_THRESHOLD:
        return "LOW"

    elif probability < HIGH_RISK_THRESHOLD:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_review(
    review_text,
    verified_purchase=None
):
    """
    Analyze a single review.

    Parameters
    ----------
    review_text : str
        Review text entered by the user.

    verified_purchase : str or None
        "Y", "N", or None.

    Returns
    -------
    dict
        Prediction results.
    """

    # --------------------------------------------------------
    # Validate review
    # --------------------------------------------------------

    if not isinstance(
        review_text,
        str
    ):
        raise ValueError(
            "Review must be a string."
        )

    review_text = review_text.strip()

    if not review_text:
        raise ValueError(
            "Review cannot be empty."
        )

    # --------------------------------------------------------
    # Verified purchase
    # --------------------------------------------------------

    if verified_purchase in ["Y", "N"]:
        verified_value = verified_purchase
    else:
        verified_value = "Unknown"

    # --------------------------------------------------------
    # Word TF-IDF
    # --------------------------------------------------------

    word_features = word_tfidf.transform(
        [review_text]
    )

    # --------------------------------------------------------
    # Character TF-IDF
    # --------------------------------------------------------

    char_features = char_tfidf.transform(
        [review_text]
    )

    # --------------------------------------------------------
    # Linguistic features
    # --------------------------------------------------------

    linguistic_dict = (
        extract_linguistic_features(
            review_text
        )
    )

    linguistic_df = pd.DataFrame(
        [linguistic_dict]
    )

    linguistic_features = scaler.transform(
        linguistic_df.values
    )

    # --------------------------------------------------------
    # Verified purchase encoding
    # --------------------------------------------------------

    verified_input = pd.DataFrame({
        "VERIFIED_PURCHASE": [
            verified_value
        ]
    })

    verified_features = (
        verified_encoder.transform(
            verified_input
        )
    )

    # --------------------------------------------------------
    # Combine all features
    # --------------------------------------------------------

    combined_features = hstack([
        word_features,
        char_features,
        linguistic_features,
        verified_features
    ]).tocsr()

    # --------------------------------------------------------
    # Base model probability
    # --------------------------------------------------------

    base_probability = (
        model.predict_proba(
            combined_features
        )[0, 1]
    )

    # --------------------------------------------------------
    # Calibrate probability
    # --------------------------------------------------------

    deception_probability = (
        calibrate_probability(
            base_probability
        )
    )

    truth_probability = (
        1.0 -
        deception_probability
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if (
        deception_probability
        >= CLASSIFICATION_THRESHOLD
    ):
        prediction = "Deceptive"
    else:
        prediction = "Truthful"

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    risk_level = get_risk_level(
        deception_probability
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "prediction": prediction,

        "deception_probability": round(
            deception_probability * 100,
            2
        ),

        "truth_probability": round(
            truth_probability * 100,
            2
        ),

        "risk_level": risk_level,

        "verified_purchase_used":
            verified_value
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_review = (
        "The product was absolutely amazing. "
        "Best purchase ever! Everyone should buy "
        "this immediately. Five stars!"
    )

    result = predict_review(
        test_review
    )

    print("\n")
    print("=" * 60)
    print("TRUTHLENS REVIEW ANALYSIS")
    print("=" * 60)

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )