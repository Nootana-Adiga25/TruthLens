"""
TruthLens - Review Explainability

Provides:
    1. Prediction
    2. Actual model calculation
    3. Probability calibration details
    4. Risk calculation
    5. Linguistic signals
    6. Model-indicative word contributions

IMPORTANT:
This file uses the same preprocessing pipeline as
inference/predict_review.py.
"""


import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack

from features.linguistic_features import (
    extract_linguistic_features
)

from inference.predict_review import (
    predict_review,
    calibrate_probability,
    get_risk_level,
    CLASSIFICATION_THRESHOLD,
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "training",
    "models"
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

print("Loading TruthLens explainability components...")

model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "calibrated_logistic.joblib"
    )
)

word_tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "calibrated_word_tfidf.joblib"
    )
)

char_tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "calibrated_char_tfidf.joblib"
    )
)

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "calibrated_scaler.joblib"
    )
)

verified_encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "calibrated_verified_encoder.joblib"
    )
)

calibration = joblib.load(
    os.path.join(
        MODEL_DIR,
        "probability_calibration.joblib"
    )
)

A = calibration["A"]
B = calibration["B"]

print("Explainability components loaded successfully.")


# ============================================================
# LINGUISTIC SIGNAL ANALYSIS
# ============================================================

def analyze_linguistic_signals(review_text):

    features = extract_linguistic_features(
        review_text
    )

    signals = []

    if features["word_count"] < 30:

        signals.append({
            "signal": "Short review",
            "value": features["word_count"],
            "description": (
                "The review contains relatively few words."
            )
        })

    if features["word_count"] > 150:

        signals.append({
            "signal": "Long review",
            "value": features["word_count"],
            "description": (
                "The review is substantially longer "
                "than a typical short review."
            )
        })

    if features["promotional_word_count"] > 2:

        signals.append({
            "signal": "Promotional language",
            "value": (
                features["promotional_word_count"]
            ),
            "description": (
                "Multiple promotional or persuasive "
                "terms were detected."
            )
        })

    if features["superlative_count"] > 0:

        signals.append({
            "signal": "Superlative language",
            "value": (
                features["superlative_count"]
            ),
            "description": (
                "Superlative wording was detected."
            )
        })

    if features["exclamation_count"] >= 2:

        signals.append({
            "signal": "Frequent exclamation marks",
            "value": (
                features["exclamation_count"]
            ),
            "description": (
                "The review uses multiple "
                "exclamation marks."
            )
        })

    if features["question_count"] >= 2:

        signals.append({
            "signal": "Multiple questions",
            "value": (
                features["question_count"]
            ),
            "description": (
                "Several questions appear "
                "in the review."
            )
        })

    if features["first_person_ratio"] > 0.10:

        signals.append({
            "signal": "High first-person usage",
            "value": round(
                features["first_person_ratio"],
                3
            ),
            "description": (
                "The review contains relatively "
                "frequent first-person references."
            )
        })

    if features["repeated_word_ratio"] > 0.30:

        signals.append({
            "signal": "Repeated wording",
            "value": round(
                features["repeated_word_ratio"],
                3
            ),
            "description": (
                "A relatively high amount of "
                "repeated vocabulary was detected."
            )
        })

    if features["positive_word_ratio"] > 0.05:

        signals.append({
            "signal": "Strong positive language",
            "value": round(
                features["positive_word_ratio"],
                3
            ),
            "description": (
                "The review contains a high "
                "proportion of positive words."
            )
        })

    if features["negative_word_ratio"] > 0.05:

        signals.append({
            "signal": "Strong negative language",
            "value": round(
                features["negative_word_ratio"],
                3
            ),
            "description": (
                "The review contains a high "
                "proportion of negative words."
            )
        })

    return features, signals


# ============================================================
# EXACT FEATURE CONSTRUCTION
# ============================================================

def build_model_features(
    review_text,
    verified_purchase=None
):

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
    #
    # EXACTLY the same approach as predict_review.py
    # --------------------------------------------------------

    linguistic_dict = (
        extract_linguistic_features(
            review_text
        )
    )

    linguistic_df = pd.DataFrame(
        [linguistic_dict]
    )

    linguistic_features = (
        scaler.transform(
            linguistic_df.values
        )
    )

    # --------------------------------------------------------
    # Verified purchase encoding
    #
    # EXACTLY the same DataFrame structure
    # as predict_review.py
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
    # Combine
    # --------------------------------------------------------

    combined_features = hstack([
        word_features,
        char_features,
        linguistic_features,
        verified_features
    ]).tocsr()

    return (
        combined_features,
        verified_value
    )


# ============================================================
# WORD CONTRIBUTIONS
# ============================================================

def get_word_contributions(
    review_text,
    top_n=10
):

    word_vector = word_tfidf.transform(
        [review_text]
    )

    feature_names = (
        word_tfidf.get_feature_names_out()
    )

    # Word TF-IDF is the first block of the
    # combined feature vector.

    word_coefficients = model.coef_[0][
        :len(feature_names)
    ]

    values = word_vector.toarray()[0]

    contributions = (
        values * word_coefficients
    )

    results = []

    for index in np.argsort(
        np.abs(contributions)
    )[::-1]:

        if values[index] == 0:
            continue

        contribution = float(
            contributions[index]
        )

        results.append({
            "term": feature_names[index],

            "contribution": round(
                contribution,
                4
            ),

            "direction": (
                "deceptive"
                if contribution > 0
                else "truthful"
            )
        })

        if len(results) >= top_n:
            break

    deceptive_terms = [
        item
        for item in results
        if item["direction"] == "deceptive"
    ]

    truthful_terms = [
        item
        for item in results
        if item["direction"] == "truthful"
    ]

    return {
        "deceptive_terms": deceptive_terms,
        "truthful_terms": truthful_terms
    }


# ============================================================
# ACTUAL MODEL CALCULATION
# ============================================================

def get_calculation_details(
    review_text,
    verified_purchase=None
):

    # Build EXACT same feature matrix
    # used by predict_review.py.

    combined_features, verified_value = (
        build_model_features(
            review_text,
            verified_purchase
        )
    )

    # --------------------------------------------------------
    # Logistic Regression decision score
    #
    # z = w^T x + b
    # --------------------------------------------------------

    decision_score = float(
        model.decision_function(
            combined_features
        )[0]
    )

    # --------------------------------------------------------
    # Base probability
    #
    # P_base = 1 / (1 + e^(-z))
    # --------------------------------------------------------

    base_probability = float(
        model.predict_proba(
            combined_features
        )[0, 1]
    )

    # Independently calculate the sigmoid
    # from z to demonstrate the mathematics.

    sigmoid_probability = (
        1.0 /
        (
            1.0 +
            np.exp(
                -decision_score
            )
        )
    )

    # --------------------------------------------------------
    # Calibrated probability
    # Use the EXACT calibration function
    # imported from predict_review.py.
    # --------------------------------------------------------

    calibrated_probability = (
        calibrate_probability(
            base_probability
        )
    )

    # --------------------------------------------------------
    # Truth probability
    # --------------------------------------------------------

    truth_probability = (
        1.0 -
        calibrated_probability
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = (
        "Deceptive"
        if calibrated_probability
        >= CLASSIFICATION_THRESHOLD
        else "Truthful"
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk_level = get_risk_level(
        calibrated_probability
    )

    return {

        "logistic": {

            "decision_score_z": round(
                decision_score,
                6
            ),

            "formula": (
                "z = w^T x + b"
            ),

            "probability_formula": (
                "P_base = 1 / "
                "(1 + exp(-z))"
            ),

            "sigmoid_probability": round(
                sigmoid_probability * 100,
                2
            ),

            "model_probability": round(
                base_probability * 100,
                2
            )
        },

        "calibration": {

            "method": (
                "Sigmoid probability calibration"
            ),

            "formula": (
                "P_calibrated = "
                "1 / "
                "(1 + exp(-(A × P_base + B)))"
            ),

            "A": A,

            "B": B,

            "base_probability": round(
                base_probability * 100,
                2
            ),

            "calibrated_probability": round(
                calibrated_probability * 100,
                2
            )
        },

        "classification": {

            "threshold": (
                CLASSIFICATION_THRESHOLD * 100
            ),

            "rule": (
                "Deceptive if "
                "P_deception >= 35%"
            ),

            "prediction": prediction
        },

        "risk": {

            "low": "< 30%",

            "medium": "30% - < 70%",

            "high": ">= 70%",

            "current": risk_level
        },

        "final": {

            "deception_probability": round(
                calibrated_probability * 100,
                2
            ),

            "truth_probability": round(
                truth_probability * 100,
                2
            ),

            "prediction": prediction,

            "risk_level": risk_level
        },

        "verified_purchase": verified_value
    }


# ============================================================
# COMPLETE EXPLANATION
# ============================================================

def explain_review(
    review_text,
    verified_purchase=None
):

    # Use the production prediction function.
    prediction = predict_review(
        review_text,
        verified_purchase=verified_purchase
    )

    # Linguistic features/signals.
    linguistic_features, signals = (
        analyze_linguistic_signals(
            review_text
        )
    )

    # Word-level contributions.
    contributions = (
        get_word_contributions(
            review_text
        )
    )

    # Actual mathematical calculation.
    calculations = (
        get_calculation_details(
            review_text,
            verified_purchase
        )
    )

    return {

        "prediction": prediction,

        "calculations": calculations,

        "linguistic_features":
            linguistic_features,

        "signals": signals,

        "term_contributions":
            contributions
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    review = (
        "Pleasant staff and housekeeping. "
        "Above average breakfast. "
        "The rooms were clean and the staff "
        "made sure we had fresh linnens. "
        "It was bedbug free!! "
        "I would definitely stay there again"
    )

    result = explain_review(
        review
    )

    print()
    print("=" * 60)
    print("TRUTHLENS EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nPrediction:")

    print(
        result["prediction"]
    )

    calculations = (
        result["calculations"]
    )

    # --------------------------------------------------------
    # Actual calculation
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ACTUAL MODEL CALCULATION")
    print("=" * 60)

    print("\nLogistic Regression:")

    print(
        f"Decision score (z): "
        f"{calculations['logistic']['decision_score_z']}"
    )

    print(
        f"Sigmoid probability: "
        f"{calculations['logistic']['sigmoid_probability']}%"
    )

    print(
        f"Model probability: "
        f"{calculations['logistic']['model_probability']}%"
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    print("\nCalibration:")

    print(
        calculations[
            "calibration"
        ]["formula"]
    )

    print(
        f"A = "
        f"{calculations['calibration']['A']}"
    )

    print(
        f"B = "
        f"{calculations['calibration']['B']}"
    )

    print(
        f"Calibrated probability: "
        f"{calculations['calibration']['calibrated_probability']}%"
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    print("\nClassification:")

    print(
        f"Threshold: "
        f"{calculations['classification']['threshold']}%"
    )

    print(
        f"Rule: "
        f"{calculations['classification']['rule']}"
    )

    print(
        f"Prediction: "
        f"{calculations['classification']['prediction']}"
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    print("\nRisk:")

    print(
        f"LOW: "
        f"{calculations['risk']['low']}"
    )

    print(
        f"MEDIUM: "
        f"{calculations['risk']['medium']}"
    )

    print(
        f"HIGH: "
        f"{calculations['risk']['high']}"
    )

    print(
        f"Current risk: "
        f"{calculations['risk']['current']}"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\nFinal Result:")

    print(
        f"Deception probability: "
        f"{calculations['final']['deception_probability']}%"
    )

    print(
        f"Truth probability: "
        f"{calculations['final']['truth_probability']}%"
    )

    print(
        f"Risk level: "
        f"{calculations['final']['risk_level']}"
    )

    # --------------------------------------------------------
    # Linguistic signals
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LINGUISTIC SIGNALS")
    print("=" * 60)

    if result["signals"]:

        for signal in result["signals"]:

            print(
                f"- {signal['signal']}: "
                f"{signal['description']}"
            )

    else:

        print(
            "No heuristic linguistic signals detected."
        )

    # --------------------------------------------------------
    # Word contributions
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL-INDICATIVE TERM CONTRIBUTIONS")
    print("=" * 60)

    print(
        "\nTerms associated with deception:"
    )

    for item in (
        result["term_contributions"]
        ["deceptive_terms"]
    ):

        print(
            f"- {item['term']}: "
            f"{item['contribution']}"
        )

    print(
        "\nTerms associated with "
        "truthful classification:"
    )

    for item in (
        result["term_contributions"]
        ["truthful_terms"]
    ):

        print(
            f"- {item['term']}: "
            f"{item['contribution']}"
        )