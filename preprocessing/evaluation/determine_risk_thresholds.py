"""
TruthLens - Risk Threshold Analysis

Determines defensible LOW / MEDIUM / HIGH risk thresholds
using calibrated probabilities on the calibration dataset.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/clean_reviews.csv"
MODEL_PATH = "training/models/calibrated_logistic.joblib"
CALIBRATION_PATH = "training/models/probability_calibration.joblib"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TRUTHLENS RISK THRESHOLD ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# Labels
# Convert dataset labels to the same numeric format used by the model
# __label1__ = deceptive = 1
# __label2__ = truthful = 0

y = (
    df["LABEL"]
    .map({
        "__label1__": 1,
        "__label2__": 0
    })
    .values
)


# ============================================================
# LOAD CALIBRATION ARTIFACTS
# ============================================================

model = joblib.load(MODEL_PATH)
calibration = joblib.load(CALIBRATION_PATH)

A = calibration["A"]
B = calibration["B"]

print("\nCalibration parameters:")
print(f"A = {A}")
print(f"B = {B}")


# ============================================================
# IMPORTANT
# ============================================================
# We need the same calibration/test split used during
# calibration.
#
# Recreate the exact split:
#   80% development
#   20% test
# Then:
#   75% training
#   25% calibration
#
# However, because the saved model was trained on the 80%
# training portion, we need the calibration portion here.
# ============================================================

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

train_val_idx, test_idx = next(skf.split(df, y))

train_val_df = df.iloc[train_val_idx].reset_index(drop=True)

y_train_val = (
    train_val_df["LABEL"]
    .map({
        "__label1__": 1,
        "__label2__": 0
    })
    .values
)

# Same 75/25 split used in calibration
from sklearn.model_selection import train_test_split

train_idx, calibration_idx = train_test_split(
    np.arange(len(train_val_df)),
    test_size=0.125,
    stratify=y_train_val,
    random_state=RANDOM_STATE
)

calibration_df = train_val_df.iloc[calibration_idx].reset_index(drop=True)

y_calibration = (
    calibration_df["LABEL"]
    .map({
        "__label1__": 1,
        "__label2__": 0
    })
    .values
)


# ============================================================
# LOAD PREPROCESSORS
# ============================================================

word_tfidf = joblib.load(
    "training/models/calibrated_word_tfidf.joblib"
)

char_tfidf = joblib.load(
    "training/models/calibrated_char_tfidf.joblib"
)

scaler = joblib.load(
    "training/models/calibrated_scaler.joblib"
)

verified_encoder = joblib.load(
    "training/models/calibrated_verified_encoder.joblib"
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

from scipy.sparse import hstack

from features.linguistic_features import extract_linguistic_features


texts = calibration_df["REVIEW_TEXT"].fillna("").astype(str)

print("\nGenerating calibration features...")

word_features = word_tfidf.transform(texts)

char_features = char_tfidf.transform(texts)


linguistic_data = pd.DataFrame(
    [
        extract_linguistic_features(text)
        for text in texts
    ]
)

linguistic_features = scaler.transform(
    linguistic_data.values
)


verified_input = pd.DataFrame({
    "VERIFIED_PURCHASE":
        calibration_df["VERIFIED_PURCHASE"]
        .fillna("Unknown")
        .replace("", "Unknown")
})

verified_features = verified_encoder.transform(
    verified_input
)


X_calibration = hstack([
    word_features,
    char_features,
    linguistic_features,
    verified_features
]).tocsr()


# ============================================================
# GENERATE BASE PROBABILITIES
# ============================================================

print("Generating calibrated probabilities...")

base_probability = model.predict_proba(
    X_calibration
)[:, 1]


# Apply sigmoid calibration

calibrated_probability = (
    1 /
    (
        1 +
        np.exp(
            -(A * base_probability + B)
        )
    )
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

thresholds = np.arange(
    0.30,
    0.81,
    0.05
)

results = []

for threshold in thresholds:

    predictions = (
        calibrated_probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_calibration,
        predictions
    )

    precision = precision_score(
        y_calibration,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_calibration,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_calibration,
        predictions,
        zero_division=0
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)

print("\n" + "=" * 60)
print("BEST F1 THRESHOLD")
print("=" * 60)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {best_row['accuracy']:.4f}"
)

print(
    f"Precision : {best_row['precision']:.4f}"
)

print(
    f"Recall    : {best_row['recall']:.4f}"
)

print(
    f"F1        : {best_row['f1']:.4f}"
)


# ============================================================
# PROPOSE RISK BANDS
# ============================================================

print("\n" + "=" * 60)
print("PROPOSED RISK BANDS")
print("=" * 60)

print("""
LOW:
    Probability < 0.30

MEDIUM:
    0.30 <= Probability < 0.70

HIGH:
    Probability >= 0.70
""")


# ============================================================
# DISTRIBUTION
# ============================================================

print("=" * 60)
print("PROBABILITY DISTRIBUTION")
print("=" * 60)

print(
    pd.Series(calibrated_probability).describe()
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "evaluation",
    exist_ok=True
)

results_df.to_csv(
    "evaluation/risk_threshold_analysis.csv",
    index=False
)

print(
    "\nSaved:"
    "\nevaluation/risk_threshold_analysis.csv"
)

print("\nDone.")