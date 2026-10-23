import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from scipy.special import expit
from scipy.optimize import minimize

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss
)

from features.linguistic_features import extract_linguistic_features


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/clean_reviews.csv")

texts = (
    df["REVIEW_TEXT"]
    .fillna("")
    .astype(str)
    .values
)

verified = (
    df["VERIFIED_PURCHASE"]
    .fillna("Unknown")
    .astype(str)
    .values
)

y = df["label"].values


print("=" * 60)
print("TRUTHLENS PROBABILITY CALIBRATION")
print("=" * 60)

print("Dataset shape:", df.shape)


# ============================================================
# 2. CREATE TRAIN / CALIBRATION / TEST SPLIT
# ============================================================

# First create a 60/40 split.
outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, temporary_idx = next(
    outer_cv.split(texts, y)
)


# Split the remaining 40% into:
# 20% calibration
# 20% final test

temporary_y = y[temporary_idx]

inner_cv = StratifiedKFold(
    n_splits=2,
    shuffle=True,
    random_state=42
)

cal_relative_idx, test_relative_idx = next(
    inner_cv.split(
        temporary_idx,
        temporary_y
    )
)

cal_idx = temporary_idx[cal_relative_idx]
test_idx = temporary_idx[test_relative_idx]


print("\nData split:")
print("Training samples   :", len(train_idx))
print("Calibration samples:", len(cal_idx))
print("Test samples       :", len(test_idx))


# ============================================================
# 3. WORD TF-IDF
# ============================================================

print("\nTraining word TF-IDF...")

word_tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    analyzer="word",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=100000
)

X_word_train = word_tfidf.fit_transform(
    texts[train_idx]
)

X_word_cal = word_tfidf.transform(
    texts[cal_idx]
)

X_word_test = word_tfidf.transform(
    texts[test_idx]
)

print(
    "Word features:",
    X_word_train.shape[1]
)


# ============================================================
# 4. CHARACTER TF-IDF
# ============================================================

print("\nTraining character TF-IDF...")

char_tfidf = TfidfVectorizer(
    lowercase=True,
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_features=100000,
    sublinear_tf=True
)

X_char_train = char_tfidf.fit_transform(
    texts[train_idx]
)

X_char_cal = char_tfidf.transform(
    texts[cal_idx]
)

X_char_test = char_tfidf.transform(
    texts[test_idx]
)

print(
    "Character features:",
    X_char_train.shape[1]
)


# ============================================================
# 5. LINGUISTIC FEATURES
# ============================================================

print("\nExtracting linguistic features...")

ling_train = pd.DataFrame([
    extract_linguistic_features(text)
    for text in texts[train_idx]
])

ling_cal = pd.DataFrame([
    extract_linguistic_features(text)
    for text in texts[cal_idx]
])

ling_test = pd.DataFrame([
    extract_linguistic_features(text)
    for text in texts[test_idx]
])


scaler = StandardScaler()

X_ling_train = scaler.fit_transform(
    ling_train.values
)

X_ling_cal = scaler.transform(
    ling_cal.values
)

X_ling_test = scaler.transform(
    ling_test.values
)

print(
    "Linguistic features:",
    X_ling_train.shape[1]
)


# ============================================================
# 6. VERIFIED PURCHASE
# ============================================================

print("\nEncoding verified purchase...")

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)


verified_train = pd.DataFrame({
    "VERIFIED_PURCHASE": verified[train_idx]
})

verified_cal = pd.DataFrame({
    "VERIFIED_PURCHASE": verified[cal_idx]
})

verified_test = pd.DataFrame({
    "VERIFIED_PURCHASE": verified[test_idx]
})


X_verified_train = encoder.fit_transform(
    verified_train
)

X_verified_cal = encoder.transform(
    verified_cal
)

X_verified_test = encoder.transform(
    verified_test
)


# ============================================================
# 7. COMBINE ALL FEATURES
# ============================================================

print("\nCombining features...")

X_train = hstack([
    X_word_train,
    X_char_train,
    X_ling_train,
    X_verified_train
]).tocsr()


X_cal = hstack([
    X_word_cal,
    X_char_cal,
    X_ling_cal,
    X_verified_cal
]).tocsr()


X_test = hstack([
    X_word_test,
    X_char_test,
    X_ling_test,
    X_verified_test
]).tocsr()


print("Training matrix   :", X_train.shape)
print("Calibration matrix:", X_cal.shape)
print("Test matrix       :", X_test.shape)


# ============================================================
# 8. TRAIN MODEL G
# ============================================================

print("\nTraining Logistic Regression...")

base_model = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    solver="liblinear",
    random_state=42
)

base_model.fit(
    X_train,
    y[train_idx]
)

print("Base model trained.")


# ============================================================
# 9. GET UNCALIBRATED CALIBRATION PROBABILITIES
# ============================================================

print("\nGenerating calibration probabilities...")

calibration_probabilities = (
    base_model.predict_proba(
        X_cal
    )[:, 1]
)

y_cal = y[cal_idx]


# ============================================================
# 10. SIGMOID CALIBRATION
# ============================================================

print("Fitting sigmoid calibration...")


def calibration_loss(
    parameters,
    probabilities,
    labels
):
    """
    Learn a sigmoid transformation:

        calibrated = sigmoid(A * probability + B)
    """

    A, B = parameters

    calibrated = expit(
        A * probabilities + B
    )

    calibrated = np.clip(
        calibrated,
        1e-15,
        1 - 1e-15
    )

    loss = -np.mean(
        labels * np.log(calibrated)
        +
        (1 - labels)
        * np.log(1 - calibrated)
    )

    return loss


initial_parameters = np.array([
    1.0,
    0.0
])


optimization_result = minimize(
    calibration_loss,
    initial_parameters,
    args=(
        calibration_probabilities,
        y_cal
    ),
    method="L-BFGS-B"
)


if not optimization_result.success:
    print(
        "\nWARNING: Calibration optimizer "
        "did not report successful convergence."
    )


A, B = optimization_result.x


print("\nCalibration parameters:")
print("A =", A)
print("B =", B)


# ============================================================
# 11. TEST SET PREDICTIONS
# ============================================================

print("\nEvaluating on untouched test set...")

# Original Model G probability
uncalibrated_probability = (
    base_model.predict_proba(
        X_test
    )[:, 1]
)


# Apply learned sigmoid calibration
calibrated_probability = expit(
    A * uncalibrated_probability + B
)


# Final calibrated classification threshold
y_pred = (
    calibrated_probability >= 0.50
).astype(int)


# ============================================================
# 12. METRICS
# ============================================================

accuracy = accuracy_score(
    y[test_idx],
    y_pred
)

precision = precision_score(
    y[test_idx],
    y_pred
)

recall = recall_score(
    y[test_idx],
    y_pred
)

f1 = f1_score(
    y[test_idx],
    y_pred
)

roc_auc = roc_auc_score(
    y[test_idx],
    calibrated_probability
)

brier = brier_score_loss(
    y[test_idx],
    calibrated_probability
)


# ============================================================
# 13. RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("CALIBRATED MODEL RESULTS")
print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"Brier    : {brier:.4f}"
)


# ============================================================
# 14. COMPARE CALIBRATED VS UNCALIBRATED
# ============================================================

uncalibrated_brier = brier_score_loss(
    y[test_idx],
    uncalibrated_probability
)


print("\n")
print("=" * 60)
print("PROBABILITY CALIBRATION COMPARISON")
print("=" * 60)

print(
    f"Uncalibrated Brier : "
    f"{uncalibrated_brier:.4f}"
)

print(
    f"Calibrated Brier   : "
    f"{brier:.4f}"
)


if brier < uncalibrated_brier:
    print(
        "\nCalibration improved the Brier score."
    )
else:
    print(
        "\nCalibration did not improve the Brier score."
    )


# ============================================================
# 15. PROBABILITY STATISTICS
# ============================================================

print("\n")
print("=" * 60)
print("CALIBRATED PROBABILITY DISTRIBUTION")
print("=" * 60)

probability_statistics = pd.Series(
    calibrated_probability
).describe()

print(
    probability_statistics
)


# ============================================================
# 16. SAMPLE PROBABILITIES
# ============================================================

print("\nSample calibrated probabilities:")

for probability in calibrated_probability[:10]:

    print(
        f"{probability * 100:.2f}%"
    )


# ============================================================
# 17. SAVE MODEL
# ============================================================

print("\nSaving calibrated model artifacts...")


joblib.dump(
    base_model,
    "training/models/calibrated_logistic.joblib"
)


joblib.dump(
    {
        "A": A,
        "B": B
    },
    "training/models/probability_calibration.joblib"
)


joblib.dump(
    word_tfidf,
    "training/models/calibrated_word_tfidf.joblib"
)


joblib.dump(
    char_tfidf,
    "training/models/calibrated_char_tfidf.joblib"
)


joblib.dump(
    scaler,
    "training/models/calibrated_scaler.joblib"
)


joblib.dump(
    encoder,
    "training/models/calibrated_verified_encoder.joblib"
)


# ============================================================
# 18. SAVE METRICS
# ============================================================

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "brier_score": brier,
    "uncalibrated_brier_score": uncalibrated_brier,
    "calibration_A": A,
    "calibration_B": B
}


pd.DataFrame(
    [metrics]
).to_csv(
    "evaluation/calibration_results.csv",
    index=False
)


print("\nCalibration artifacts saved successfully.")

print(
    "\nModel:"
    "\n  training/models/calibrated_logistic.joblib"
)

print(
    "\nCalibration:"
    "\n  training/models/probability_calibration.joblib"
)

print(
    "\nPreprocessors:"
    "\n  calibrated_word_tfidf.joblib"
    "\n  calibrated_char_tfidf.joblib"
    "\n  calibrated_scaler.joblib"
    "\n  calibrated_verified_encoder.joblib"
)

print(
    "\nMetrics:"
    "\n  evaluation/calibration_results.csv"
)

print("\nDone.")