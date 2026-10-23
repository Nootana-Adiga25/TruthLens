import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "clean_reviews.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. SELECT FEATURES
# ============================================================

feature_columns = [
    "REVIEW_TEXT",
    "REVIEW_TITLE",
    "RATING",
    "VERIFIED_PURCHASE",
    "PRODUCT_CATEGORY"
]

X = df[feature_columns].copy()
y = df["label"]


# ============================================================
# 4. CLEAN FEATURE TYPES
# ============================================================

X["REVIEW_TEXT"] = X["REVIEW_TEXT"].fillna("").astype(str)

X["REVIEW_TITLE"] = X["REVIEW_TITLE"].fillna("").astype(str)

X["VERIFIED_PURCHASE"] = (
    X["VERIFIED_PURCHASE"]
    .fillna("UNKNOWN")
    .astype(str)
)

X["PRODUCT_CATEGORY"] = (
    X["PRODUCT_CATEGORY"]
    .fillna("UNKNOWN")
    .astype(str)
)

X["RATING"] = pd.to_numeric(
    X["RATING"],
    errors="coerce"
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData split:")
print(f"Training reviews: {len(X_train)}")
print(f"Testing reviews:  {len(X_test)}")


# ============================================================
# 6. FEATURE TRANSFORMERS
# ============================================================

text_features = "REVIEW_TEXT"
title_features = "REVIEW_TITLE"

categorical_features = [
    "VERIFIED_PURCHASE",
    "PRODUCT_CATEGORY"
]

numeric_features = [
    "RATING"
]


# ============================================================
# 7. PREPROCESSING
# ============================================================

text_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=100000
        )
    )
])


title_pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000
        )
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])


preprocessor = ColumnTransformer(
    transformers=[
        (
            "review_text",
            text_pipeline,
            text_features
        ),
        (
            "review_title",
            title_pipeline,
            title_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    ]
)


# ============================================================
# 8. COMPLETE MODEL PIPELINE
# ============================================================

model_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42
        )
    )
])


# ============================================================
# 9. TRAIN
# ============================================================

print("\nTraining metadata-enhanced model...")

model_pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 10. PREDICT
# ============================================================

print("\nGenerating predictions...")

y_pred = model_pipeline.predict(X_test)

y_scores = model_pipeline.decision_function(
    X_test
)


# ============================================================
# 11. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_scores
)


print("\n" + "=" * 60)
print("TRUTHLENS MODEL B RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Truthful",
            "Deceptive"
        ]
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "metadata_svm.joblib"
)

joblib.dump(
    model_pipeline,
    model_path
)

print("\nModel saved to:")
print(model_path)