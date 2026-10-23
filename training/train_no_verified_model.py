import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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

# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv("data/clean_reviews.csv")

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. Select features
# --------------------------------------------------
# IMPORTANT:
# VERIFIED_PURCHASE is intentionally excluded.
# We want to measure the contribution of the other
# available metadata.

X = df[
    [
        "REVIEW_TEXT",
        "REVIEW_TITLE",
        "RATING",
        "PRODUCT_CATEGORY"
    ]
]

y = df["label"]

# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "review_text",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=100000
            ),
            "REVIEW_TEXT"
        ),
        (
            "review_title",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=30000
            ),
            "REVIEW_TITLE"
        ),
        (
            "rating",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            ),
            ["RATING"]
        ),
        (
            "product_category",
            OneHotEncoder(handle_unknown="ignore"),
            ["PRODUCT_CATEGORY"]
        )
    ]
)

# --------------------------------------------------
# 5. Model
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

# --------------------------------------------------
# 6. Train
# --------------------------------------------------

print("\nTraining Model D...")
model.fit(X_train, y_train)

# --------------------------------------------------
# 7. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)
y_score = model.decision_function(X_test)

# --------------------------------------------------
# 8. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_score)

print("\n" + "=" * 50)
print("MODEL D — TEXT + TITLE + RATING + CATEGORY")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Truthful", "Deceptive"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 9. Save model
# --------------------------------------------------

joblib.dump(
    model,
    "models/no_verified_svm.joblib"
)

print("\nModel saved to:")
print("models/no_verified_svm.joblib")