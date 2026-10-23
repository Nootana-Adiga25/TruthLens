import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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
# 3. SELECT TEXT AND LABEL
# ============================================================

X = df["REVIEW_TEXT"].astype(str)
y = df["label"]


# ============================================================
# 4. TRAIN / TEST SPLIT
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
# 5. TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=100000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print(
    f"TF-IDF training matrix: "
    f"{X_train_tfidf.shape}"
)

print(
    f"TF-IDF test matrix: "
    f"{X_test_tfidf.shape}"
)


# ============================================================
# 6. TRAIN LINEAR SVM
# ============================================================

print("\nTraining Linear SVM...")

model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)


# ============================================================
# 7. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_tfidf)

# LinearSVC gives a decision score rather than probability.
y_scores = model.decision_function(X_test_tfidf)


# ============================================================
# 8. EVALUATION
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
print("TRUTHLENS BASELINE RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 9. CLASSIFICATION REPORT
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
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 11. SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "baseline_svm.joblib"

vectorizer_path = MODEL_DIR / "baseline_tfidf.joblib"

joblib.dump(
    model,
    model_path
)

joblib.dump(
    vectorizer,
    vectorizer_path
)


print("\nModel saved to:")
print(model_path)

print("\nTF-IDF vectorizer saved to:")
print(vectorizer_path)