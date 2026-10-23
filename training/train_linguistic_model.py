import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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

from features.linguistic_features import extract_linguistic_features


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/clean_reviews.csv")

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Extract linguistic features
# --------------------------------------------------

print("\nExtracting linguistic features...")

linguistic_rows = []

for text in df["REVIEW_TEXT"]:
    linguistic_rows.append(
        extract_linguistic_features(text)
    )

linguistic_df = pd.DataFrame(linguistic_rows)

print("Linguistic features:", linguistic_df.shape)


# --------------------------------------------------
# 3. Separate target
# --------------------------------------------------

texts = df["REVIEW_TEXT"]
y = df["label"]


# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train_text, X_test_text, X_train_ling, X_test_ling, y_train, y_test = train_test_split(
    texts,
    linguistic_df,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train_text))
print("Testing samples:", len(X_test_text))


# --------------------------------------------------
# 5. TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=100000
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

print("TF-IDF training shape:", X_train_tfidf.shape)


# --------------------------------------------------
# 6. Scale linguistic features
# --------------------------------------------------

scaler = StandardScaler()

X_train_ling_scaled = scaler.fit_transform(X_train_ling)
X_test_ling_scaled = scaler.transform(X_test_ling)


# --------------------------------------------------
# 7. Combine TF-IDF + linguistic features
# --------------------------------------------------

from scipy.sparse import hstack

X_train_combined = hstack([
    X_train_tfidf,
    X_train_ling_scaled
])

X_test_combined = hstack([
    X_test_tfidf,
    X_test_ling_scaled
])

print("Combined training shape:", X_train_combined.shape)


# --------------------------------------------------
# 8. Train Linear SVM
# --------------------------------------------------

print("\nTraining Model E...")

model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_combined,
    y_train
)


# --------------------------------------------------
# 9. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test_combined)
y_score = model.decision_function(X_test_combined)


# --------------------------------------------------
# 10. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_score)

print("\n" + "=" * 60)
print("MODEL E — TF-IDF + LINGUISTIC FEATURES")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# --------------------------------------------------
# 11. Classification report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Truthful", "Deceptive"]
    )
)


# --------------------------------------------------
# 12. Confusion matrix
# --------------------------------------------------

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 13. Save everything needed for inference
# --------------------------------------------------

joblib.dump(
    model,
    "models/linguistic_svm.joblib"
)

joblib.dump(
    tfidf,
    "models/linguistic_tfidf.joblib"
)

joblib.dump(
    scaler,
    "models/linguistic_scaler.joblib"
)

print("\nSaved:")
print("models/linguistic_svm.joblib")
print("models/linguistic_tfidf.joblib")
print("models/linguistic_scaler.joblib")