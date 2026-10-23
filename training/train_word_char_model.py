import pandas as pd
import numpy as np
import joblib

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

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


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv("data/clean_reviews.csv")

print("Dataset shape:", df.shape)


# ==========================================================
# 2. BASIC DATA
# ==========================================================

texts = (
    df["REVIEW_TEXT"]
    .fillna("")
    .astype(str)
    .values
)

y = df["label"].values


# ==========================================================
# 3. TRAIN / TEST SPLIT
# ==========================================================

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(train_idx))
print("Testing samples:", len(test_idx))


X_train_text = texts[train_idx]
X_test_text = texts[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]


# ==========================================================
# 4. LINGUISTIC FEATURES
# ==========================================================

print("\nExtracting linguistic features...")

linguistic_data = [
    extract_linguistic_features(text)
    for text in df["REVIEW_TEXT"].fillna("").astype(str)
]

linguistic_df = pd.DataFrame(
    linguistic_data
)

print(
    "Linguistic features:",
    linguistic_df.shape
)

linguistic_train = linguistic_df.iloc[
    train_idx
].values

linguistic_test = linguistic_df.iloc[
    test_idx
].values


# Scale linguistic features using TRAINING data only

scaler = StandardScaler()

linguistic_train_scaled = scaler.fit_transform(
    linguistic_train
)

linguistic_test_scaled = scaler.transform(
    linguistic_test
)

linguistic_train_scaled = csr_matrix(
    linguistic_train_scaled
)

linguistic_test_scaled = csr_matrix(
    linguistic_test_scaled
)


# ==========================================================
# 5. VERIFIED PURCHASE
# ==========================================================

print("\nEncoding verified purchase...")

verified_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

verified_train = verified_encoder.fit_transform(
    df.iloc[train_idx][["VERIFIED_PURCHASE"]]
)

verified_test = verified_encoder.transform(
    df.iloc[test_idx][["VERIFIED_PURCHASE"]]
)

print(
    "Verified purchase features:",
    verified_train.shape
)


# ==========================================================
# 6. WORD TF-IDF
# ==========================================================

print("\nCreating WORD TF-IDF features...")

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

X_train_word = word_tfidf.fit_transform(
    X_train_text
)

X_test_word = word_tfidf.transform(
    X_test_text
)

print(
    "Word TF-IDF training shape:",
    X_train_word.shape
)


# ==========================================================
# 7. CHARACTER TF-IDF
# ==========================================================

print("\nCreating CHARACTER TF-IDF features...")

char_tfidf = TfidfVectorizer(
    lowercase=True,
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_features=100000,
    sublinear_tf=True
)

X_train_char = char_tfidf.fit_transform(
    X_train_text
)

X_test_char = char_tfidf.transform(
    X_test_text
)

print(
    "Character TF-IDF training shape:",
    X_train_char.shape
)


# ==========================================================
# 8. COMBINE FEATURES
# ==========================================================

print("\nCombining features...")

X_train_combined = hstack([
    X_train_word,
    X_train_char,
    linguistic_train_scaled,
    verified_train
]).tocsr()

X_test_combined = hstack([
    X_test_word,
    X_test_char,
    linguistic_test_scaled,
    verified_test
]).tocsr()


print(
    "Combined training shape:",
    X_train_combined.shape
)

print(
    "Combined testing shape:",
    X_test_combined.shape
)


# ==========================================================
# 9. LOGISTIC REGRESSION
# ==========================================================

print("\nTraining Model G...")

model = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    solver="liblinear",
    random_state=42
)

model.fit(
    X_train_combined,
    y_train
)


# ==========================================================
# 10. PREDICTIONS
# ==========================================================

y_pred = model.predict(
    X_test_combined
)

y_probability = model.predict_proba(
    X_test_combined
)[:, 1]


# ==========================================================
# 11. METRICS
# ==========================================================

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
    y_probability
)


# ==========================================================
# 12. RESULTS
# ==========================================================

print("\n" + "=" * 60)

print(
    "MODEL G — WORD + CHARACTER TF-IDF + "
    "LINGUISTIC + VERIFIED"
)

print("=" * 60)

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
        target_names=[
            "Truthful",
            "Deceptive"
        ]
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================================
# 13. SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    "training/models/word_char_logistic.joblib"
)

joblib.dump(
    word_tfidf,
    "training/models/word_tfidf.joblib"
)

joblib.dump(
    char_tfidf,
    "training/models/char_tfidf.joblib"
)

joblib.dump(
    scaler,
    "training/models/word_char_scaler.joblib"
)

joblib.dump(
    verified_encoder,
    "training/models/word_char_verified_encoder.joblib"
)


print("\nSaved Model G files:")

print(
    "training/models/word_char_logistic.joblib"
)

print(
    "training/models/word_tfidf.joblib"
)

print(
    "training/models/char_tfidf.joblib"
)

print(
    "training/models/word_char_scaler.joblib"
)

print(
    "training/models/word_char_verified_encoder.joblib"
)