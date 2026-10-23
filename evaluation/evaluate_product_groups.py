import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from scipy.sparse import hstack

from features.linguistic_features import extract_linguistic_features


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/clean_reviews.csv")

print("Dataset shape:", df.shape)
print("Unique products:", df["PRODUCT_ID"].nunique())


# ============================================================
# 2. BASIC DATA
# ============================================================

texts = df["REVIEW_TEXT"].fillna("").astype(str).values
verified = df["VERIFIED_PURCHASE"].fillna("Unknown").astype(str).values
groups = df["PRODUCT_ID"].astype(str).values
y = df["label"].values


# ============================================================
# 3. STRATIFIED GROUP K-FOLD
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


results = []


# ============================================================
# 4. CROSS-VALIDATION
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    cv.split(texts, y, groups),
    start=1
):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    print("Training samples:", len(train_idx))
    print("Testing samples :", len(test_idx))

    print(
        "Training products:",
        len(set(groups[train_idx]))
    )

    print(
        "Testing products :",
        len(set(groups[test_idx]))
    )

    # --------------------------------------------------------
    # WORD TF-IDF
    # --------------------------------------------------------

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

    X_word_test = word_tfidf.transform(
        texts[test_idx]
    )


    # --------------------------------------------------------
    # CHARACTER TF-IDF
    # --------------------------------------------------------

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

    X_char_test = char_tfidf.transform(
        texts[test_idx]
    )


    # --------------------------------------------------------
    # LINGUISTIC FEATURES
    # --------------------------------------------------------

    linguistic_train = pd.DataFrame([
        extract_linguistic_features(text)
        for text in texts[train_idx]
    ])

    linguistic_test = pd.DataFrame([
        extract_linguistic_features(text)
        for text in texts[test_idx]
    ])

    scaler = StandardScaler()

    X_ling_train = scaler.fit_transform(
        linguistic_train
    )

    X_ling_test = scaler.transform(
        linguistic_test
    )


    # --------------------------------------------------------
    # VERIFIED PURCHASE
    # --------------------------------------------------------

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

    X_verified_train = encoder.fit_transform(
        verified[train_idx].reshape(-1, 1)
    )

    X_verified_test = encoder.transform(
        verified[test_idx].reshape(-1, 1)
    )


    # --------------------------------------------------------
    # COMBINE FEATURES
    # --------------------------------------------------------

    X_train = hstack([
        X_word_train,
        X_char_train,
        X_ling_train,
        X_verified_train
    ]).tocsr()

    X_test = hstack([
        X_word_test,
        X_char_test,
        X_ling_test,
        X_verified_test
    ]).tocsr()


    print("Feature matrix:", X_train.shape)


    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42
    )

    model.fit(X_train, y[train_idx])


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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
        y_prob
    )


    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


    results.append({
        "fold": fold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    })


# ============================================================
# 5. FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("PRODUCT-GROUP CROSS-VALIDATION RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))


print("\nMean ± Standard Deviation")
print("-" * 40)

for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]:

    mean = results_df[metric].mean()
    std = results_df[metric].std()

    print(
        f"{metric.upper():10s}: "
        f"{mean:.4f} ± {std:.4f}"
    )


# ============================================================
# 6. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "evaluation/product_group_results.csv",
    index=False
)

print(
    "\nResults saved to "
    "evaluation/product_group_results.csv"
)

