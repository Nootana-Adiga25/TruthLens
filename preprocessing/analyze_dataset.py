import pandas as pd
from pathlib import Path


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "clean_reviews.csv"


# Load data
df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")


# -----------------------------------
# Review length
# -----------------------------------

df["review_length"] = df["REVIEW_TEXT"].str.len()

df["word_count"] = (
    df["REVIEW_TEXT"]
    .str.split()
    .str.len()
)


# -----------------------------------
# Overall statistics
# -----------------------------------

print("\n" + "=" * 50)
print("OVERALL REVIEW STATISTICS")
print("=" * 50)

print(
    df[
        ["review_length", "word_count"]
    ].describe()
)


# -----------------------------------
# Statistics by label
# -----------------------------------

print("\n" + "=" * 50)
print("STATISTICS BY LABEL")
print("=" * 50)

label_stats = df.groupby("label_name")[
    ["review_length", "word_count", "RATING"]
].mean()

print(label_stats)


# -----------------------------------
# Rating by label
# -----------------------------------

print("\n" + "=" * 50)
print("RATING DISTRIBUTION BY LABEL")
print("=" * 50)

rating_by_label = pd.crosstab(
    df["RATING"],
    df["label_name"],
    normalize="index"
) * 100

print(rating_by_label.round(2))


# -----------------------------------
# Verified purchase by label
# -----------------------------------

print("\n" + "=" * 50)
print("VERIFIED PURCHASE BY LABEL")
print("=" * 50)

verified_by_label = pd.crosstab(
    df["VERIFIED_PURCHASE"],
    df["label_name"],
    normalize="columns"
) * 100

print(verified_by_label.round(2))


# -----------------------------------
# Category distribution
# -----------------------------------

print("\n" + "=" * 50)
print("PRODUCT CATEGORIES")
print("=" * 50)

print(
    df["PRODUCT_CATEGORY"]
    .value_counts()
)


# -----------------------------------
# Very short reviews
# -----------------------------------

print("\n" + "=" * 50)
print("VERY SHORT REVIEWS")
print("=" * 50)

print(
    f"Reviews with fewer than 10 words: "
    f"{(df['word_count'] < 10).sum()}"
)

print(
    f"Reviews with fewer than 5 words: "
    f"{(df['word_count'] < 5).sum()}"
)


# -----------------------------------
# Example deceptive reviews
# -----------------------------------

print("\n" + "=" * 50)
print("SAMPLE DECEPTIVE REVIEWS")
print("=" * 50)

for review in df[
    df["label_name"] == "deceptive"
]["REVIEW_TEXT"].head(3):

    print("\n-", review)


# -----------------------------------
# Example truthful reviews
# -----------------------------------

print("\n" + "=" * 50)
print("SAMPLE TRUTHFUL REVIEWS")
print("=" * 50)

for review in df[
    df["label_name"] == "truthful"
]["REVIEW_TEXT"].head(3):

    print("\n-", review)
    