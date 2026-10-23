import pandas as pd
from pathlib import Path


# -----------------------------
# 1. File paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "amazon_reviews.txt"
OUTPUT_FILE = BASE_DIR / "data" / "clean_reviews.csv"


# -----------------------------
# 2. Load dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE, sep="\t")

print(f"Original dataset shape: {df.shape}")


# -----------------------------
# 3. Check required columns
# -----------------------------

required_columns = [
    "DOC_ID",
    "LABEL",
    "RATING",
    "VERIFIED_PURCHASE",
    "PRODUCT_CATEGORY",
    "PRODUCT_ID",
    "PRODUCT_TITLE",
    "REVIEW_TITLE",
    "REVIEW_TEXT",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# -----------------------------
# 4. Remove rows with missing
#    review text or label
# -----------------------------

before = len(df)

df = df.dropna(
    subset=["REVIEW_TEXT", "LABEL"]
).copy()

print(
    f"Removed missing text/labels: "
    f"{before - len(df)}"
)


# -----------------------------
# 5. Clean review text
# -----------------------------

df["REVIEW_TEXT"] = (
    df["REVIEW_TEXT"]
    .astype(str)
    .str.strip()
)

# Remove empty reviews
df = df[df["REVIEW_TEXT"].str.len() > 0].copy()


# -----------------------------
# 6. Remove duplicate reviews
# -----------------------------

before = len(df)

df = df.drop_duplicates(
    subset=["REVIEW_TEXT"]
).copy()

print(
    f"Removed duplicate reviews: "
    f"{before - len(df)}"
)


# -----------------------------
# 7. Validate labels
# -----------------------------

valid_labels = {
    "__label1__",
    "__label2__"
}

invalid_labels = set(df["LABEL"].unique()) - valid_labels

if invalid_labels:
    raise ValueError(
        f"Unexpected labels found: {invalid_labels}"
    )


# Convert labels:

# __label1__ = deceptive
# __label2__ = truthful

df["label"] = df["LABEL"].map({
    "__label1__": 1,
    "__label2__": 0
})


# Human-readable label
df["label_name"] = df["label"].map({
    0: "truthful",
    1: "deceptive"
})


# -----------------------------
# 8. Validate ratings
# -----------------------------

df["RATING"] = pd.to_numeric(
    df["RATING"],
    errors="coerce"
)

df = df[
    df["RATING"].between(1, 5)
].copy()


# -----------------------------
# 9. Normalize verified purchase
# -----------------------------

df["VERIFIED_PURCHASE"] = (
    df["VERIFIED_PURCHASE"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# -----------------------------
# 10. Save cleaned dataset
# -----------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# -----------------------------
# 11. Print final statistics
# -----------------------------

print("\nCleaning completed!")

print(
    f"Final dataset shape: {df.shape}"
)

print("\nLabel distribution:")

print(
    df["label_name"].value_counts()
)

print("\nVerified purchase distribution:")

print(
    df["VERIFIED_PURCHASE"].value_counts()
)

print("\nRating distribution:")

print(
    df["RATING"].value_counts()
    .sort_index()
)

print(
    f"\nClean dataset saved to:\n"
    f"{OUTPUT_FILE}"
)