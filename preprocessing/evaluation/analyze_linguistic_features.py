import pandas as pd
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

feature_rows = []

for text in df["REVIEW_TEXT"]:
    feature_rows.append(
        extract_linguistic_features(text)
    )

features_df = pd.DataFrame(feature_rows)

print("Features extracted:", features_df.shape)


# --------------------------------------------------
# 3. Combine with labels
# --------------------------------------------------

features_df["label"] = df["label"].values

features_df["label_name"] = features_df["label"].map({
    0: "Truthful",
    1: "Deceptive"
})


# --------------------------------------------------
# 4. Compare classes
# --------------------------------------------------

numeric_features = [
    column
    for column in features_df.columns
    if column not in ["label", "label_name"]
]

comparison = (
    features_df
    .groupby("label_name")[numeric_features]
    .mean()
    .T
)

print("\n" + "=" * 70)
print("LINGUISTIC FEATURE COMPARISON")
print("=" * 70)

print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))


# --------------------------------------------------
# 5. Difference between classes
# --------------------------------------------------

comparison["absolute_difference"] = (
    comparison["Deceptive"] -
    comparison["Truthful"]
).abs()

comparison = comparison.sort_values(
    "absolute_difference",
    ascending=False
)

print("\n" + "=" * 70)
print("FEATURES WITH LARGEST CLASS DIFFERENCES")
print("=" * 70)

print(comparison.to_string(
    float_format=lambda x: f"{x:.4f}"
))


# --------------------------------------------------
# 6. Save extracted features
# --------------------------------------------------

output = pd.concat(
    [
        df.reset_index(drop=True),
        features_df.drop(
            columns=["label", "label_name"]
        )
    ],
    axis=1
)

output.to_csv(
    "data/linguistic_features.csv",
    index=False
)

print("\nSaved:")
print("data/linguistic_features.csv")