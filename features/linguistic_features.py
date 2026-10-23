import re
import numpy as np
import pandas as pd


# --------------------------------------------------
# Basic word extraction
# --------------------------------------------------

def get_words(text):
    return re.findall(r"\b[a-zA-Z]+\b", str(text).lower())


# --------------------------------------------------
# Sentence extraction
# --------------------------------------------------

def get_sentences(text):
    sentences = re.split(r"[.!?]+", str(text))
    return [s.strip() for s in sentences if s.strip()]


# --------------------------------------------------
# Linguistic feature extraction
# --------------------------------------------------

def extract_linguistic_features(text):

    text = str(text)

    words = get_words(text)
    sentences = get_sentences(text)

    word_count = len(words)
    sentence_count = len(sentences)

    # Avoid division by zero
    if word_count == 0:
        word_count = 1

    if sentence_count == 0:
        sentence_count = 1

    # --------------------------------------------------
    # Word statistics
    # --------------------------------------------------

    word_lengths = [len(word) for word in words]

    avg_word_length = (
        np.mean(word_lengths)
        if word_lengths
        else 0
    )

    unique_word_ratio = (
        len(set(words)) / word_count
        if word_count > 0
        else 0
    )

    # --------------------------------------------------
    # Repetition
    # --------------------------------------------------

    word_counts = pd.Series(words).value_counts()

    repeated_words = sum(
        count - 1
        for count in word_counts
        if count > 1
    )

    repeated_word_ratio = (
        repeated_words / word_count
        if word_count > 0
        else 0
    )

    # --------------------------------------------------
    # Punctuation
    # --------------------------------------------------

    exclamation_count = text.count("!")
    question_count = text.count("?")

    comma_count = text.count(",")
    period_count = text.count(".")

    # --------------------------------------------------
    # Uppercase
    # --------------------------------------------------

    alphabetic_chars = [
        char for char in text
        if char.isalpha()
    ]

    uppercase_chars = [
        char for char in alphabetic_chars
        if char.isupper()
    ]

    uppercase_ratio = (
        len(uppercase_chars) / len(alphabetic_chars)
        if alphabetic_chars
        else 0
    )

    # --------------------------------------------------
    # Digits
    # --------------------------------------------------

    digit_count = sum(char.isdigit() for char in text)

    digit_ratio = (
        digit_count / len(text)
        if len(text) > 0
        else 0
    )

    # --------------------------------------------------
    # First-person language
    # --------------------------------------------------

    first_person_words = {
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "we",
        "us",
        "our",
        "ours",
        "ourselves"
    }

    first_person_count = sum(
        1 for word in words
        if word in first_person_words
    )

    first_person_ratio = (
        first_person_count / word_count
        if word_count > 0
        else 0
    )

    # --------------------------------------------------
    # Promotional language
    # --------------------------------------------------

    promotional_words = {
        "buy",
        "best",
        "amazing",
        "excellent",
        "perfect",
        "awesome",
        "fantastic",
        "wonderful",
        "incredible",
        "must",
        "highly",
        "recommend",
        "recommended",
        "worth",
        "deal",
        "sale",
        "cheap",
        "love",
        "loved"
    }

    promotional_word_count = sum(
        1 for word in words
        if word in promotional_words
    )

    promotional_word_ratio = (
        promotional_word_count / word_count
        if word_count > 0
        else 0
    )

    # --------------------------------------------------
    # Superlative language
    # --------------------------------------------------

    superlative_words = {
        "best",
        "greatest",
        "perfect",
        "amazing",
        "incredible",
        "excellent",
        "fantastic",
        "wonderful",
        "worst",
        "terrible"
    }

    superlative_count = sum(
        1 for word in words
        if word in superlative_words
    )

    # --------------------------------------------------
    # Sentiment word counts
    # --------------------------------------------------

    positive_words = {
        "good",
        "great",
        "excellent",
        "amazing",
        "awesome",
        "perfect",
        "love",
        "loved",
        "wonderful",
        "fantastic",
        "happy",
        "best",
        "nice",
        "beautiful"
    }

    negative_words = {
        "bad",
        "poor",
        "terrible",
        "awful",
        "worst",
        "hate",
        "hated",
        "horrible",
        "disappointing",
        "disappointed",
        "useless",
        "broken"
    }

    positive_count = sum(
        1 for word in words
        if word in positive_words
    )

    negative_count = sum(
        1 for word in words
        if word in negative_words
    )

    positive_word_ratio = positive_count / word_count
    negative_word_ratio = negative_count / word_count

    # --------------------------------------------------
    # Sentence length
    # --------------------------------------------------

    avg_sentence_length = (
        word_count / sentence_count
    )

    # --------------------------------------------------
    # Return all features
    # --------------------------------------------------

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "unique_word_ratio": unique_word_ratio,
        "repeated_word_ratio": repeated_word_ratio,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
        "first_person_count": first_person_count,
        "first_person_ratio": first_person_ratio,
        "promotional_word_count": promotional_word_count,
        "promotional_word_ratio": promotional_word_ratio,
        "superlative_count": superlative_count,
        "positive_word_ratio": positive_word_ratio,
        "negative_word_ratio": negative_word_ratio
    }


# --------------------------------------------------
# Test the extractor
# --------------------------------------------------

if __name__ == "__main__":

    example = """
    This product is AMAZING!!! I absolutely LOVE it.
    Best purchase ever! I highly recommend this product!!!
    """

    features = extract_linguistic_features(example)

    print("\nLinguistic Features")
    print("=" * 40)

    for name, value in features.items():
        print(f"{name:30} {value}")