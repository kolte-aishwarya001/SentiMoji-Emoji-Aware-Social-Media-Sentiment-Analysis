import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "emoji_enhanced_train.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "transformer_train_30000.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TRAINING_SIZE = 30000
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading emoji-enhanced dataset...")

df = pd.read_csv(
    TRAIN_FILE,
    encoding="utf-8"
)

print("\nOriginal dataset:")
print("Total records:", len(df))

print("\nOriginal sentiment distribution:")
print(df["sentiment"].value_counts())


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

df = df.dropna(
    subset=["text", "sentiment"]
)

df["text"] = df["text"].astype(str)

print("\nAfter cleaning:")
print("Total records:", len(df))


# ============================================================
# STRATIFIED SAMPLING
# ============================================================

print("\nCreating stratified Transformer dataset...")

transformer_train, _ = train_test_split(
    df,
    train_size=TRAINING_SIZE,
    stratify=df["sentiment"],
    random_state=RANDOM_STATE
)


# ============================================================
# SHUFFLE
# ============================================================

transformer_train = transformer_train.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

transformer_train.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("TRANSFORMER DATASET CREATED")
print("========================================")

print(
    "Total training records:",
    len(transformer_train)
)

print("\nSentiment distribution:")

print(
    transformer_train["sentiment"].value_counts()
)

print("\nPercentage distribution:")

print(
    transformer_train["sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nEmoji coverage:")

# ============================================================
# EMOJI COVERAGE
# ============================================================

import emoji

def contains_emoji(text):
    return bool(emoji.emoji_list(str(text)))


emoji_mask = transformer_train["text"].apply(
    contains_emoji
)

print(
    "Records containing emojis:",
    emoji_mask.sum()
)

print(
    "Emoji coverage:",
    round(
        emoji_mask.mean() * 100,
        2
    ),
    "%"
)


# ============================================================
# SAVE
# ============================================================

print("\nSaved to:")

print(OUTPUT_FILE)

print("\n========================================")
print("DONE")
print("========================================")

print("\n========================================")
print("DONE")
print("========================================")