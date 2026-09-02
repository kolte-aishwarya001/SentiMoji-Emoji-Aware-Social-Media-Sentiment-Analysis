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

INPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "emoji_enhanced_train.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "transformer_train_8000.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 8000
RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8"
)

print("Original records:", len(df))


# ============================================================
# CLEAN DATA
# ============================================================

df = df[
    ["text", "sentiment", "source"]
].copy()

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
)

df["sentiment"] = (
    df["sentiment"]
    .str.lower()
    .str.strip()
)


# ============================================================
# STRATIFIED SAMPLING
# ============================================================

print("\nCreating stratified sample...")

sample_df, _ = train_test_split(
    df,
    train_size=SAMPLE_SIZE,
    stratify=df["sentiment"],
    random_state=RANDOM_STATE
)


# ============================================================
# SHUFFLE
# ============================================================

sample_df = sample_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# SAVE DATASET
# ============================================================

sample_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("TRANSFORMER 8000 DATASET")
print("========================================")

print("Total records:", len(sample_df))

print("\nSentiment distribution:")
print(sample_df["sentiment"].value_counts())

print("\nPercentage distribution:")
print(
    sample_df["sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n========================================")
print("DONE")
print("========================================")