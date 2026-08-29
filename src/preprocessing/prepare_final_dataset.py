import pandas as pd
import os

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_FILE = os.path.join(DATASET_DIR, "tweet_eval_train.csv")
VALIDATION_FILE = os.path.join(DATASET_DIR, "tweet_eval_validation.csv")
TEST_FILE = os.path.join(DATASET_DIR, "tweet_eval_test.csv")
CUSTOM_FILE = os.path.join(DATASET_DIR, "social_media_comments.csv")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE, encoding="utf-8")
validation_df = pd.read_csv(VALIDATION_FILE, encoding="utf-8")
test_df = pd.read_csv(TEST_FILE, encoding="utf-8")
custom_df = pd.read_csv(CUSTOM_FILE, encoding="utf-8")


print("\nOriginal datasets:")
print("Train:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))
print("Custom:", len(custom_df))


# --------------------------------------------------
# CONVERT LABELS
# --------------------------------------------------

label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

train_df["sentiment"] = train_df["label"].map(label_map)
validation_df["sentiment"] = validation_df["label"].map(label_map)
test_df["sentiment"] = test_df["label"].map(label_map)


# --------------------------------------------------
# CUSTOM DATASET
# --------------------------------------------------

custom_df["sentiment"] = custom_df["sentiment"].str.lower()

custom_df["text"] = custom_df["comment"]

custom_df = custom_df[["text", "sentiment"]]

print("\nCustom dataset:")
print(custom_df.head())
print(custom_df["sentiment"].value_counts())


# --------------------------------------------------
# PREPARE TWEET EVAL TRAIN DATA
# --------------------------------------------------

train_final = train_df[["text", "sentiment"]].copy()


# --------------------------------------------------
# ADD CUSTOM EMOJI DATA
# --------------------------------------------------

train_final = pd.concat(
    [train_final, custom_df],
    ignore_index=True
)


# --------------------------------------------------
# REMOVE DUPLICATES
# --------------------------------------------------

train_final = train_final.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)


# --------------------------------------------------
# SHUFFLE
# --------------------------------------------------

train_final = train_final.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

validation_final = validation_df[
    ["text", "sentiment"]
].copy()


# --------------------------------------------------
# TEST
# --------------------------------------------------

test_final = test_df[
    ["text", "sentiment"]
].copy()


# --------------------------------------------------
# SAVE FINAL DATASETS
# --------------------------------------------------

train_final.to_csv(
    os.path.join(DATASET_DIR, "final_train.csv"),
    index=False,
    encoding="utf-8"
)

validation_final.to_csv(
    os.path.join(DATASET_DIR, "final_validation.csv"),
    index=False,
    encoding="utf-8"
)

test_final.to_csv(
    os.path.join(DATASET_DIR, "final_test.csv"),
    index=False,
    encoding="utf-8"
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n===================================")
print("FINAL DATASET CREATED")
print("===================================")

print("\nTRAIN")
print("Total:", len(train_final))
print(train_final["sentiment"].value_counts())

print("\nVALIDATION")
print("Total:", len(validation_final))
print(validation_final["sentiment"].value_counts())

print("\nTEST")
print("Total:", len(test_final))
print(test_final["sentiment"].value_counts())

print("\nFiles created:")
print("final_train.csv")
print("final_validation.csv")
print("final_test.csv")