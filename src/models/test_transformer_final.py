import os
import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/distilbert_emoji_sentiment_8000"

TEST_PATH = "dataset/final_test.csv"

MAX_LENGTH = 64

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("FINAL DISTILBERT TEST")
print("=" * 60)

print("Device:", device)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(
    TEST_PATH,
    encoding="utf-8"
)

print("Test dataset shape:", test_df.shape)

print("\nColumns:")
print(test_df.columns.tolist())


# ============================================================
# CLEAN TEST DATA
# ============================================================

test_df = test_df[["text", "sentiment"]].copy()

test_df["text"] = (
    test_df["text"]
    .fillna("")
    .astype(str)
)

test_df["sentiment"] = (
    test_df["sentiment"]
    .str.lower()
    .str.strip()
)

test_df = test_df[
    test_df["sentiment"].isin(LABEL2ID.keys())
].copy()

test_df["label"] = test_df["sentiment"].map(LABEL2ID)


print("\nFinal test records:", len(test_df))

print("\nTest sentiment distribution:")
print(
    test_df["sentiment"].value_counts()
)


# ============================================================
# CREATE DATASET
# ============================================================

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]],
    preserve_index=False
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Tokenizer loaded successfully.")


# ============================================================
# TOKENIZATION
# ============================================================

print("\nTokenizing test dataset...")


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing test data"
)


# ============================================================
# REMOVE TEXT
# ============================================================

test_dataset = test_dataset.remove_columns(
    ["text"]
)

test_dataset.set_format("torch")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained DistilBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model
)


# ============================================================
# PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("RUNNING FINAL TEST PREDICTION")
print("=" * 60)

predictions = trainer.predict(
    test_dataset
)


# ============================================================
# PREDICT LABELS
# ============================================================

predicted_labels = np.argmax(
    predictions.predictions,
    axis=-1
)

true_labels = predictions.label_ids


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test Accuracy:       {accuracy:.4f}")
print(f"Test Precision:      {precision:.4f}")
print(f"Test Recall:         {recall:.4f}")
print(f"Test Weighted F1:    {f1:.4f}")

print(f"Macro Precision:     {macro_precision:.4f}")
print(f"Macro Recall:        {macro_recall:.4f}")
print(f"Macro F1:            {macro_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "negative",
            "neutral",
            "positive"
        ],
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print(cm)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

result_df = test_df.copy()

result_df["predicted_sentiment"] = [
    ID2LABEL[x]
    for x in predicted_labels
]

result_df["correct"] = (
    result_df["sentiment"]
    == result_df["predicted_sentiment"]
)


output_file = (
    "dataset/distilbert_test_predictions.csv"
)

result_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)


print("\n")
print("Predictions saved to:")
print(
    os.path.abspath(output_file)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TRANSFORMER TEST COMPLETED")
print("=" * 60)