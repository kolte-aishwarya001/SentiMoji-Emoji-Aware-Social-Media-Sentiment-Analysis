import os
import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "dataset/transformer_train_8000.csv"

MODEL_NAME = "distilbert-base-uncased"

OUTPUT_DIR = "models/distilbert_emoji_sentiment_8000"

MAX_LENGTH = 64

NUM_LABELS = 3

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

print("=" * 50)
print("TRANSFORMER TRAINING")
print("=" * 50)

print("Device:", device)

if device == "cpu":
    print("CPU training enabled.")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())


# ============================================================
# CLEAN DATA
# ============================================================

df = df[["text", "sentiment"]].copy()

df["text"] = df["text"].fillna("").astype(str)

df["sentiment"] = df["sentiment"].str.lower().str.strip()

df = df[df["sentiment"].isin(LABEL2ID.keys())]

df["label"] = df["sentiment"].map(LABEL2ID)

print("\nDataset after cleaning:", df.shape)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating train/validation split...")

train_df, val_df = train_test_split(
    df,
    test_size=0.10,
    random_state=42,
    stratify=df["label"]
)

print("Training records:", len(train_df))
print("Validation records:", len(val_df))


# ============================================================
# CONVERT TO HUGGINGFACE DATASET
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "label"]],
    preserve_index=False
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully.")


# ============================================================
# TOKENIZATION
# ============================================================

print("\nTokenizing training data...")

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing training dataset"
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing validation dataset"
)


# ============================================================
# REMOVE TEXT COLUMN
# ============================================================

train_dataset = train_dataset.remove_columns(["text"])

val_dataset = val_dataset.remove_columns(["text"])


# ============================================================
# FORMAT FOR PYTORCH
# ============================================================

train_dataset.set_format("torch")

val_dataset.set_format("torch")


print("\nTokenization completed.")

print("Training dataset:")
print(train_dataset)

print("\nValidation dataset:")
print(val_dataset)


# ============================================================
# LOAD DISTILBERT
# ============================================================

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

print("Model loaded successfully.")


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=1,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    gradient_accumulation_steps=2,

    learning_rate=2e-5,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=100,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=1,

    report_to="none",

    dataloader_num_workers=0,

    use_cpu=True
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics
)


# ============================================================
# START TRAINING
# ============================================================

print("\n")
print("=" * 50)
print("STARTING DISTILBERT TRAINING")
print("=" * 50)

trainer.train()


# ============================================================
# VALIDATION
# ============================================================

print("\n")
print("=" * 50)
print("VALIDATION RESULTS")
print("=" * 50)

results = trainer.evaluate()

print(results)


# ============================================================
# DETAILED VALIDATION REPORT
# ============================================================

print("\nGenerating detailed validation report...")

predictions = trainer.predict(val_dataset)

predicted_labels = np.argmax(
    predictions.predictions,
    axis=-1
)

true_labels = predictions.label_ids

print("\nClassification Report:")

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

print("\nConfusion Matrix:")

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving trained model...")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nModel saved successfully!")

print(
    "Location:",
    os.path.abspath(OUTPUT_DIR)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 50)
print("TRANSFORMER TRAINING COMPLETED")
print("=" * 50)