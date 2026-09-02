import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "distilbert_emoji_sentiment_8000"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LABEL MAPPING
# ============================================================

id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("EMOJI IMPACT ANALYSIS")
print("=" * 70)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Tokenizer loaded.")

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    predicted_id = torch.argmax(
        probabilities
    ).item()

    sentiment = id2label[predicted_id]

    confidence = (
        probabilities[predicted_id].item() * 100
    )

    return sentiment, round(confidence, 2)


# ============================================================
# TEST CASES
# ============================================================

test_pairs = [

    (
        "I love this movie",
        "I love this movie 😍❤️"
    ),

    (
        "This product is amazing",
        "This product is amazing 🔥🥰"
    ),

    (
        "I hate this service",
        "I hate this service 😡💔"
    ),

    (
        "This is terrible",
        "This is terrible 😭😠"
    ),

    (
        "I am okay",
        "I am okay 😂"
    ),

    (
        "I am okay",
        "I am okay 😭"
    ),

    (
        "The meeting was fine",
        "The meeting was fine 🙂"
    ),

    (
        "The meeting was fine",
        "The meeting was fine 😒"
    )
]


# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []

print("\n")
print("=" * 70)
print("RUNNING EMOJI IMPACT EXPERIMENT")
print("=" * 70)


for i, (plain_text, emoji_text) in enumerate(
    test_pairs,
    start=1
):

    print(f"\nTEST CASE {i}")
    print("-" * 70)

    plain_prediction, plain_confidence = predict(
        plain_text
    )

    emoji_prediction, emoji_confidence = predict(
        emoji_text
    )

    prediction_changed = (
        plain_prediction != emoji_prediction
    )

    confidence_change = round(
        emoji_confidence - plain_confidence,
        2
    )

    print("Without Emoji:")
    print(plain_text)

    print(
        f"Prediction: {plain_prediction}"
    )

    print(
        f"Confidence: {plain_confidence}%"
    )

    print("\nWith Emoji:")
    print(emoji_text)

    print(
        f"Prediction: {emoji_prediction}"
    )

    print(
        f"Confidence: {emoji_confidence}%"
    )

    print(
        f"\nPrediction Changed: {prediction_changed}"
    )

    print(
        f"Confidence Change: {confidence_change}%"
    )

    results.append({

        "plain_text": plain_text,

        "emoji_text": emoji_text,

        "plain_prediction": plain_prediction,

        "plain_confidence": plain_confidence,

        "emoji_prediction": emoji_prediction,

        "emoji_confidence": emoji_confidence,

        "prediction_changed": prediction_changed,

        "confidence_change": confidence_change
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

result_df = pd.DataFrame(results)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    RESULTS_DIR,
    "emoji_impact_analysis.csv"
)

result_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

total_cases = len(result_df)

changed_cases = result_df[
    "prediction_changed"
].sum()

unchanged_cases = (
    total_cases - changed_cases
)

change_percentage = round(
    (changed_cases / total_cases) * 100,
    2
)


print("\n")
print("=" * 70)
print("EMOJI IMPACT SUMMARY")
print("=" * 70)

print(
    f"Total test cases: {total_cases}"
)

print(
    f"Prediction changed: {changed_cases}"
)

print(
    f"Prediction unchanged: {unchanged_cases}"
)

print(
    f"Prediction change percentage: {change_percentage}%"
)

print("\nResults saved to:")

print(output_file)

print("\n")
print("=" * 70)
print("EMOJI IMPACT ANALYSIS COMPLETED")
print("=" * 70)