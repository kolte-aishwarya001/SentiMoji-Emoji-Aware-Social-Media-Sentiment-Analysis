import os
import sys

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


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


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "distilbert_emoji_sentiment_8000"
)


# ============================================================
# IMPORT EMOJI PROCESSOR
# ============================================================

PREPROCESSING_DIR = os.path.join(
    BASE_DIR,
    "src",
    "preprocessing"
)

sys.path.insert(
    0,
    PREPROCESSING_DIR
)

from emoji_processor import analyze_emojis


# ============================================================
# LABELS
# ============================================================

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("EMOJI-AWARE SENTIMENT PREDICTION SYSTEM")
print("=" * 60)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Tokenizer loaded.")

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):

    # --------------------------------------------------------
    # Emoji analysis
    # --------------------------------------------------------

    emoji_info = analyze_emojis(text)

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_id = torch.argmax(
        probabilities
    ).item()

    sentiment = ID2LABEL[
        predicted_id
    ]

    confidence = (
        probabilities[predicted_id].item()
        * 100
    )

    # --------------------------------------------------------
    # All probabilities
    # --------------------------------------------------------

    scores = {

        "negative": round(
            probabilities[0].item() * 100,
            2
        ),

        "neutral": round(
            probabilities[1].item() * 100,
            2
        ),

        "positive": round(
            probabilities[2].item() * 100,
            2
        )
    }

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {

        "text": text,

        "sentiment": sentiment,

        "confidence": round(
            confidence,
            2
        ),

        "emojis": emoji_info["emojis"],

        "emoji_count": emoji_info["emoji_count"],

        "has_emoji": emoji_info["has_emoji"],

        "scores": scores
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    test_texts = [

        "I absolutely love this movie 😍🔥",

        "This product is terrible 😡💔",

        "The meeting is scheduled for tomorrow.",

        "I am so happy today 🥰❤️"
    ]

    for text in test_texts:

        print("\n")
        print("=" * 60)

        result = predict_sentiment(text)

        print("TEXT:")
        print(result["text"])

        print("\nDETECTED EMOJIS:")
        print(result["emojis"])

        print("\nEMOJI COUNT:")
        print(result["emoji_count"])

        print("\nSENTIMENT:")
        print(
            result["sentiment"].upper()
        )

        print("\nCONFIDENCE:")
        print(
            f"{result['confidence']}%"
        )

        print("\nSENTIMENT SCORES:")

        for label, score in result["scores"].items():

            print(
                f"{label}: {score}%"
            )