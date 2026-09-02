import os
import sys

import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Emoji-Based Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "distilbert_emoji_sentiment_8000"
)

PREPROCESSING_PATH = os.path.join(
    BASE_DIR,
    "src",
    "preprocessing"
)

sys.path.insert(
    0,
    PREPROCESSING_PATH
)

from emoji_processor import analyze_emojis


# ============================================================
# LABEL MAPPING
# ============================================================

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to("cpu")
    model.eval()

    return tokenizer, model


# ============================================================
# PREDICTION
# ============================================================

def predict_sentiment(
    text,
    tokenizer,
    model
):

    emoji_info = analyze_emojis(text)

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

    sentiment = ID2LABEL[
        predicted_id
    ]

    confidence = (
        probabilities[predicted_id].item()
        * 100
    )

    scores = {
        "negative": probabilities[0].item() * 100,
        "neutral": probabilities[1].item() * 100,
        "positive": probabilities[2].item() * 100
    }

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "emojis": emoji_info["emojis"],
        "emoji_count": emoji_info["emoji_count"],
        "scores": scores
    }


# ============================================================
# HEADER
# ============================================================

st.title("😊 Emoji-Based Sentiment Analysis")

st.markdown(
    """
### Analyze the sentiment of social media text

This application uses a **DistilBERT Transformer model**
trained on an emoji-enhanced sentiment dataset.
"""
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    tokenizer, model = load_model()

except Exception as e:

    st.error(
        f"Unable to load the trained model: {e}"
    )

    st.stop()


# ============================================================
# INPUT
# ============================================================

st.subheader("Enter Text")

text = st.text_area(
    "Write a social media comment, tweet, or sentence:",
    placeholder="Example: I absolutely love this movie 😍🔥",
    height=150
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Sentiment",
    type="primary"
):

    if not text.strip():

        st.warning(
            "Please enter some text first."
        )

    else:

        with st.spinner(
            "Analyzing sentiment..."
        ):

            result = predict_sentiment(
                text,
                tokenizer,
                model
            )


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "Sentiment Analysis Result"
        )

        sentiment = result["sentiment"]

        confidence = result["confidence"]

        # ----------------------------------------------------
        # Main result
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Sentiment",
                sentiment.upper()
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        with col3:

            st.metric(
                "Emoji Count",
                result["emoji_count"]
            )


        # ====================================================
        # EMOJI INFORMATION
        # ====================================================

        st.subheader(
            "Emoji Analysis"
        )

        if result["emoji_count"] > 0:

            emoji_text = " ".join(
                result["emojis"]
            )

            st.write(
                f"**Detected Emojis:** {emoji_text}"
            )

        else:

            st.write(
                "No emojis detected."
            )


        # ====================================================
        # SENTIMENT PROBABILITIES
        # ====================================================

        st.subheader(
            "Sentiment Probabilities"
        )

        scores = result["scores"]

        st.write(
            f"Negative: {scores['negative']:.2f}%"
        )

        st.progress(
            min(scores["negative"] / 100, 1.0)
        )

        st.write(
            f"Neutral: {scores['neutral']:.2f}%"
        )

        st.progress(
            min(scores["neutral"] / 100, 1.0)
        )

        st.write(
            f"Positive: {scores['positive']:.2f}%"
        )

        st.progress(
            min(scores["positive"] / 100, 1.0)
        )


        # ====================================================
        # ANALYZED TEXT
        # ====================================================

        st.subheader(
            "Analyzed Text"
        )

        st.info(text)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Project Information"
)

st.sidebar.write(
    """
**Model:** DistilBERT

**Task:** 3-Class Sentiment Classification

**Classes:**
- Negative
- Neutral
- Positive

**Training Dataset:** 8,000 records

**Final Test Dataset:** 12,284 records

**Test Accuracy:** 67.67%
"""
)

st.sidebar.divider()

st.sidebar.write(
    "Emoji-Based Sentiment Analysis Project"
)