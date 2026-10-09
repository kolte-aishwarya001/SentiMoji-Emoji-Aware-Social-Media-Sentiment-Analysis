import os
import sys
import io

import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SentiMoji - Emoji-Based Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "distilbert_emoji_sentiment_8000"
)


# ============================================================
# PREPROCESSING PATH
# ============================================================

PREPROCESSING_PATH = os.path.join(
    BASE_DIR,
    "src",
    "preprocessing"
)


# ============================================================
# FUSION PATH
# ============================================================

FUSION_PATH = os.path.join(
    BASE_DIR,
    "src",
    "fusion"
)


# Add paths to Python path
sys.path.insert(
    0,
    PREPROCESSING_PATH
)

sys.path.insert(
    0,
    FUSION_PATH
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from emoji_processor import analyze_emojis

from emoji_fusion import (
    calculate_emoji_probabilities,
    fuse_text_and_emoji
)


# ============================================================
# LABEL MAPPING
# ============================================================

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


# ============================================================
# FUSION WEIGHTS
# ============================================================

TEXT_WEIGHT = 0.70
EMOJI_WEIGHT = 0.30


# ============================================================
# LOAD DISTILBERT MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    # CPU execution
    model.to("cpu")

    # Evaluation mode
    model.eval()

    return tokenizer, model


# ============================================================
# SINGLE TEXT PREDICTION WITH TEXT + EMOJI FUSION
# ============================================================
def predict_sentiment(text, tokenizer, model):
    """Return a consistent text, emoji, and fused sentiment result."""
    text = "" if text is None else str(text)

    # Emoji extraction and sentiment probabilities.
    emoji_info = analyze_emojis(text)
    detected_emojis = emoji_info.get("emojis", []) or []

    # DistilBERT text probabilities.
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    # The trained model uses label order: negative, neutral, positive.
    text_scores = {
        "negative": float(probabilities[0].item() * 100),
        "neutral": float(probabilities[1].item() * 100),
        "positive": float(probabilities[2].item() * 100),
    }

    # Emoji scorer returns probabilities on a 0-1 scale.
    emoji_scores_raw = calculate_emoji_probabilities(detected_emojis)
    emoji_scores = {
        "negative": float(emoji_scores_raw.get("negative", 0.0) * 100),
        "neutral": float(emoji_scores_raw.get("neutral", 0.0) * 100),
        "positive": float(emoji_scores_raw.get("positive", 0.0) * 100),
    }

    # Do not apply an emoji signal when the text contains no emojis.
    if detected_emojis:
        fused_raw = fuse_text_and_emoji(
            text_scores,
            emoji_scores_raw,
            text_weight=TEXT_WEIGHT,
            emoji_weight=EMOJI_WEIGHT,
        )
        scores = {
            k: float(fused_raw[k])
            for k in ("negative", "neutral", "positive")
        }
        fusion_applied = True
    else:
        scores = text_scores.copy()
        fusion_applied = False

    sentiment = max(scores, key=scores.get)
    confidence = float(scores[sentiment])

    # Keep every key expected by the single-comment, multi-comment,
    # CSV/Excel, and evaluation UI paths.
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "emojis": detected_emojis,
        "emoji_count": int(emoji_info.get("emoji_count", len(detected_emojis))),
        "text_scores": text_scores,
        "emoji_scores": emoji_scores,
        "scores": scores,
        "fusion_applied": fusion_applied,
    }


# ============================================================
# BATCH PREDICTION WITH TEXT + EMOJI FUSION
# ============================================================

def predict_sentiment_batch(
    texts,
    tokenizer,
    model,
    batch_size=16
):
    """
    Perform batch inference using DistilBERT.

    Text model:
        DistilBERT

    Emoji model:
        Rule-based emoji sentiment scoring

    Fusion:
        70% Text + 30% Emoji
    """

    all_results = []

    # ========================================================
    # PROCESS DATA IN BATCHES
    # ========================================================

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch_texts = texts[
            start:start + batch_size
        ]


        # ====================================================
        # TOKENIZE ENTIRE BATCH
        # ====================================================

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )


        # ====================================================
        # DISTILBERT INFERENCE
        # ====================================================

        with torch.inference_mode():

            outputs = model(
                **inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )


        # ====================================================
        # PROCESS EACH RESULT IN BATCH
        # ========================================================

        for i, text in enumerate(
            batch_texts
        ):

            # -----------------------------------------------
            # TEXT MODEL SCORES
            # -----------------------------------------------

            text_scores = {

                "negative":
                    probabilities[i][0].item()
                    * 100,

                "neutral":
                    probabilities[i][1].item()
                    * 100,

                "positive":
                    probabilities[i][2].item()
                    * 100
            }


            # -----------------------------------------------
            # EMOJI ANALYSIS
            # -----------------------------------------------

            emoji_info = analyze_emojis(
                text
            )

            detected_emojis = (
                emoji_info["emojis"]
            )


            # -----------------------------------------------
            # EMOJI SCORES
            # -----------------------------------------------

            emoji_scores_raw = (
                calculate_emoji_probabilities(
                    detected_emojis
                )
            )


            emoji_scores = {

                "negative":
                    emoji_scores_raw[
                        "negative"
                    ] * 100,

                "neutral":
                    emoji_scores_raw[
                        "neutral"
                    ] * 100,

                "positive":
                    emoji_scores_raw[
                        "positive"
                    ] * 100
            }


            # -----------------------------------------------
            # TEXT + EMOJI FUSION
            # -----------------------------------------------

            if len(
                detected_emojis
            ) > 0:

                fused_scores = (
                    fuse_text_and_emoji(

                        text_scores,

                        emoji_scores_raw,

                        text_weight=TEXT_WEIGHT,

                        emoji_weight=EMOJI_WEIGHT
                    )
                )

                fusion_applied = True

            else:

                fused_scores = (
                    text_scores.copy()
                )

                fusion_applied = False


            # -----------------------------------------------
            # FINAL SENTIMENT
            # -----------------------------------------------

            final_scores = [

                fused_scores[
                    "negative"
                ],

                fused_scores[
                    "neutral"
                ],

                fused_scores[
                    "positive"
                ]
            ]


            predicted_id = int(
                np.argmax(
                    final_scores
                )
            )


            sentiment = ID2LABEL[
                predicted_id
            ]


            # -----------------------------------------------
            # CONFIDENCE
            # -----------------------------------------------

            confidence = fused_scores[
                sentiment
            ]


            # -----------------------------------------------
            # SAVE RESULT
            # -----------------------------------------------

            all_results.append({

                "sentiment":
                    sentiment,

                "confidence":
                    confidence,

                "emojis":
                    detected_emojis,

                "emoji_count":
                    emoji_info[
                        "emoji_count"
                    ],

                "text_scores":
                    text_scores,

                "emoji_scores":
                    emoji_scores,

                "scores":
                    fused_scores,

                "fusion_applied":
                    fusion_applied
            })


    return all_results

    # ========================================================
    # 1. EMOJI DETECTION
    # ========================================================

    emoji_info = analyze_emojis(text)

    detected_emojis = emoji_info["emojis"]


    # ========================================================
    # 2. TOKENIZATION
    # ========================================================

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )


    # ========================================================
    # 3. DISTILBERT PREDICTION
    # ========================================================

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]


    # ========================================================
    # 4. TEXT MODEL PROBABILITIES
    # ========================================================

    text_scores = {

        "negative":
            probabilities[0].item() * 100,

        "neutral":
            probabilities[1].item() * 100,

        "positive":
            probabilities[2].item() * 100
    }


    # ========================================================
    # 5. EMOJI MODEL PROBABILITIES
    # ========================================================

    emoji_scores_raw = calculate_emoji_probabilities(
        detected_emojis
    )


    emoji_scores = {

        "negative":
            emoji_scores_raw["negative"] * 100,

        "neutral":
            emoji_scores_raw["neutral"] * 100,

        "positive":
            emoji_scores_raw["positive"] * 100
    }


    # ========================================================
    # 6. TEXT + EMOJI FUSION
    # ========================================================

    if len(detected_emojis) > 0:

        fused_scores = fuse_text_and_emoji(
            text_scores,
            emoji_scores_raw,
            text_weight=TEXT_WEIGHT,
            emoji_weight=EMOJI_WEIGHT
        )

        fusion_applied = True

    else:

        # If there are no emojis, do not artificially
        # introduce a neutral emoji signal.

        fused_scores = text_scores.copy()

        fusion_applied = False


    # ========================================================
    # 7. FINAL PREDICTION
    # ========================================================

    final_probability_list = [

        fused_scores["negative"],

        fused_scores["neutral"],

        fused_scores["positive"]
    ]


    predicted_id = int(
        np.argmax(
            final_probability_list
        )
    )


    sentiment = ID2LABEL[
        predicted_id
    ]


    # ========================================================
    # 8. FINAL CONFIDENCE
    # ========================================================

    confidence = fused_scores[
        sentiment
    ]


    # ========================================================
    # 9. RETURN RESULT
    # ========================================================

    return {

        "sentiment":
            sentiment,

        "confidence":
            confidence,

        "emojis":
            detected_emojis,

        "emoji_count":
            emoji_info["emoji_count"],

        "text_scores":
            text_scores,

        "emoji_scores":
            emoji_scores,

        "scores":
            fused_scores,

        "fusion_applied":
            fusion_applied
    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "😊 SentiMoji - Emoji-Based Sentiment Analysis"
)


st.markdown(
    """
### Analyze the sentiment of social media text

This application uses a **fine-tuned DistilBERT Transformer**
combined with an **explicit emoji sentiment fusion layer**.

The system classifies text into:

- 🔴 Negative
- 🟡 Neutral
- 🟢 Positive

### Fusion Strategy

**70% Text Model + 30% Emoji Model**
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
# APPLICATION TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Single Comment",
        "📊 Multiple Comments",
        "📁 CSV / Excel Upload",
        "📈 Model Evaluation"
    ]
)


# ============================================================
# TAB 1 - SINGLE COMMENT ANALYSIS
# ============================================================

with tab1:

    st.subheader(
        "Enter a Social Media Comment"
    )


    text = st.text_area(
        "Write a comment, tweet, or sentence:",
        placeholder=(
            "Example: The product was awesome 😡"
        ),
        height=150
    )


    # ========================================================
    # ANALYZE SINGLE COMMENT
    # ========================================================

    if st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        key="single_analyze"
    ):

        if not text.strip():

            st.warning(
                "Please enter some text first."
            )

        else:

            with st.spinner(
                "Analyzing text and emojis..."
            ):

                result = predict_sentiment(
                    text,
                    tokenizer,
                    model
                )


            st.divider()


            # =================================================
            # MAIN RESULT
            # =================================================

            st.subheader(
                "🎯 Final Sentiment Analysis"
            )


            sentiment = result[
                "sentiment"
            ]

            confidence = result[
                "confidence"
            ]


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Predicted Sentiment",
                    sentiment.upper()
                )


            with col2:

                st.metric(
                    "Final Confidence",
                    f"{confidence:.2f}%"
                )


            with col3:

                st.metric(
                    "Emoji Count",
                    result["emoji_count"]
                )


            # =================================================
            # EMOJI INFORMATION
            # =================================================

            st.subheader(
                "😊 Emoji Analysis"
            )


            if result["emoji_count"] > 0:

                emoji_text = " ".join(
                    result["emojis"]
                )


                st.success(
                    f"Detected Emojis: {emoji_text}"
                )

            else:

                st.info(
                    "No emojis detected. "
                    "Text model prediction is used."
                )


            # =================================================
            # TEXT + EMOJI FUSION
            # =================================================

            st.subheader(
                "🔬 Text + Emoji Fusion"
            )


            fusion_col1, fusion_col2, fusion_col3 = (
                st.columns(3)
            )


            # -------------------------------------------------
            # TEXT MODEL
            # -------------------------------------------------

            with fusion_col1:

                st.markdown(
                    "### 📝 Text Model"
                )

                text_scores = result[
                    "text_scores"
                ]

                st.write(
                    f"🔴 Negative: "
                    f"{text_scores['negative']:.2f}%"
                )

                st.write(
                    f"🟡 Neutral: "
                    f"{text_scores['neutral']:.2f}%"
                )

                st.write(
                    f"🟢 Positive: "
                    f"{text_scores['positive']:.2f}%"
                )


            # -------------------------------------------------
            # EMOJI MODEL
            # -------------------------------------------------

            with fusion_col2:

                st.markdown(
                    "### 😊 Emoji Model"
                )

                emoji_scores = result[
                    "emoji_scores"
                ]

                if result["emoji_count"] > 0:

                    st.write(
                        f"🔴 Negative: "
                        f"{emoji_scores['negative']:.2f}%"
                    )

                    st.write(
                        f"🟡 Neutral: "
                        f"{emoji_scores['neutral']:.2f}%"
                    )

                    st.write(
                        f"🟢 Positive: "
                        f"{emoji_scores['positive']:.2f}%"
                    )

                else:

                    st.write(
                        "No emoji signal"
                    )


            # -------------------------------------------------
            # FINAL FUSION
            # -------------------------------------------------

            with fusion_col3:

                st.markdown(
                    "### 🎯 Final Fusion"
                )

                scores = result[
                    "scores"
                ]

                st.write(
                    f"🔴 Negative: "
                    f"{scores['negative']:.2f}%"
                )

                st.write(
                    f"🟡 Neutral: "
                    f"{scores['neutral']:.2f}%"
                )

                st.write(
                    f"🟢 Positive: "
                    f"{scores['positive']:.2f}%"
                )


            # =================================================
            # FUSION FORMULA
            # =================================================

            if result["fusion_applied"]:

                st.info(
                    """
### Fusion Formula

**Final Score = (0.70 × Text Score) + (0.30 × Emoji Score)**

Text contribution = **70%**

Emoji contribution = **30%**
"""
                )

            else:

                st.info(
                    """
No emoji was detected.

Therefore, the final prediction is based
entirely on the DistilBERT text prediction.
"""
                )


            # =================================================
            # FINAL SENTIMENT PROBABILITIES
            # =================================================

            st.subheader(
                "📊 Final Sentiment Probabilities"
            )


            # Negative

            st.write(
                f"🔴 Negative: "
                f"{scores['negative']:.2f}%"
            )

            st.progress(
                min(
                    scores["negative"] / 100,
                    1.0
                )
            )


            # Neutral

            st.write(
                f"🟡 Neutral: "
                f"{scores['neutral']:.2f}%"
            )

            st.progress(
                min(
                    scores["neutral"] / 100,
                    1.0
                )
            )


            # Positive

            st.write(
                f"🟢 Positive: "
                f"{scores['positive']:.2f}%"
            )

            st.progress(
                min(
                    scores["positive"] / 100,
                    1.0
                )
            )


            # =================================================
            # ANALYZED TEXT
            # =================================================

            st.subheader(
                "📝 Analyzed Text"
            )

            st.info(
                text
            )


# ============================================================
# TAB 2 - MULTIPLE COMMENT ANALYSIS
# ============================================================

with tab2:

    st.subheader(
        "📊 Analyze Multiple Comments"
    )


    st.markdown(
        """
Enter **one comment per line**.

Example:

Amazing product 😍

Very poor quality 😡

The product was awesome 😡

The product is okay 😐

I absolutely love it ❤️
"""
    )


    comments_text = st.text_area(
        "Enter multiple comments:",
        placeholder=(
            "Amazing product 😍\n"
            "Very poor quality 😡\n"
            "The product was awesome 😡\n"
            "The product is okay 😐\n"
            "I absolutely love it ❤️"
        ),
        height=250,
        key="multiple_comments"
    )


    # ========================================================
    # ANALYZE ALL COMMENTS
    # ========================================================

    if st.button(
        "🔍 Analyze All Comments",
        type="primary",
        key="multiple_analyze"
    ):

        if not comments_text.strip():

            st.warning(
                "Please enter at least one comment."
            )

        else:

            # ------------------------------------------------
            # SPLIT COMMENTS
            # ------------------------------------------------

            comments = [

                line.strip()

                for line
                in comments_text.split("\n")

                if line.strip()
            ]


            # ------------------------------------------------
            # REMOVE DUPLICATES
            # ------------------------------------------------

            comments = list(
                dict.fromkeys(
                    comments
                )
            )


            st.info(
                f"Total comments detected: "
                f"{len(comments)}"
            )


            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            results = []


            progress_bar = st.progress(
                0
            )


            for index, comment in enumerate(
                comments
            ):

                result = predict_sentiment(
                    comment,
                    tokenizer,
                    model
                )


                detected_emojis = " ".join(
                    result["emojis"]
                )


                text_scores = result[
                    "text_scores"
                ]


                emoji_scores = result[
                    "emoji_scores"
                ]


                final_scores = result[
                    "scores"
                ]


                results.append({

                    "Comment":
                        comment,

                    "Sentiment":
                        result[
                            "sentiment"
                        ].capitalize(),

                    "Confidence (%)":
                        round(
                            result[
                                "confidence"
                            ],
                            2
                        ),

                    "Text Negative (%)":
                        round(
                            text_scores[
                                "negative"
                            ],
                            2
                        ),

                    "Text Neutral (%)":
                        round(
                            text_scores[
                                "neutral"
                            ],
                            2
                        ),

                    "Text Positive (%)":
                        round(
                            text_scores[
                                "positive"
                            ],
                            2
                        ),

                    "Emoji Negative (%)":
                        round(
                            emoji_scores[
                                "negative"
                            ],
                            2
                        ),

                    "Emoji Neutral (%)":
                        round(
                            emoji_scores[
                                "neutral"
                            ],
                            2
                        ),

                    "Emoji Positive (%)":
                        round(
                            emoji_scores[
                                "positive"
                            ],
                            2
                        ),

                    "Final Negative (%)":
                        round(
                            final_scores[
                                "negative"
                            ],
                            2
                        ),

                    "Final Neutral (%)":
                        round(
                            final_scores[
                                "neutral"
                            ],
                            2
                        ),

                    "Final Positive (%)":
                        round(
                            final_scores[
                                "positive"
                            ],
                            2
                        ),

                    "Emojis":
                        detected_emojis,

                    "Emoji Count":
                        result[
                            "emoji_count"
                        ]
                })


                progress_bar.progress(
                    (index + 1)
                    / len(comments)
                )


            progress_bar.empty()


            # =================================================
            # DATAFRAME
            # =================================================

            results_df = pd.DataFrame(
                results
            )


            # =================================================
            # SENTIMENT COUNTS
            # =================================================

            positive_count = int(
                (
                    results_df[
                        "Sentiment"
                    ]
                    == "Positive"
                ).sum()
            )


            neutral_count = int(
                (
                    results_df[
                        "Sentiment"
                    ]
                    == "Neutral"
                ).sum()
            )


            negative_count = int(
                (
                    results_df[
                        "Sentiment"
                    ]
                    == "Negative"
                ).sum()
            )


            total_count = len(
                results_df
            )


            # =================================================
            # SENTIMENT PERCENTAGES
            # =================================================

            positive_percentage = (
                positive_count
                / total_count
            ) * 100


            neutral_percentage = (
                neutral_count
                / total_count
            ) * 100


            negative_percentage = (
                negative_count
                / total_count
            ) * 100


            # =================================================
            # OVERALL RESULT
            # =================================================

            st.divider()


            st.subheader(
                "📈 Overall Sentiment Analysis"
            )


            col1, col2, col3, col4 = (
                st.columns(4)
            )


            with col1:

                st.metric(
                    "😊 Positive",
                    f"{positive_percentage:.2f}%"
                )


            with col2:

                st.metric(
                    "😐 Neutral",
                    f"{neutral_percentage:.2f}%"
                )


            with col3:

                st.metric(
                    "😡 Negative",
                    f"{negative_percentage:.2f}%"
                )


            with col4:

                st.metric(
                    "💬 Total Comments",
                    total_count
                )


            # =================================================
            # SENTIMENT COUNTS
            # =================================================

            st.write(
                f"""
🟢 Positive: **{positive_count}**

🟡 Neutral: **{neutral_count}**

🔴 Negative: **{negative_count}**
"""
            )


            # =================================================
            # CHARTS
            # =================================================

            st.subheader(
                "📊 Sentiment Distribution"
            )


            chart_col1, chart_col2 = (
                st.columns(2)
            )


            # -------------------------------------------------
            # PIE CHART
            # -------------------------------------------------

            with chart_col1:

                labels = [
                    "Positive",
                    "Neutral",
                    "Negative"
                ]


                values = [
                    positive_count,
                    neutral_count,
                    negative_count
                ]


                fig1, ax1 = plt.subplots()


                ax1.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90
                )


                ax1.set_title(
                    "Overall Sentiment Distribution"
                )


                ax1.axis(
                    "equal"
                )


                st.pyplot(
                    fig1
                )


                plt.close(
                    fig1
                )


            # -------------------------------------------------
            # BAR CHART
            # -------------------------------------------------

            with chart_col2:

                sentiment_labels = [
                    "Positive",
                    "Neutral",
                    "Negative"
                ]


                sentiment_values = [
                    positive_count,
                    neutral_count,
                    negative_count
                ]


                fig2, ax2 = plt.subplots()


                ax2.bar(
                    sentiment_labels,
                    sentiment_values
                )


                ax2.set_title(
                    "Sentiment Count"
                )


                ax2.set_xlabel(
                    "Sentiment"
                )


                ax2.set_ylabel(
                    "Number of Comments"
                )


                st.pyplot(
                    fig2
                )


                plt.close(
                    fig2
                )


            # =================================================
            # EMOJI STATISTICS
            # =================================================

            st.subheader(
                "😊 Emoji Statistics"
            )


            total_emojis = int(
                results_df[
                    "Emoji Count"
                ].sum()
            )


            emoji_comments = int(
                (
                    results_df[
                        "Emoji Count"
                    ] > 0
                ).sum()
            )


            emoji_col1, emoji_col2 = (
                st.columns(2)
            )


            with emoji_col1:

                st.metric(
                    "Total Emojis Detected",
                    total_emojis
                )


            with emoji_col2:

                st.metric(
                    "Comments Containing Emojis",
                    emoji_comments
                )


            # =================================================
            # DETAILED RESULTS TABLE
            # =================================================

            st.subheader(
                "📋 Individual Comment Results"
            )


            st.dataframe(
                results_df,
                width="stretch",
                hide_index=True
            )


            # =================================================
            # DOWNLOAD RESULTS
            # =================================================

            st.subheader(
                "⬇️ Download Results"
            )


            csv_data = results_df.to_csv(
                index=False
            )


            st.download_button(
                label="📥 Download CSV Results",
                data=csv_data,
                file_name=(
                    "sentimoji_fusion_results.csv"
                ),
                mime="text/csv"
            )


            # =================================================
            # FINAL SUMMARY
            # =================================================

            st.divider()


            st.subheader(
                "🎯 Analysis Summary"
            )


            if (
                positive_percentage
                > negative_percentage
            ):

                st.success(
                    f"""
Overall sentiment is **mostly positive**.

Positive: {positive_percentage:.2f}%

Neutral: {neutral_percentage:.2f}%

Negative: {negative_percentage:.2f}%
"""
                )


            elif (
                negative_percentage
                > positive_percentage
            ):

                st.error(
                    f"""
Overall sentiment is **mostly negative**.

Positive: {positive_percentage:.2f}%

Neutral: {neutral_percentage:.2f}%

Negative: {negative_percentage:.2f}%
"""
                )


            else:

                st.warning(
                    f"""
Overall sentiment is **balanced between
positive and negative**.

Positive: {positive_percentage:.2f}%

Neutral: {neutral_percentage:.2f}%

Negative: {negative_percentage:.2f}%
"""
                )


# ============================================================
# TAB 3 - CSV / EXCEL UPLOAD
# ============================================================

with tab3:

    st.subheader(
        "📁 Upload Comments Dataset"
    )

    st.markdown(
        """
Upload a **CSV or Excel file** containing social media comments.

The system will automatically:

1. Read the comments
2. Analyze text using DistilBERT
3. Detect emojis
4. Calculate emoji sentiment
5. Apply 70% Text + 30% Emoji fusion
6. Generate final sentiment
7. Calculate overall sentiment percentages
8. Generate charts
9. Allow you to download the analyzed results
"""
    )

    st.divider()

    # ========================================================
    # FILE UPLOADER
    # ========================================================

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        key="dataset_uploader"
    )

    # ========================================================
    # PROCESS UPLOADED FILE
    # ========================================================

    if uploaded_file is not None:

        try:

            # ------------------------------------------------
            # READ CSV
            # ------------------------------------------------

            if uploaded_file.name.lower().endswith(".csv"):

                uploaded_df = pd.read_csv(
                    uploaded_file
                )

            # ------------------------------------------------
            # READ EXCEL
            # ------------------------------------------------

            elif uploaded_file.name.lower().endswith(
                ".xlsx"
            ):

                uploaded_df = pd.read_excel(
                    uploaded_file
                )

            else:

                st.error(
                    "Unsupported file format."
                )

                st.stop()


            # =================================================
            # FILE INFORMATION
            # =================================================

            st.success(
                f"File uploaded successfully: "
                f"**{uploaded_file.name}**"
            )


            st.write(
                f"Rows detected: **{len(uploaded_df)}**"
            )


            st.write(
                f"Columns detected: "
                f"**{len(uploaded_df.columns)}**"
            )


            # =================================================
            # SHOW ORIGINAL DATA
            # =================================================

            st.subheader(
                "📋 Uploaded Data Preview"
            )


            st.dataframe(
                uploaded_df.head(10),
                width="stretch",
                hide_index=True
            )


            # =================================================
            # SELECT COMMENT COLUMN
            # =================================================

            st.subheader(
                "📝 Select Comment Column"
            )


            column_options = list(
                uploaded_df.columns
            )


            # Try to automatically find a likely
            # comment/text column

            preferred_columns = [
                "comment",
                "comments",
                "text",
                "review",
                "reviews",
                "content",
                "tweet",
                "message"
            ]


            detected_column = None


            for column in column_options:

                if str(column).lower().strip() in (
                    preferred_columns
                ):

                    detected_column = column

                    break


            if detected_column is not None:

                default_index = column_options.index(
                    detected_column
                )

            else:

                default_index = 0


            selected_column = st.selectbox(
                "Select the column containing comments:",
                column_options,
                index=default_index,
                key="comment_column"
            )


            # =================================================
            # NUMBER OF ROWS
            # =================================================

            max_rows = len(
                uploaded_df
            )


            st.subheader(
                "⚙️ Analysis Settings"
            )


            rows_to_analyze = st.number_input(
                "Number of rows to analyze:",
                min_value=1,
                max_value=max_rows,
                value=min(50, max_rows),
                step=1,
                key="rows_to_analyze"
            )


            st.caption(
                "For the first test, use 10–50 rows. "
                "Large files can take longer because "
                "each comment is processed by DistilBERT."
            )


            # =================================================
            # START ANALYSIS
            # =================================================

            if st.button(
                "🚀 Analyze Uploaded Dataset",
                type="primary",
                key="upload_analyze"
            ):

                # --------------------------------------------
                # SELECT ROWS
                # --------------------------------------------

                analysis_df = uploaded_df.head(
                    int(rows_to_analyze)
                ).copy()


                # --------------------------------------------
                # REMOVE EMPTY COMMENTS
                # --------------------------------------------

                analysis_df = analysis_df[
                    analysis_df[
                        selected_column
                    ].notna()
                ]


                analysis_df[
                    selected_column
                ] = analysis_df[
                    selected_column
                ].astype(str)


                analysis_df[
                    selected_column
                ] = analysis_df[
                    selected_column
                ].str.strip()


                analysis_df = analysis_df[
                    analysis_df[
                        selected_column
                    ] != ""
                ]


                if len(analysis_df) == 0:

                    st.warning(
                        "No valid comments were found "
                        "in the selected column."
                    )

                    st.stop()


                # --------------------------------------------
                # ANALYSIS START
                # --------------------------------------------

                st.info(
                    f"Analyzing **{len(analysis_df)} "
                    f"comments**..."
                )


                results = []


                progress_bar = st.progress(
                    0
                )


                # --------------------------------------------
                # PROCESS EACH COMMENT
                # --------------------------------------------

                for index, comment in enumerate(
                    analysis_df[
                        selected_column
                    ].tolist()
                ):

                    result = predict_sentiment(
                        comment,
                        tokenizer,
                        model
                    )


                    text_scores = result[
                        "text_scores"
                    ]


                    emoji_scores = result[
                        "emoji_scores"
                    ]


                    final_scores = result[
                        "scores"
                    ]


                    detected_emojis = " ".join(
                        result["emojis"]
                    )


                    results.append({

                        "Comment":
                            comment,

                        "Sentiment":
                            result[
                                "sentiment"
                            ].capitalize(),

                        "Confidence (%)":
                            round(
                                result[
                                    "confidence"
                                ],
                                2
                            ),

                        "Text Negative (%)":
                            round(
                                text_scores[
                                    "negative"
                                ],
                                2
                            ),

                        "Text Neutral (%)":
                            round(
                                text_scores[
                                    "neutral"
                                ],
                                2
                            ),

                        "Text Positive (%)":
                            round(
                                text_scores[
                                    "positive"
                                ],
                                2
                            ),

                        "Emoji Negative (%)":
                            round(
                                emoji_scores[
                                    "negative"
                                ],
                                2
                            ),

                        "Emoji Neutral (%)":
                            round(
                                emoji_scores[
                                    "neutral"
                                ],
                                2
                            ),

                        "Emoji Positive (%)":
                            round(
                                emoji_scores[
                                    "positive"
                                ],
                                2
                            ),

                        "Final Negative (%)":
                            round(
                                final_scores[
                                    "negative"
                                ],
                                2
                            ),

                        "Final Neutral (%)":
                            round(
                                final_scores[
                                    "neutral"
                                ],
                                2
                            ),

                        "Final Positive (%)":
                            round(
                                final_scores[
                                    "positive"
                                ],
                                2
                            ),

                        "Detected Emojis":
                            detected_emojis,

                        "Emoji Count":
                            result[
                                "emoji_count"
                            ]
                    })


                    # Update progress
                    progress_bar.progress(
                        (index + 1)
                        / len(analysis_df)
                    )


                progress_bar.empty()


                # =================================================
                # CREATE RESULTS DATAFRAME
                # =================================================

                upload_results_df = pd.DataFrame(
                    results
                )


                st.success(
                    "Dataset analysis completed successfully! 🎉"
                )


                # =================================================
                # OVERALL SENTIMENT COUNTS
                # =================================================

                positive_count = int(
                    (
                        upload_results_df[
                            "Sentiment"
                        ]
                        == "Positive"
                    ).sum()
                )


                neutral_count = int(
                    (
                        upload_results_df[
                            "Sentiment"
                        ]
                        == "Neutral"
                    ).sum()
                )


                negative_count = int(
                    (
                        upload_results_df[
                            "Sentiment"
                        ]
                        == "Negative"
                    ).sum()
                )


                total_count = len(
                    upload_results_df
                )


                # =================================================
                # PERCENTAGES
                # =================================================

                positive_percentage = (
                    positive_count
                    / total_count
                ) * 100


                neutral_percentage = (
                    neutral_count
                    / total_count
                ) * 100


                negative_percentage = (
                    negative_count
                    / total_count
                ) * 100


                # =================================================
                # OVERALL SENTIMENT
                # =================================================

                st.divider()


                st.subheader(
                    "📈 Overall Dataset Sentiment"
                )


                col1, col2, col3, col4 = (
                    st.columns(4)
                )


                with col1:

                    st.metric(
                        "🟢 Positive",
                        f"{positive_percentage:.2f}%"
                    )


                with col2:

                    st.metric(
                        "🟡 Neutral",
                        f"{neutral_percentage:.2f}%"
                    )


                with col3:

                    st.metric(
                        "🔴 Negative",
                        f"{negative_percentage:.2f}%"
                    )


                with col4:

                    st.metric(
                        "💬 Comments",
                        total_count
                    )


                # =================================================
                # COUNTS
                # =================================================

                st.write(
                    f"""
🟢 Positive: **{positive_count}**

🟡 Neutral: **{neutral_count}**

🔴 Negative: **{negative_count}**
"""
                )


                # =================================================
                # CHARTS
                # =================================================

                st.subheader(
                    "📊 Sentiment Distribution"
                )


                chart_col1, chart_col2 = (
                    st.columns(2)
                )


                # -------------------------------------------------
                # PIE CHART
                # -------------------------------------------------

                with chart_col1:

                    labels = [
                        "Positive",
                        "Neutral",
                        "Negative"
                    ]


                    values = [
                        positive_count,
                        neutral_count,
                        negative_count
                    ]


                    fig1, ax1 = plt.subplots()


                    ax1.pie(
                        values,
                        labels=labels,
                        autopct="%1.1f%%",
                        startangle=90
                    )


                    ax1.set_title(
                        "Dataset Sentiment Distribution"
                    )


                    ax1.axis(
                        "equal"
                    )


                    st.pyplot(
                        fig1
                    )


                    plt.close(
                        fig1
                    )


                # -------------------------------------------------
                # BAR CHART
                # -------------------------------------------------

                with chart_col2:

                    sentiment_labels = [
                        "Positive",
                        "Neutral",
                        "Negative"
                    ]


                    sentiment_values = [
                        positive_count,
                        neutral_count,
                        negative_count
                    ]


                    fig2, ax2 = plt.subplots()


                    ax2.bar(
                        sentiment_labels,
                        sentiment_values
                    )


                    ax2.set_title(
                        "Sentiment Count"
                    )


                    ax2.set_xlabel(
                        "Sentiment"
                    )


                    ax2.set_ylabel(
                        "Number of Comments"
                    )


                    st.pyplot(
                        fig2
                    )


                    plt.close(
                        fig2
                    )


                # =================================================
                # EMOJI STATISTICS
                # =================================================

                st.subheader(
                    "😊 Emoji Statistics"
                )


                total_emojis = int(
                    upload_results_df[
                        "Emoji Count"
                    ].sum()
                )


                emoji_comments = int(
                    (
                        upload_results_df[
                            "Emoji Count"
                        ] > 0
                    ).sum()
                )


                emoji_col1, emoji_col2 = (
                    st.columns(2)
                )


                with emoji_col1:

                    st.metric(
                        "Total Emojis Detected",
                        total_emojis
                    )


                with emoji_col2:

                    st.metric(
                        "Comments Containing Emojis",
                        emoji_comments
                    )


                # =================================================
                # RESULTS TABLE
                # =================================================

                st.subheader(
                    "📋 Detailed Analysis Results"
                )


                st.dataframe(
                    upload_results_df,
                    width="stretch",
                    hide_index=True
                )


                # =================================================
                # DOWNLOAD CSV
                # =================================================

                st.subheader(
                    "⬇️ Download Results"
                )


                csv_data = (
                    upload_results_df
                    .to_csv(index=False)
                )


                st.download_button(
                    label="📥 Download CSV Results",
                    data=csv_data,
                    file_name=(
                        "SentiMoji_Dataset_Analysis.csv"
                    ),
                    mime="text/csv",
                    key="download_upload_csv"
                )


                # =================================================
                # DOWNLOAD EXCEL
                # =================================================

                excel_buffer = io.BytesIO()


                with pd.ExcelWriter(
                    excel_buffer,
                    engine="openpyxl"
                ) as writer:

                    upload_results_df.to_excel(
                        writer,
                        index=False,
                        sheet_name="Sentiment Results"
                    )


                excel_buffer.seek(0)


                st.download_button(
                    label="📊 Download Excel Results",
                    data=excel_buffer,
                    file_name=(
                        "SentiMoji_Dataset_Analysis.xlsx"
                    ),
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    key="download_upload_excel"
                )


                # =================================================
                # SUMMARY
                # =================================================

                st.divider()


                st.subheader(
                    "🎯 Dataset Analysis Summary"
                )


                if (
                    positive_percentage
                    > negative_percentage
                ):

                    st.success(
                        f"""
Overall sentiment of the uploaded
dataset is **mostly positive**.

Positive: {positive_percentage:.2f}%

Neutral: {neutral_percentage:.2f}%

Negative: {negative_percentage:.2f}%
"""
                    )


                elif (
                    negative_percentage
                    > positive_percentage
                ):

                    st.error(
                        f"""
Overall sentiment of the uploaded
dataset is **mostly negative**.

Positive: {positive_percentage:.2f}%

Neutral: {neutral_percentage:.2f}%

Negative: {negative_percentage:.2f}%
"""
                    )


                else:

                    st.warning(
                        f"""
Overall sentiment is relatively
balanced.

Positive: {positive_percentage:.2f}%

Neutral: {neutral_percentage:.2f}%

Negative: {negative_percentage:.2f}%
"""
                    )


        except Exception as e:

            st.error(
                f"Error while processing the uploaded file: {e}"
            )

            st.exception(e)

            # ============================================================
# TAB 4 - MODEL EVALUATION
# ============================================================

with tab4:

    st.subheader(
        "📈 Model Evaluation"
    )

    st.markdown(
        """
This section compares the performance of the three
sentiment classification models developed in the project.

The models are:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- DistilBERT Transformer
"""
    )

    st.divider()

    # ========================================================
    # MODEL PERFORMANCE DATA
    # ========================================================

    model_names = [
        "TF-IDF +\nLogistic Regression",
        "TF-IDF + SVM",
        "DistilBERT"
    ]

    validation_accuracy = [
        67.15,
        64.00,
        72.62
    ]

    test_accuracy = [
        58.82,
        55.59,
        67.67
    ]

    # ========================================================
    # BEST MODEL
    # ========================================================

    st.subheader(
        "🏆 Best Performing Model"
    )

    best_model_col1, best_model_col2 = st.columns(2)

    with best_model_col1:

        st.metric(
            "Best Model",
            "DistilBERT"
        )

    with best_model_col2:

        st.metric(
            "Test Accuracy",
            "67.67%"
        )

    st.success(
        """
DistilBERT achieved the highest test accuracy among
the three evaluated models.
"""
    )

    # ========================================================
    # MODEL ACCURACY TABLE
    # ========================================================

    st.subheader(
        "📊 Model Accuracy Comparison"
    )

    comparison_df = pd.DataFrame({

        "Model": [
            "TF-IDF + Logistic Regression",
            "TF-IDF + SVM",
            "DistilBERT"
        ],

        "Validation Accuracy (%)": [
            67.15,
            64.00,
            72.62
        ],

        "Test Accuracy (%)": [
            58.82,
            55.59,
            67.67
        ]
    })

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True
    )

    # ========================================================
    # ACCURACY CHART
    # ========================================================

    st.subheader(
        "📈 Model Accuracy Comparison"
    )

    fig_accuracy, ax_accuracy = plt.subplots()

    x = np.arange(
        len(model_names)
    )

    width = 0.35

    ax_accuracy.bar(
        x - width / 2,
        validation_accuracy,
        width,
        label="Validation Accuracy"
    )

    ax_accuracy.bar(
        x + width / 2,
        test_accuracy,
        width,
        label="Test Accuracy"
    )

    ax_accuracy.set_ylabel(
        "Accuracy (%)"
    )

    ax_accuracy.set_xlabel(
        "Model"
    )

    ax_accuracy.set_title(
        "Model Performance Comparison"
    )

    ax_accuracy.set_xticks(
        x
    )

    ax_accuracy.set_xticklabels(
        model_names
    )

    ax_accuracy.set_ylim(
        0,
        100
    )

    ax_accuracy.legend()

    st.pyplot(
        fig_accuracy
    )

    plt.close(
        fig_accuracy
    )

    # ========================================================
    # DISTILBERT TEST METRICS
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 DistilBERT Test Performance"
    )

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    with metric1:

        st.metric(
            "Accuracy",
            "67.67%"
        )

    with metric2:

        st.metric(
            "Precision",
            "67.79%"
        )

    with metric3:

        st.metric(
            "Recall",
            "67.67%"
        )

    with metric4:

        st.metric(
            "F1 Score",
            "67.69%"
        )

    # ========================================================
    # CLASS-WISE PERFORMANCE
    # ========================================================

    st.subheader(
        "📋 Class-wise Performance"
    )

    class_report_df = pd.DataFrame({

        "Sentiment": [
            "Negative",
            "Neutral",
            "Positive"
        ],

        "Precision (%)": [
            69.39,
            68.99,
            62.13
        ],

        "Recall (%)": [
            65.46,
            69.13,
            67.71
        ],

        "F1 Score (%)": [
            67.37,
            69.06,
            64.80
        ]
    })

    st.dataframe(
        class_report_df,
        width="stretch",
        hide_index=True
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔲 DistilBERT Confusion Matrix"
    )

    confusion_matrix = np.array([
        [2600, 1140, 232],
        [1085, 4104, 748],
        [62, 705, 1608]
    ])

    fig_cm, ax_cm = plt.subplots()

    image = ax_cm.imshow(
        confusion_matrix
    )

    ax_cm.set_title(
        "DistilBERT Test Confusion Matrix"
    )

    ax_cm.set_xlabel(
        "Predicted Label"
    )

    ax_cm.set_ylabel(
        "Actual Label"
    )

    ax_cm.set_xticks(
        [0, 1, 2]
    )

    ax_cm.set_yticks(
        [0, 1, 2]
    )

    ax_cm.set_xticklabels(
        [
            "Negative",
            "Neutral",
            "Positive"
        ]
    )

    ax_cm.set_yticklabels(
        [
            "Negative",
            "Neutral",
            "Positive"
        ]
    )

    # Add values inside cells

    for i in range(3):

        for j in range(3):

            ax_cm.text(
                j,
                i,
                confusion_matrix[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(
        fig_cm
    )

    plt.close(
        fig_cm
    )

    # ========================================================
    # CONFUSION MATRIX EXPLANATION
    # ========================================================

    st.info(
        """
### How to read the confusion matrix

Rows represent the **actual sentiment**.

Columns represent the **predicted sentiment**.

The diagonal values represent correctly classified
samples.

For example:

- 2,600 negative samples were correctly classified
  as negative.
- 4,104 neutral samples were correctly classified
  as neutral.
- 1,608 positive samples were correctly classified
  as positive.
"""
    )

    # ========================================================
    # MODEL SELECTION CONCLUSION
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Model Selection"
    )

    st.success(
        """
Based on the test results, **DistilBERT** was selected
as the final sentiment classification model because it
achieved the highest test accuracy of **67.67%** among
the evaluated models.
"""
    )

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📌 Project Information"
)


st.sidebar.markdown(
    """
### SentiMoji

**Model:** DistilBERT

**Task:** 3-Class Sentiment Classification

**Classes:**

🔴 Negative

🟡 Neutral

🟢 Positive

**Training Dataset:** 8,000 records

**Final Test Dataset:** 12,284 records

**Test Accuracy:** 67.67%
"""
)


st.sidebar.divider()


st.sidebar.subheader(
    "🔬 Fusion Architecture"
)


st.sidebar.write(
    """
**Text Branch**

DistilBERT

↓

Text Sentiment Scores


**Emoji Branch**

Emoji Detection

↓

Emoji Sentiment Scores


**Fusion**

70% Text

+

30% Emoji


↓

Final Sentiment
"""
)


st.sidebar.divider()


st.sidebar.subheader(
    "📊 Model Information"
)


st.sidebar.write(
    """
The application uses a fine-tuned
DistilBERT model for three-class
sentiment classification.

Emoji information is additionally
processed using an explicit emoji
sentiment scoring component.
"""
)


st.sidebar.divider()


st.sidebar.write(
    "😊 Emoji-Based Sentiment Analysis Project"
)