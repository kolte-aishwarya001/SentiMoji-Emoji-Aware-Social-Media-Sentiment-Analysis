
import os
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from amazon_reviews import extract_asin


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SentiMoji | Amazon Review Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(APP_DIR / ".env", override=True)

# Support either environment-variable name.
API_TOKEN = (
    os.getenv("API_TOKEN")
    or os.getenv("JUSTONE_API_TOKEN")
    or ""
).strip()

API_URL = (
    "https://api.justoneapi.com/"
    "api/amazon/get-product-top-reviews/v1"
)


# ============================================================
# 2. PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 14px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "amazon_reviews" not in st.session_state:
    st.session_state["amazon_reviews"] = []

if "amazon_asin" not in st.session_state:
    st.session_state["amazon_asin"] = ""

if "sentimoji_results" not in st.session_state:
    st.session_state["sentimoji_results"] = []


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def fetch_amazon_reviews(product_url):
    """Fetch Amazon reviews using the Just One API."""

    if not API_TOKEN or API_TOKEN == "paste_your_real_token_here":
        raise ValueError(
            "API token not found. Check API_TOKEN or "
            "JUSTONE_API_TOKEN in your .env file."
        )

    asin = extract_asin(product_url)

    response = requests.get(
        API_URL,
        params={
            "token": API_TOKEN,
            "asin": asin,
            "country": "IN",
        },
        timeout=120,
    )

    response.raise_for_status()
    body = response.json()

    if not isinstance(body, dict):
        raise RuntimeError("The API returned an unexpected response.")

    api_status = body.get("status", body.get("code"))

    if api_status is not None and str(api_status) != "0":
        message = (
            body.get("message")
            or body.get("msg")
            or f"API returned status {api_status}"
        )
        raise RuntimeError(message)

    # Handle either a top-level reviews list or a nested response.
    payload = body

    for key in ("data", "result"):
        nested = body.get(key)
        if isinstance(nested, dict) and "reviews" in nested:
            payload = nested
            break

    reviews = payload.get("reviews", [])

    if not isinstance(reviews, list):
        raise RuntimeError("The API response contains an invalid review list.")

    return asin, reviews


def get_review_text(review):
    """Combine the review title and comment."""

    title = str(review.get("review_title") or "").strip()
    comment = str(review.get("review_comment") or "").strip()

    if title and comment:
        return f"{title}. {comment}"

    return title or comment


def analyze_reviews(reviews):
    """Run the existing SentiMoji model on retrieved reviews."""

    from src.prediction.predict import predict_sentiment

    results = []
    total = len(reviews)
    progress = st.progress(0)
    status = st.empty()

    try:
        for index, review in enumerate(reviews, start=1):
            text = get_review_text(review)

            if not text:
                continue

            status.write(f"Analyzing review {index} of {total}...")

            prediction = predict_sentiment(text)

            sentiment = str(
                prediction.get("sentiment", "UNKNOWN")
            ).upper()

            try:
                confidence = float(prediction.get("confidence", 0))
            except (TypeError, ValueError):
                confidence = 0.0

            detected_emojis = prediction.get("emojis", [])

            if isinstance(detected_emojis, (list, tuple)):
                emoji_text = " ".join(map(str, detected_emojis))
            else:
                emoji_text = str(detected_emojis or "")

            try:
                rating = float(review.get("review_star_rating"))
            except (TypeError, ValueError):
                rating = None

            results.append({
                "Review": text,
                "Amazon Rating": rating,
                "Predicted Sentiment": sentiment,
                "Confidence (%)": round(confidence, 2),
                "Emojis": emoji_text,
                "Emoji Count": prediction.get("emoji_count", len(emoji_text)),
                "Verified Purchase": review.get(
                    "is_verified_purchase", False
                ),
                "Review Date": review.get("review_date", "N/A"),
                "Review Link": review.get("review_link", ""),
            })

            progress.progress(index / total)

        status.success(f"Analysis completed: {len(results)} reviews analyzed.")
        return results

    finally:
        # Remove the progress indicator after the run.
        progress.empty()


# ============================================================
# 5. HEADER
# ============================================================

with st.sidebar:
    st.title("😊 SentiMoji")
    st.caption("Emoji-Aware Sentiment Analysis")
    st.divider()
    st.write("**Features**")
    st.write("• Amazon review retrieval")
    st.write("• Sentiment classification")
    st.write("• Model confidence")
    st.write("• Emoji insights")
    st.write("• CSV report export")

st.markdown(
    '<p class="main-title">😊 SentiMoji</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">'
    'Amazon Product Review Sentiment Analysis'
    '</p>',
    unsafe_allow_html=True,
)

st.write(
    "Retrieve customer reviews from Amazon and analyze them using "
    "your existing emoji-aware SentiMoji model."
)


# ============================================================
# 6. AMAZON PRODUCT INPUT
# ============================================================

st.subheader("🛍️ Analyze an Amazon Product")

product_url = st.text_input(
    "Amazon product URL",
    placeholder="https://www.amazon.in/dp/B0C248QHSS",
    key="product_url_input",
)

fetch_col, clear_col = st.columns([3, 1])

with fetch_col:
    fetch_clicked = st.button(
        "Fetch Amazon Reviews",
        type="primary",
        use_container_width=True,
    )

with clear_col:
    clear_clicked = st.button(
        "Clear Results",
        use_container_width=True,
    )

if clear_clicked:
    st.session_state["amazon_reviews"] = []
    st.session_state["amazon_asin"] = ""
    st.session_state["sentimoji_results"] = []
    st.rerun()

if fetch_clicked:
    if not product_url.strip():
        st.warning("Please enter an Amazon product URL.")
    else:
        try:
            with st.spinner("Fetching Amazon reviews..."):
                asin, reviews = fetch_amazon_reviews(product_url.strip())

            if not reviews:
                st.warning(
                    "The API returned no reviews for this product."
                )
                st.session_state["amazon_reviews"] = []
                st.session_state["sentimoji_results"] = []
            else:
                st.session_state["amazon_asin"] = asin
                st.session_state["amazon_reviews"] = reviews
                st.session_state["sentimoji_results"] = []

                st.success(
                    f"Retrieved {len(reviews)} reviews successfully."
                )

        except ValueError as error:
            st.error(str(error))
        except requests.RequestException:
            st.error(
                "Could not contact the Amazon review API. "
                "Check your internet connection and try again."
            )
        except Exception as error:
            st.error(f"Review retrieval failed: {error}")


# ============================================================
# 7. FETCHED REVIEW INFORMATION
# ============================================================

reviews = st.session_state["amazon_reviews"]

if reviews:
    st.divider()
    st.subheader("📦 Product Information")

    info1, info2 = st.columns(2)
    info1.metric("Product ASIN", st.session_state["amazon_asin"])
    info2.metric("Reviews Retrieved", len(reviews))

    with st.expander("Preview retrieved Amazon reviews"):
        preview_rows = []

        for review in reviews:
            preview_rows.append({
                "Title": review.get("review_title", ""),
                "Review": review.get("review_comment", ""),
                "Rating": review.get("review_star_rating", "N/A"),
                "Date": review.get("review_date", "N/A"),
            })

        st.dataframe(
            pd.DataFrame(preview_rows),
            use_container_width=True,
            hide_index=True,
        )

    if st.button(
        "Run SentiMoji Analysis",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Loading the sentiment model and analyzing reviews..."
            ):
                results = analyze_reviews(reviews)

            st.session_state["sentimoji_results"] = results

            if not results:
                st.warning("No review text was available to analyze.")
            else:
                st.success("SentiMoji analysis completed.")

        except Exception as error:
            st.error(f"Sentiment analysis failed: {error}")
            st.caption(
                "Check the model path, dependencies, and available memory."
            )


# ============================================================
# 8. RESULTS DASHBOARD
# ============================================================

results = st.session_state["sentimoji_results"]

if results:
    df = pd.DataFrame(results)

    st.divider()
    st.header("📊 Sentiment Analysis Dashboard")

    counts = (
        df["Predicted Sentiment"]
        .value_counts()
        .to_dict()
    )

    positive = counts.get("POSITIVE", 0)
    negative = counts.get("NEGATIVE", 0)
    neutral = counts.get("NEUTRAL", 0)
    total_reviews = len(df)

    positive_pct = positive / total_reviews * 100
    negative_pct = negative / total_reviews * 100
    neutral_pct = neutral / total_reviews * 100

    # --------------------------------------------------------
    # 8.1 SENTIMENT METRICS
    # --------------------------------------------------------

    st.subheader("Sentiment Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Reviews Analyzed", total_reviews)
    col2.metric("Positive", f"{positive} ({positive_pct:.1f}%)")
    col3.metric("Negative", f"{negative} ({negative_pct:.1f}%)")
    col4.metric("Neutral", f"{neutral} ({neutral_pct:.1f}%)")

    # --------------------------------------------------------
    # 8.2 SENTIMENT CHART
    # --------------------------------------------------------

    st.subheader("Sentiment Distribution")

    sentiment_chart = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Reviews": [positive, negative, neutral],
    }).set_index("Sentiment")

    st.bar_chart(sentiment_chart)

    # --------------------------------------------------------
    # 8.3 CONFIDENCE METRICS
    # --------------------------------------------------------

    st.subheader("🎯 Model Confidence")

    confidence_values = pd.to_numeric(
        df["Confidence (%)"], errors="coerce"
    ).dropna()

    if not confidence_values.empty:
        avg_confidence = confidence_values.mean()
        confidence_col1, confidence_col2 = st.columns(2)

        confidence_col1.metric(
            "Average Reported Confidence",
            f"{avg_confidence:.2f}%",
        )
        confidence_col2.metric(
            "Reviews with Confidence Scores",
            len(confidence_values),
        )

        st.caption(
            "Model confidence is not the same as verified prediction "
            "accuracy."
        )

    # --------------------------------------------------------
    # 8.4 AMAZON STAR RATINGS
    # --------------------------------------------------------

    st.subheader("⭐ Amazon Rating Insights")

    ratings = pd.to_numeric(
        df["Amazon Rating"], errors="coerce"
    ).dropna()

    rating_col1, rating_col2, rating_col3 = st.columns(3)

    if not ratings.empty:
        rating_col1.metric(
            "Average Amazon Rating",
            f"{ratings.mean():.2f} / 5",
        )
        rating_col2.metric(
            "Reviews with Ratings",
            f"{len(ratings)} / {total_reviews}",
        )
        rating_col3.metric(
            "Five-Star Reviews",
            int((ratings == 5).sum()),
        )

        rating_distribution = (
            ratings.value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .sort_index()
        )

        st.write("Star Rating Distribution")
        st.bar_chart(rating_distribution)
    else:
        st.info("No valid Amazon star ratings were returned.")

    # --------------------------------------------------------
    # 8.5 EMOJI INSIGHTS
    # --------------------------------------------------------

    st.subheader("😊 Emoji Insights")

    # Count emoji tokens supplied by the existing prediction function.
    emoji_counter = Counter()
    reviews_with_emojis = 0
    emoji_occurrences = 0

    for emoji_text in df["Emojis"].fillna("").astype(str):
        tokens = emoji_text.split()

        if tokens:
            reviews_with_emojis += 1
            emoji_occurrences += len(tokens)
            emoji_counter.update(tokens)

    emoji_col1, emoji_col2, emoji_col3 = st.columns(3)

    emoji_col1.metric(
        "Reviews with Emojis",
        reviews_with_emojis,
    )
    emoji_col2.metric(
        "Emoji Occurrences",
        emoji_occurrences,
    )
    emoji_col3.metric(
        "Unique Emoji Tokens",
        len(emoji_counter),
    )

    if emoji_counter:
        emoji_chart = pd.Series(
            dict(emoji_counter.most_common(10)),
            name="Occurrences",
        )
        st.write("Most Frequently Detected Emojis")
        st.bar_chart(emoji_chart)
    else:
        st.info(
            "No emojis were detected by the prediction pipeline "
            "in these reviews."
        )

    # --------------------------------------------------------
    # 8.6 INDIVIDUAL PREDICTIONS
    # --------------------------------------------------------

    st.subheader("📝 Individual Review Predictions")

    display_columns = [
        "Review",
        "Amazon Rating",
        "Predicted Sentiment",
        "Confidence (%)",
        "Emojis",
        "Verified Purchase",
        "Review Date",
    ]

    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # 8.7 SENTIMENT FILTER
    # --------------------------------------------------------

    st.subheader("🔎 Filter Reviews")

    available_sentiments = sorted(
        df["Predicted Sentiment"].dropna().unique().tolist()
    )

    selected_sentiments = st.multiselect(
        "Select sentiment labels",
        options=available_sentiments,
        default=available_sentiments,
    )

    filtered_df = df[
        df["Predicted Sentiment"].isin(selected_sentiments)
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # 8.8 CSV DOWNLOAD
    # --------------------------------------------------------

    st.subheader("📥 Export Results")

    csv_data = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Download Complete Analysis as CSV",
        data=csv_data,
        file_name="sentimoji_amazon_review_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Amazon ratings and SentiMoji predictions are separate signals. "
        "A review's star rating may not always match its predicted sentiment."
    )


# ============================================================
# 9. MANUAL REVIEW ANALYSIS
# ============================================================

st.divider()
st.header("🧪 Test a Single Review Manually")

manual_review = st.text_area(
    "Enter a customer review",
    placeholder="Example: Excellent quality! I love this product 😊❤️",
    height=120,
)

if st.button("Analyze Manual Review"):
    if not manual_review.strip():
        st.warning("Please enter review text first.")
    else:
        try:
            from src.prediction.predict import predict_sentiment

            with st.spinner("Analyzing your review..."):
                prediction = predict_sentiment(manual_review.strip())

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "Predicted Sentiment",
                str(prediction.get("sentiment", "Unknown")).upper(),
            )

            result_col2.metric(
                "Confidence",
                f"{float(prediction.get('confidence', 0)):.2f}%",
            )

            st.subheader("Sentiment Scores")
            st.bar_chart(prediction.get("scores", {}))

            st.write(
                "**Detected emojis:**",
                prediction.get("emojis", []),
            )
            st.write(
                "**Emoji count:**",
                prediction.get("emoji_count", 0),
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.caption(
                "Check the model path, dependencies, and available memory."
            )


# ============================================================
# 10. FOOTER
# ============================================================

st.divider()

st.caption(
    "SentiMoji | Emoji-Aware Social Media Sentiment Analysis"
)

st.caption(
    "This prototype analyzes the reviews returned by the configured "
    "Amazon review API. Results depend on the retrieved sample and "
    "the trained model; they do not represent every customer review."
)
