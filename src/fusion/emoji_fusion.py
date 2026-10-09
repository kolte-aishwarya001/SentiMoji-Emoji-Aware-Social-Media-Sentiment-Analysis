# ============================================================
# EMOJI SENTIMENT FUSION
# ============================================================

import numpy as np


# ============================================================
# EMOJI SENTIMENT LEXICON
# ============================================================

EMOJI_SENTIMENT = {

    # --------------------------------------------------------
    # Positive emojis
    # --------------------------------------------------------

    "😀": "positive",
    "😃": "positive",
    "😄": "positive",
    "😁": "positive",
    "😆": "positive",
    "😅": "positive",
    "😂": "positive",
    "🤣": "positive",
    "😊": "positive",
    "😇": "positive",
    "🙂": "positive",
    "🙃": "positive",
    "😉": "positive",
    "😌": "positive",
    "😍": "positive",
    "🥰": "positive",
    "😘": "positive",
    "😗": "positive",
    "😙": "positive",
    "😚": "positive",
    "🤩": "positive",
    "🥳": "positive",
    "😎": "positive",
    "🤗": "positive",
    "❤️": "positive",
    "❤": "positive",
    "💕": "positive",
    "💖": "positive",
    "💗": "positive",
    "💓": "positive",
    "💞": "positive",
    "💯": "positive",
    "👍": "positive",
    "👏": "positive",
    "🙌": "positive",
    "✨": "positive",
    "🔥": "positive",
    "🎉": "positive",

    # --------------------------------------------------------
    # Negative emojis
    # --------------------------------------------------------

    "😡": "negative",
    "😠": "negative",
    "🤬": "negative",
    "😤": "negative",
    "😞": "negative",
    "😔": "negative",
    "😟": "negative",
    "😕": "negative",
    "🙁": "negative",
    "☹️": "negative",
    "😣": "negative",
    "😖": "negative",
    "😫": "negative",
    "😩": "negative",
    "🥺": "negative",
    "😢": "negative",
    "😭": "negative",
    "😥": "negative",
    "😓": "negative",
    "😰": "negative",
    "😨": "negative",
    "😱": "negative",
    "🤢": "negative",
    "🤮": "negative",
    "🤧": "negative",
    "💔": "negative",
    "👎": "negative",
    "😒": "negative",
    "🙄": "negative",
    "😐": "neutral",
    "😑": "neutral",
    "😶": "neutral",
    "🤔": "neutral",
    "😬": "neutral",
    "🫤": "neutral",
}


# ============================================================
# EMOJI PROBABILITY CALCULATION
# ============================================================

def calculate_emoji_probabilities(emojis):

    """
    Convert detected emojis into a probability distribution:

    [negative, neutral, positive]
    """

    if not emojis:

        return {
            "negative": 0.0,
            "neutral": 1.0,
            "positive": 0.0
        }


    positive_count = 0
    negative_count = 0
    neutral_count = 0


    for emoji in emojis:

        sentiment = EMOJI_SENTIMENT.get(
            emoji,
            "neutral"
        )


        if sentiment == "positive":

            positive_count += 1


        elif sentiment == "negative":

            negative_count += 1


        else:

            neutral_count += 1


    total = (
        positive_count
        + negative_count
        + neutral_count
    )


    if total == 0:

        return {
            "negative": 0.0,
            "neutral": 1.0,
            "positive": 0.0
        }


    # --------------------------------------------------------
    # Convert counts to probabilities
    # --------------------------------------------------------

    negative_probability = (
        negative_count / total
    )

    neutral_probability = (
        neutral_count / total
    )

    positive_probability = (
        positive_count / total
    )


    return {

        "negative":
            negative_probability,

        "neutral":
            neutral_probability,

        "positive":
            positive_probability
    }


# ============================================================
# TEXT + EMOJI FUSION
# ============================================================

def fuse_text_and_emoji(
    text_scores,
    emoji_scores,
    text_weight=0.70,
    emoji_weight=0.30
):

    """
    Combine text-model probabilities and
    emoji probabilities.

    Default:
        Text  = 70%
        Emoji = 30%
    """


    # --------------------------------------------------------
    # Text probabilities
    # --------------------------------------------------------

    text_negative = (
        text_scores["negative"] / 100
    )

    text_neutral = (
        text_scores["neutral"] / 100
    )

    text_positive = (
        text_scores["positive"] / 100
    )


    # --------------------------------------------------------
    # Emoji probabilities
    # --------------------------------------------------------

    emoji_negative = (
        emoji_scores["negative"]
    )

    emoji_neutral = (
        emoji_scores["neutral"]
    )

    emoji_positive = (
        emoji_scores["positive"]
    )


    # --------------------------------------------------------
    # Weighted fusion
    # --------------------------------------------------------

    final_negative = (

        text_weight * text_negative

        +

        emoji_weight * emoji_negative
    )


    final_neutral = (

        text_weight * text_neutral

        +

        emoji_weight * emoji_neutral
    )


    final_positive = (

        text_weight * text_positive

        +

        emoji_weight * emoji_positive
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    total = (

        final_negative

        +

        final_neutral

        +

        final_positive
    )


    if total > 0:

        final_negative /= total

        final_neutral /= total

        final_positive /= total


    return {

        "negative":
            final_negative * 100,

        "neutral":
            final_neutral * 100,

        "positive":
            final_positive * 100
    }