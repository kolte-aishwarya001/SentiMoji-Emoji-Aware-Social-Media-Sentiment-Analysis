import pandas as pd
import random
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "dataset/final_train.csv"
OUTPUT_FILE = "dataset/emoji_enhanced_train.csv"

random.seed(42)

# ============================================================
# LOAD ORIGINAL TRAINING DATA
# ============================================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8")

print("Original training dataset:", len(df))
print(df["sentiment"].value_counts())

# ============================================================
# EMOJI GROUPS
# ============================================================

positive_emojis = [
    "😀", "😃", "😄", "😁", "😆",
    "😊", "😍", "🥰", "😘", "😎",
    "🤩", "🥳", "❤️", "💕", "💖",
    "💗", "👍", "👏", "🎉", "🔥",
    "✨", "🙌", "💯", "🌟"
]

negative_emojis = [
    "😞", "😔", "😟", "😕", "🙁",
    "☹️", "😣", "😖", "😫", "😩",
    "🥺", "😢", "😭", "😡", "😠",
    "🤬", "😤", "💔", "👎", "😱",
    "🤮", "🤢", "😒", "😓"
]

neutral_emojis = [
    "🤔", "😐", "😑", "😶",
    "🙄", "😮", "😯", "😲",
    "👀", "📌", "📅", "📦",
    "📰", "💬", "❓", "❗",
    "🙂", "😌"
]

# ============================================================
# FUNCTION TO ADD EMOJIS
# ============================================================

def add_emojis(text, sentiment):

    if sentiment == "positive":
        emojis = positive_emojis

    elif sentiment == "negative":
        emojis = negative_emojis

    else:
        emojis = neutral_emojis

    # Choose 1 or 2 emojis
    number = random.choice([1, 1, 1, 2])

    selected = random.sample(emojis, number)

    # Different placement styles
    style = random.choice([1, 2, 3])

    if style == 1:
        return text + " " + "".join(selected)

    elif style == 2:
        return "".join(selected) + " " + text

    else:
        words = text.split()

        if len(words) > 3:
            position = random.randint(1, len(words) - 1)
            words.insert(position, "".join(selected))
            return " ".join(words)

        return text + " " + "".join(selected)


# ============================================================
# CREATE AUGMENTED DATA
# ============================================================

augmented_rows = []

for _, row in df.iterrows():

    text = str(row["text"])
    sentiment = row["sentiment"]

    # Keep ORIGINAL example
    augmented_rows.append({
        "text": text,
        "sentiment": sentiment,
        "source": "original"
    })

    # Add 2 emoji-enhanced versions
    for _ in range(2):

        new_text = add_emojis(text, sentiment)

        augmented_rows.append({
            "text": new_text,
            "sentiment": sentiment,
            "source": "emoji_augmented"
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

enhanced_df = pd.DataFrame(augmented_rows)

# Remove exact duplicates
enhanced_df = enhanced_df.drop_duplicates(
    subset=["text", "sentiment"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

enhanced_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

# ============================================================
# REPORT
# ============================================================

print("\n==========================================")
print("EMOJI ENHANCED DATASET CREATED")
print("==========================================")

print("Total records:", len(enhanced_df))

print("\nSentiment distribution:")
print(enhanced_df["sentiment"].value_counts())

print("\nSource distribution:")
print(enhanced_df["source"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)