import re


# ============================================================
# EMOJI DETECTION
# ============================================================

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE
)


def extract_emojis(text):
    """
    Extract emojis from text.
    """

    if not isinstance(text, str):
        return []

    emojis = EMOJI_PATTERN.findall(text)

    result = []

    for group in emojis:
        for char in group:
            result.append(char)

    return result


def count_emojis(text):
    """
    Count emojis in text.
    """

    return len(extract_emojis(text))


def analyze_emojis(text):
    """
    Return emoji analysis information.
    """

    emojis = extract_emojis(text)

    return {
        "emojis": emojis,
        "emoji_count": len(emojis),
        "has_emoji": len(emojis) > 0
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    text = "I absolutely love this movie 😍🔥❤️"

    result = analyze_emojis(text)

    print("Text:")
    print(text)

    print("\nDetected emojis:")
    print(result["emojis"])

    print("\nEmoji count:")
    print(result["emoji_count"])

    print("\nContains emoji:")
    print(result["has_emoji"])