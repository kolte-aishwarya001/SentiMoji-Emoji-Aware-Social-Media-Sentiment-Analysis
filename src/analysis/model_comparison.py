import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# MODEL RESULTS
# ============================================================

results = {
    "Model": [
        "TF-IDF + Logistic Regression",
        "TF-IDF + SVM",
        "DistilBERT"
    ],

    "Validation Accuracy": [
        0.6715,
        0.6400,
        0.72625
    ],

    "Test Accuracy": [
        0.5882,
        0.5559,
        0.6767
    ],

    "Test Precision": [
        0.59,
        0.56,
        0.6779
    ],

    "Test Recall": [
        0.59,
        0.56,
        0.6767
    ],

    "Test F1": [
        0.59,
        0.55,
        0.6769
    ]
}


df = pd.DataFrame(results)


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


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print()

print(df.to_string(index=False))


# ============================================================
# CONVERT TO PERCENTAGE
# ============================================================

percentage_df = df.copy()

metric_columns = [
    "Validation Accuracy",
    "Test Accuracy",
    "Test Precision",
    "Test Recall",
    "Test F1"
]

for column in metric_columns:

    percentage_df[column] = (
        percentage_df[column] * 100
    ).round(2)


print("\n")
print("=" * 70)
print("PERFORMANCE IN PERCENTAGE")
print("=" * 70)

print()

print(
    percentage_df.to_string(index=False)
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

percentage_df.to_csv(
    csv_path,
    index=False
)


print("\nComparison saved to:")
print(csv_path)


# ============================================================
# TEST ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    percentage_df["Model"],
    percentage_df["Test Accuracy"]
)

plt.ylabel("Accuracy (%)")

plt.xlabel("Model")

plt.title(
    "Test Accuracy Comparison of Sentiment Analysis Models"
)

plt.ylim(0, 100)

plt.xticks(
    rotation=15,
    ha="right"
)

for index, value in enumerate(
    percentage_df["Test Accuracy"]
):

    plt.text(
        index,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )


plt.tight_layout()


accuracy_graph = os.path.join(
    OUTPUT_DIR,
    "test_accuracy_comparison.png"
)

plt.savefig(
    accuracy_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nAccuracy graph saved to:")
print(accuracy_graph)


# ============================================================
# ALL METRICS GRAPH
# ============================================================

metrics = [
    "Test Accuracy",
    "Test Precision",
    "Test Recall",
    "Test F1"
]

x = range(
    len(percentage_df["Model"])
)

width = 0.18

plt.figure(figsize=(12, 7))

for i, metric in enumerate(metrics):

    values = percentage_df[metric]

    positions = [
        value + (i - 1.5) * width
        for value in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )


plt.xlabel("Model")

plt.ylabel("Score (%)")

plt.title(
    "Performance Comparison of Sentiment Analysis Models"
)

plt.xticks(
    list(x),
    percentage_df["Model"],
    rotation=15,
    ha="right"
)

plt.ylim(0, 100)

plt.legend()

plt.tight_layout()


metrics_graph = os.path.join(
    OUTPUT_DIR,
    "model_metrics_comparison.png"
)

plt.savefig(
    metrics_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nMetrics graph saved to:")
print(metrics_graph)


# ============================================================
# FIND BEST MODEL
# ============================================================

best_index = percentage_df[
    "Test Accuracy"
].idxmax()

best_model = percentage_df.loc[
    best_index,
    "Model"
]

best_accuracy = percentage_df.loc[
    best_index,
    "Test Accuracy"
]


print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Model: {best_model}"
)

print(
    f"Test Accuracy: {best_accuracy:.2f}%"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)

print("\nResults directory:")
print(OUTPUT_DIR)