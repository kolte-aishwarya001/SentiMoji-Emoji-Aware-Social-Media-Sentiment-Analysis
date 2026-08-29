import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score
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

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "emoji_enhanced_train.csv"
)

VALIDATION_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "final_validation.csv"
)

TEST_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "final_test.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_logistic_regression.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

train_df = pd.read_csv(
    TRAIN_FILE,
    encoding="utf-8"
)

validation_df = pd.read_csv(
    VALIDATION_FILE,
    encoding="utf-8"
)

test_df = pd.read_csv(
    TEST_FILE,
    encoding="utf-8"
)


print("\nDataset sizes:")
print("Training:", train_df.shape)
print("Validation:", validation_df.shape)
print("Test:", test_df.shape)


# ============================================================
# FEATURES / LABELS
# ============================================================

X_train = train_df["text"].fillna("").astype(str)
y_train = train_df["sentiment"]

X_validation = validation_df["text"].fillna("").astype(str)
y_validation = validation_df["sentiment"]

X_test = test_df["text"].fillna("").astype(str)
y_test = test_df["sentiment"]


# ============================================================
# MEMORY-EFFICIENT TF-IDF
# ============================================================

print("\nCreating TF-IDF model...")

tfidf = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_features=50000,
    sublinear_tf=True
)


# ============================================================
# TRANSFORM TRAIN DATA
# ============================================================

print("Creating training TF-IDF matrix...")

X_train_tfidf = tfidf.fit_transform(X_train)

print(
    "Training matrix:",
    X_train_tfidf.shape
)


# ============================================================
# VALIDATION TRANSFORMATION
# ============================================================

print("Transforming validation data...")

X_validation_tfidf = tfidf.transform(
    X_validation
)

print(
    "Validation matrix:",
    X_validation_tfidf.shape
)


# ============================================================
# TEST TRANSFORMATION
# ============================================================

print("Transforming test data...")

X_test_tfidf = tfidf.transform(
    X_test
)

print(
    "Test matrix:",
    X_test_tfidf.shape
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

classifier = LogisticRegression(
    max_iter=500,
    class_weight="balanced",
    solver="lbfgs",
    
)

classifier.fit(
    X_train_tfidf,
    y_train
)

print("Training completed.")


# ============================================================
# VALIDATION
# ============================================================

print("\n========================================")
print("VALIDATION RESULTS")
print("========================================")

validation_predictions = classifier.predict(
    X_validation_tfidf
)

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

print(
    "Validation Accuracy:",
    round(validation_accuracy, 4)
)

print(
    classification_report(
        y_validation,
        validation_predictions
    )
)


# ============================================================
# TEST
# ============================================================

print("\n========================================")
print("TEST RESULTS")
print("========================================")

test_predictions = classifier.predict(
    X_test_tfidf
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print(
    "Test Accuracy:",
    round(test_accuracy, 4)
)

print(
    classification_report(
        y_test,
        test_predictions
    )
)


# ============================================================
# SAVE TF-IDF + CLASSIFIER
# ============================================================

print("\nSaving model...")

model_package = {
    "tfidf": tfidf,
    "classifier": classifier
}

joblib.dump(
    model_package,
    MODEL_FILE,
    compress=3
)

print(
    "Model saved successfully:"
)

print(MODEL_FILE)