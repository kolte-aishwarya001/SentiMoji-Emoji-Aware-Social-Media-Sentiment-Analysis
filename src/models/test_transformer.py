from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully.")

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

print("Model loaded successfully.")

print("\n========================================")
print("TRANSFORMER TEST")
print("========================================")
print("Model:", MODEL_NAME)
print("Number of labels:", model.config.num_labels)
print("Tokenizer vocabulary:", tokenizer.vocab_size)
print("Status: READY")