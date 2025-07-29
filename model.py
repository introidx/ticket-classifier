from transformers import AutoModelForSequenceClassification
from config import MODEL_NAME, LABELS

def get_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
