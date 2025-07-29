from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import MODEL_NAME, LABELS, MAX_LENGTH

# Load model and tokenizer from saved directory
model_path = r"C:\Users\praka\Projects\ticket-classifier\saved_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Make sure model is in eval mode
model.eval()

def predict(text: str):
    # Preprocess the input text
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(dim=1).item()
        predicted_label = LABELS[predicted_class_id]
        return predicted_label

# --- Example usage ---
if __name__ == "__main__":
    ticket_id = "JIRA-2025"
    text = "Why does the invoice show a higher charge than what was estimated before dispatch?"
    predicted_category = predict(text)
    print(f"Ticket ID: {ticket_id}")
    print(f"Predicted Category: {predicted_category}")
