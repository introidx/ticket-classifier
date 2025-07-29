# 🧾 Ticket Classifier [Vibe Coded]

This project fine-tunes a transformer model to classify customer support tickets (e.g., billing-related vs shipping-related) and uses OpenAI's API to extract structured information like summary, purchase order ID, and tracking number.

---

## 🚀 Features

- Classify JIRA support ticket text into categories (like `billing`, `shipping`)
- Fine-tune your own model using a CSV dataset
- Predict new tickets using your trained model
- Extract structured information using OpenAI API:
  - Summary (1-liner)
  - Purchase Order ID (poId, purchaseOrderId, purchase order id, etc.)
  - Tracking Number (trackingId, tracking_number, tracking number, etc.)

---

## 📁 Project Structure

```
ticket-classifier/
├── config.py                # Global config (model name, label enums)
├── dataset.py               # Custom Dataset class
├── model.py                 # Model loader (DistilBERT classifier)
├── train.py                 # Fine-tunes model using labeled CSV
├── predict.py               # Predicts label from new ticket text
├── predict_json.py          # Returns full structured JSON output
├── summarizer.py            # Extracts summary, PO ID, tracking number using OpenAI
├── utils.py                 # Data loading, preprocessing utils
├── requirements.txt         # Dependencies
├── .gitignore               # Excludes model, logs, venv
├── README.md                # You're reading it!
└── data/
    └── labeled_tickets.csv  # Your training dataset
```

---

## 🛠 Setup Instructions

### 1. 🧪 Create a Conda Environment

```bash
conda create -p venv python=3.12
conda activate venv/
```

### 2. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. 🔐 Set Up OpenAI API

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo
TEMPERATURE=0.3
```

---

## 🏋️‍♂️ Train the Classifier

```bash
python train.py
```

This will create a trained model inside the `saved_model/` folder.

---

## 🔮 Predict and Extract Full Ticket Information

Run this to get the full structured output:

```bash
python predict_json.py
```

Example output:

```json
{
  "ticketId": "JIRA-2025",
  "category": "billing",
  "purchaseOrderId": "PO-12345",
  "trackingNumber": "TRK-67890",
  "summary": "Customer was billed more than expected after delivery."
}
```

---

## ✅ Coming Soon

- [ ] Batch prediction from CSV
- [ ] FastAPI REST endpoint
- [ ] Feedback loop for retraining

---

## 🧠 Powered By

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [OpenAI Chat API](https://platform.openai.com/docs/)
- PyTorch or TensorFlow backend

---
