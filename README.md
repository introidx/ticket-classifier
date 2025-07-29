# 🧾 Ticket Classifier (Billing vs Others)

This project fine-tunes a transformer model to classify customer support tickets (e.g., billing-related vs shipping-related) using labeled JIRA ticket text.

---

## 🚀 Features

- Classify JIRA support ticket text into categories (like `billing`, `shipping`)
- Fine-tune your own model using a CSV dataset
- Predict new tickets using your trained model
- Designed for easy deployment on local or cloud (Azure-ready)

---

## 📁 Project Structure

```
ticket-classifier/
├── config.py              # Global config (model name, label enums)
├── dataset.py             # Custom Dataset class
├── model.py               # Model loader (DistilBERT classifier)
├── train.py               # Fine-tunes model using labeled CSV
├── predict.py             # Predicts label from new ticket text
├── utils.py               # Data loading, preprocessing utils
├── requirements.txt       # Dependencies
├── .gitignore             # Excludes model, logs, venv
├── README.md              # You're reading it!
└── data/
    └── labeled_tickets.csv  # Your training dataset
```

---

## 🛠 Setup Instructions

### 1. 🧪 Create a Conda Environment

```bash
conda create -n ticket-classifier python=3.10 -y
conda activate ticket-classifier
```

### 2. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. ✍️ Prepare Your Training Data

Place your labeled support tickets in:

```
data/labeled_tickets.csv
```

Example:

```csv
ticket_id,text,label
JIRA-101,"Why was I charged more than estimated?",billing
JIRA-102,"Where is my package?",shipping
```

---

## 🏋️‍♂️ Train the Model

```bash
python train.py
```

This will create a trained model inside the `saved_model/` folder.

---

## 🔮 Predict a New Ticket

Edit `predict.py` to change the input ticket text.

Then run:

```bash
python predict.py
```

Example output:

```
Ticket ID: JIRA-999
Predicted Category: billing
```

---

## ✅ Coming Soon (optional)

- [ ] Batch prediction from CSV
- [ ] FastAPI REST endpoint
- [ ] ID extraction (purchaseOrderId, trackingNumber)
- [ ] Feedback loop for retraining

---

## 🧠 Powered By

- [Transformers (Hugging Face)](https://huggingface.co/docs/transformers)
- PyTorch or TensorFlow (backend)
- Sentence classification with `distilbert-base-uncased`

---

## 📬 Questions?

Open an issue or reach out in the discussions tab.
