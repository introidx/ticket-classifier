MODEL_NAME = "distilbert-base-uncased"
LABELS = ["billing", "shipping", "other"]
EPOCHS = 4
BATCH_SIZE = 8
MAX_LENGTH = 256
TRAIN_FILE = "data/labeled_tickets.csv"