from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, LABELS, MAX_LENGTH

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TicketDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = [LABELS.index(label) for label in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item["labels"] = self.labels[idx]
        return item
