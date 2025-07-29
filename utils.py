import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import TicketDataset

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df["text"].tolist(), df["label"].tolist()

def prepare_datasets(train_file):
    texts, labels = load_data(train_file)
    x_train, x_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, stratify=labels)
    return TicketDataset(x_train, y_train), TicketDataset(x_val, y_val)
