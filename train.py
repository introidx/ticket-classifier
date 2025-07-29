from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from model import get_model
from utils import prepare_datasets
from config import TRAIN_FILE, EPOCHS, BATCH_SIZE, MODEL_NAME

def train():
    model = get_model()
    train_ds, val_ds = prepare_datasets(TRAIN_FILE)

    training_args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        load_best_model_at_end=True,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds
    )

    trainer.train()
    trainer.save_model("saved_model")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained("saved_model")

if __name__ == "__main__":
    train()
