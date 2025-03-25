import json
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_from_disk

def train_roberta_single_label(train_dataset, val_dataset, num_labels):


    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir="results_kaggle",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=2,
        learning_rate=2e-5,
        logging_dir="./logs_kaggle",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("✅ Trening modelu na danych Kaggle zakończony.")

# ====== Główna sekcja ======
if __name__ == "__main__":
    train_dataset = load_from_disk("data/kaggle_train")
    val_dataset = load_from_disk("data/kaggle_val")
    with open("data/kaggle_labels.json", "r") as f:
        class_names = json.load(f)

    train_roberta_single_label(train_dataset, val_dataset, num_labels=len(class_names))