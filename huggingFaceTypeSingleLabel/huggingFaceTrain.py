from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_from_disk
import json


def train_roberta_single_label(train_dataset, val_dataset):
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=28  # tyle, ile klas emocji
    )

    training_args = TrainingArguments(
        output_dir="results_goemotions_single_without_cast",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_dir="./logs_goemotions_single",
        load_best_model_at_end=True,
        fp16=True,
        metric_for_best_model="accuracy"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_micro": f1_score(labels, preds, average="micro"),
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
    print("✅ Trening single-label GoEmotions zakończony.")



if __name__ == "__main__":
    tokenized_datasets = load_from_disk("data/goemotions_single_tokenized")
    train_ds = tokenized_datasets["train"]
    val_ds = tokenized_datasets["validation"]

    # Wczytaj etykiety
    with open("data/goemotions_labels.json", "r") as f:
        label_names = json.load(f)

    print(f"✅ Wczytano dane: train={len(train_ds)}, val={len(val_ds)}")

    train_roberta_single_label(train_ds, val_ds)
