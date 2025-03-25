from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_from_disk
import json


def train_roberta_multi_label(train_dataset, val_dataset):
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=28,
                                                               problem_type="multi_label_classification")

    training_args = TrainingArguments(
        output_dir="./results_goemotions",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_dir="./logs_goemotions",
        load_best_model_at_end=True,
        fp16=True
        # metric_for_best_model="f1"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs > 0.5).astype(int)

        sample_accuracy = np.mean(np.all(preds == labels, axis=1))

        return {
            "f1_micro": f1_score(labels, preds, average="micro"),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "accuracy": sample_accuracy
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("✅ Trening modelu na GoEmotions zakończony.")


if __name__ == "__main__":
    tokenized_datasets = load_from_disk("data/goemotions_tokenized")
    train_ds = tokenized_datasets["train"]
    val_ds = tokenized_datasets["validation"]

    # Wczytaj etykiety
    with open("data/goemotions_labels.json", "r") as f:
        label_names = json.load(f)

    print(f"✅ Wczytano dane: train={len(train_ds)}, val={len(val_ds)}")

    train_roberta_multi_label(train_ds, val_ds)
