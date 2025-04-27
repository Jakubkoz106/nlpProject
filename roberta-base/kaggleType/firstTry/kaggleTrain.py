import json
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score, \
    recall_score
from datasets import load_from_disk
import matplotlib.pyplot as plt
import seaborn as sns

def train_roberta_single_label(train_dataset, val_dataset, num_labels):


    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir="results_kaggle",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_dir="./logs_kaggle",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        weight_decay=0.01,
        warmup_ratio=0.1
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_micro": f1_score(labels, preds, average="micro"),
            "f1_weighted": f1_score(labels, preds, average="weighted"),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0)
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

    print("\n📊 Generuję klasyfikację na zbiorze walidacyjnym...")
    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids

    print("\n📋 Classification Report:")
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)
    print(report)

    with open("results_kaggle/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, xticklabels=class_names, yticklabels=class_names, cmap="Blues", fmt="d")
    plt.xlabel("Predykcja")
    plt.ylabel("Prawdziwa etykieta")
    plt.title("Macierz pomyłek (confusion matrix)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results_kaggle/confusion_matrix.png")
    print("✅ Zapisano confusion matrix i classification report.")

    # 🧾 Zapis metryk do JSON
    metrics = compute_metrics((preds_output.predictions, preds_output.label_ids))
    with open("results_kaggle/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

# ====== Główna sekcja ======
if __name__ == "__main__":
    train_dataset = load_from_disk("data/kaggle_train")
    val_dataset = load_from_disk("data/kaggle_val")
    with open("data/kaggle_labels.json", "r") as f:
        class_names = json.load(f)

    train_roberta_single_label(train_dataset, val_dataset, num_labels=len(class_names))