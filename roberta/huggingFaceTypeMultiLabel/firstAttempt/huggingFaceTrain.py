from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report, multilabel_confusion_matrix
from datasets import load_from_disk
import json


def train_roberta_multi_label(train_dataset, val_dataset):
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=28,
                                                               problem_type="multi_label_classification")

    training_args = TrainingArguments(
        output_dir="results_goemotions",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_dir="./logs_goemotions",
        load_best_model_at_end=True,
        fp16=True,
        metric_for_best_model="f1_micro",
        warmup_ratio=0.1,
        weight_decay=0.01
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs > 0.5).astype(int)

        metrics = {
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "accuracy_samples": np.mean(np.all(preds == labels, axis=1)),  # Sample-based accuracy
            "exact_match_ratio": (preds == labels).all(axis=1).mean()  # Alias
        }

        with open("results_goemotions/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print("✅ Zapisano metryki do metrics.json.")


        return metrics




    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("✅ Trening modelu na GoEmotions zakończony.")

    def tune_threshold(model, val_dataset, thresholds=np.arange(0.3, 0.61, 0.05)):
        all_logits = []
        all_labels = []

        device = next(model.parameters()).device

        for i in range(0, len(val_dataset), 32):
            batch = val_dataset[i:i+32]
            inputs = {k: torch.tensor(batch[k]).to(device) for k in ["input_ids", "attention_mask"]}
            with torch.no_grad():
                logits = model(**inputs).logits.cpu()
            all_logits.append(logits)
            all_labels.append(torch.tensor(batch["labels"]))

        logits = torch.cat(all_logits).numpy()
        true_labels = torch.cat(all_labels).numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()

        best_f1 = 0
        best_threshold = 0.5

        print("\n🔎 Testowanie progów decyzyjnych:")
        for t in thresholds:
            preds = (probs > t).astype(int)
            f1 = f1_score(true_labels, preds, average="micro", zero_division=0)
            print(f" - próg={t:.2f} → f1_micro={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        print(f"\n✅ Najlepszy próg: {best_threshold:.2f} z f1_micro={best_f1:.4f}")

        final_preds = (probs > best_threshold).astype(int)
        report = classification_report(true_labels, final_preds, target_names=label_names, zero_division=0)
        print("\n📋 Classification report (dla najlepszego progu):")
        print(report)

        with open("results_goemotions/classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        print("\n📊 Macierze pomyłek zapisywane do pliku:")
        mcm = multilabel_confusion_matrix(true_labels, final_preds)
        with open("results_goemotions/confusion_matrices.txt", "w", encoding="utf-8") as f:
            for i, matrix in enumerate(mcm):
                f.write(f"\nConfusion Matrix - {label_names[i]}\n")
                f.write(np.array2string(matrix))
                f.write("\n")


        return best_threshold

    best_threshold = tune_threshold(model, val_dataset)



if __name__ == "__main__":
    tokenized_datasets = load_from_disk("data/goemotions_tokenized")
    train_ds = tokenized_datasets["train"]
    val_ds = tokenized_datasets["validation"]

    # Wczytaj etykiety
    with open("data/goemotions_labels.json", "r") as f:
        label_names = json.load(f)

    print(f"✅ Wczytano dane: train={len(train_ds)}, val={len(val_ds)}")

    train_roberta_multi_label(train_ds, val_ds)
