"""
Roberta-base na GoEmotions + pełny pakiet wykresów
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    multilabel_confusion_matrix,
    precision_recall_curve, average_precision_score
)

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


# ───────────────────────────────────────── util ───────────────────────────────────────── #

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ────────────────────────────── funkcje do rysowania wykresów ─────────────────────────── #

def plot_labels_per_sample(labels, save_path):
    counts = labels.sum(axis=1)
    plt.figure()
    plt.hist(counts, bins=range(0, counts.max() + 2), align="left", rwidth=0.8)
    plt.xlabel("Liczba etykiet w próbce")
    plt.ylabel("Liczba próbek")
    plt.title("Rozkład liczby etykiet na przykład")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_per_label_metrics(y_true, y_pred, label_names, save_path):
    f1  = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, zero_division=0)

    x = np.arange(len(label_names))
    width = 0.27
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, prec,  width, label="Precision")
    plt.bar(x,          rec,   width, label="Recall")
    plt.bar(x + width, f1,    width, label="F1")
    plt.xticks(x, label_names, rotation=90)
    plt.ylabel("Wartość")
    plt.title("Per-label precision / recall / F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_aggregated_confusion(y_true, y_pred, save_path):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    agg = mcm.sum(axis=0)               # [[TN, FP], [FN, TP]]
    agg_norm = agg / agg.sum()
    plt.figure()
    sns.heatmap(agg_norm, annot=True, fmt=".2f",
                xticklabels=["0 (pred.)", "1 (pred.)"],
                yticklabels=["0 (true)",  "1 (true)"],
                cmap="Blues")
    plt.title("Zagregowana macierz pomyłek (znormalizowana)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_threshold_curve(thresholds, f1_scores, save_path):
    plt.figure()
    plt.plot(thresholds, f1_scores, marker="o")
    plt.xlabel("Próg decyzyjny")
    plt.ylabel("F1-macro")
    plt.title("F1-macro vs próg")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve_micro(y_true, probs, save_path):
    precision, recall, _ = precision_recall_curve(y_true.ravel(), probs.ravel())
    ap = average_precision_score(y_true, probs, average="micro")
    plt.figure()
    plt.plot(recall, precision, lw=2, label=f"micro-avg AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Krzywa Precision-Recall (micro-average)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_cooccurrence_heatmap(y_true, label_names, save_path):
    A = y_true.astype(bool)
    inter = (A.T @ A).astype(float)
    union = (A.sum(axis=0) + A.sum(axis=0)[:, None] - inter)
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union == 0, 0, inter / union)
    np.fill_diagonal(jaccard, np.nan)
    plt.figure(figsize=(10, 8))
    sns.heatmap(jaccard, cmap="OrRd", xticklabels=label_names,
                yticklabels=label_names, vmin=0, vmax=1, square=True)
    plt.title("Heatmapa współwystępowania etykiet (Jaccard)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_all_plots(y_true, y_pred, probs, label_names, thresholds, f1_scores):
    ensure_dir("results_goemotions/plots")
    plot_labels_per_sample(y_true, "results_goemotions/plots/labels_per_sample.png")
    plot_per_label_metrics(y_true, y_pred, label_names,
                           "results_goemotions/plots/per_label_metrics.png")
    plot_aggregated_confusion(y_true, y_pred,
                              "results_goemotions/plots/confusion_heatmap.png")
    plot_threshold_curve(thresholds, f1_scores,
                         "results_goemotions/plots/f1_vs_threshold.png")
    plot_pr_curve_micro(y_true, probs,
                        "results_goemotions/plots/pr_curve_micro.png")
    plot_cooccurrence_heatmap(y_true, label_names,
                              "results_goemotions/plots/label_cooccurrence.png")


# ───────────────────────────────────── trening + tuning ───────────────────────────────── #

def train_roberta_multi_label(train_dataset, val_dataset):
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=28,
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir="results_goemotions",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        learning_rate=2e-5,
        logging_dir="./logs_goemotions",
        load_best_model_at_end=True,
        fp16=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        warmup_ratio=0.1,
        weight_decay=0.01
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs > 0.5).astype(int)

        metrics = {
            "f1_micro":    f1_score(labels, preds, average="micro",   zero_division=0),
            "f1_macro":    f1_score(labels, preds, average="macro",   zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "accuracy_samples": np.mean(np.all(preds == labels, axis=1)),
            "exact_match_ratio": (preds == labels).all(axis=1).mean()
        }
        with open("results_goemotions/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
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

    best_threshold = tune_threshold_and_plot(model, val_dataset)


def tune_threshold_and_plot(model, val_dataset,
                            thresholds=np.arange(0.3, 0.61, 0.05)):
    all_logits, all_labels = [], []
    device = next(model.parameters()).device

    for i in range(0, len(val_dataset), 32):
        batch = val_dataset[i:i+32]
        inputs = {k: torch.tensor(batch[k]).to(device)
                  for k in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            logits = model(**inputs).logits.cpu()
        all_logits.append(logits)
        all_labels.append(torch.tensor(batch["labels"]))

    logits = torch.cat(all_logits).numpy()
    true_labels = torch.cat(all_labels).numpy()
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    f1_scores = []
    best_f1, best_threshold = 0, 0.5

    print("\n🔎 Testowanie progów decyzyjnych:")
    for t in thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
        f1_scores.append(f1)
        print(f" - próg={t:.2f} → f1_macro={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    print(f"\n✅ Najlepszy próg: {best_threshold:.2f} z f1_macro={best_f1:.4f}")

    final_preds = (probs > best_threshold).astype(int)
    report = classification_report(true_labels, final_preds,
                                   target_names=label_names, zero_division=0)
    with open("results_goemotions/classification_report.txt", "w",
              encoding="utf-8") as f:
        f.write(report)

    mcm = multilabel_confusion_matrix(true_labels, final_preds)
    with open("results_goemotions/confusion_matrices.txt", "w",
              encoding="utf-8") as f:
        for i, matrix in enumerate(mcm):
            f.write(f"\nConfusion Matrix - {label_names[i]}\n")
            f.write(np.array2string(matrix))
            f.write("\n")

    # ──────────── wykresy ────────────
    generate_all_plots(true_labels, final_preds, probs,
                       label_names, thresholds, f1_scores)

    return best_threshold


# ─────────────────────────────────────────── main ─────────────────────────────────────── #

if __name__ == "__main__":
    tokenized_datasets = load_from_disk("data/goemotions_tokenized")
    train_ds = tokenized_datasets["train"]
    val_ds   = tokenized_datasets["validation"]

    with open("data/goemotions_labels.json", "r") as f:
        label_names = json.load(f)

    print(f"✅ Wczytano dane: train={len(train_ds)}, val={len(val_ds)}")
    ensure_dir("results_goemotions")
    ensure_dir("results_goemotions/plots")

    train_roberta_multi_label(train_ds, val_ds)
