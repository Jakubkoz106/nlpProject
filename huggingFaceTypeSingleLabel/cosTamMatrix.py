import torch
import numpy as np
import json
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Ścieżki do modelu i danych
MODEL_PATH = "results_goemotions_single_with_cast_int64/checkpoint-4071"  # Twój checkpoint
DATASET_PATH = "data/goemotions_single_tokenized"
LABELS_PATH = "data/goemotions_labels.json"

# Wczytaj etykiety (nazwy klas)
with open(LABELS_PATH, "r") as f:
    label_names = json.load(f)

# Wczytaj zbiór walidacyjny
dataset = load_from_disk(DATASET_PATH)
val_dataset = dataset["validation"]

# Wczytaj tokenizer i model
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


# 🔮 Funkcja do predykcji dla jednej próbki
def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        return torch.argmax(logits, dim=1).item()


# 🔁 Zbierz prawdziwe i przewidziane etykiety
y_true = []
y_pred = []

for example in val_dataset:
    text = example["text"]
    true_label = example["label"]
    pred_label = predict_label(text)

    y_true.append(true_label)
    y_pred.append(pred_label)

# 🧮 Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, xticklabels=label_names, yticklabels=label_names, annot=False, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – GoEmotions (single-label)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 📊 Raport tekstowy
print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
