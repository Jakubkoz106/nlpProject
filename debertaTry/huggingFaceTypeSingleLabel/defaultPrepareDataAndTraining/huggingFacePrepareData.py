"""Prepare data 

DATASET: google-research-datasets/go_emotions
MODEL: https://huggingface.co/microsoft/deberta-v3-base


deberta-v3-base
google-research-datasets/go_emotions
singlelabel classification


"""


from datasets import load_dataset, Sequence
import numpy as np
import json
# import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
# from sklearn.model_selection import train_test_split
# from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
# import torch
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
from collections import Counter
import itertools
from datasets import Value


def load_goemotions_dataset():
    dataset = load_dataset("google-research-datasets/go_emotions")
    label_names = dataset["train"].features["labels"].feature.names
    print("\nUnikalne występujące emocje w zbiorze danych (count: "+str(len(label_names))+"):")
    print(label_names)
    return dataset, label_names


def plot_label_frequencies(dataset, label_names):
    label_counter = Counter(itertools.chain.from_iterable(dataset["train"]["labels"]))
    counts = [label_counter[i] for i in range(len(label_names))]

    plt.figure(figsize=(12, 6))
    plt.bar(label_names, counts)
    plt.xticks(rotation=90)
    plt.title("Częstotliwość emocji w zbiorze treningowym GoEmotions")
    plt.tight_layout()
    plt.show()


def plot_message_lengths(dataset):
    lengths = [len(text.split()) for text in dataset["train"]["text"]]
    plt.hist(lengths, bins=30)
    plt.title("Długość wiadomości (liczba słów w wiadomości)")
    plt.xlabel("Liczba słów")
    plt.ylabel("Liczba przykładów")
    plt.tight_layout()
    plt.show()


def plot_top_multilabel_combinations(dataset, label_names, top_n=20):
    multi_label_combos = []
    for labels in dataset["train"]["labels"]:
        if len(labels) > 1:
            combo = "_".join(sorted([label_names[i] for i in labels]))
            multi_label_combos.append(combo)

    combo_counts = Counter(multi_label_combos)
    print(f"Liczba unikalnych kombinacji multi-label: {len(combo_counts)}")
    top_combos = combo_counts.most_common(top_n)
    combo_labels, counts = zip(*top_combos)

    plt.figure(figsize=(14, 6))
    plt.bar(combo_labels, counts)
    plt.xticks(rotation=90)
    plt.title(f"Top {top_n} najczęstszych kombinacji emocji (multi-label) w GoEmotions")
    plt.xlabel("Kombinacje emocji")
    plt.ylabel("Liczba przykładów")
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()


def print_sample_texts_with_emotion(dataset, label_names, emotion_name, n=5):
    emotion_index = label_names.index(emotion_name)
    examples = [ex["text"] for ex in dataset["train"] if emotion_index in ex["labels"]]
    print(f"\nPrzykładowe teksty z emocją '{emotion_name}':\n")
    for ex in examples[:n]:
        print("-", ex)



def prepare_goemotions_single_label(dataset):
    from transformers import AutoTokenizer
    from datasets import Value
    import torch

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    def tokenize(example):
        # Zakładamy, że każda etykieta ma tylko jeden element
        encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
        encoding["label"] = example["labels"][0]  # tylko jedna etykieta
        return encoding

    tokenized = dataset.map(tokenize, batched=False)

    # Rzutuj na typ całkowity (long)
    tokenized = tokenized.cast_column("label", Value("int64"))

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "text"])

    print("\n✅ Dane GoEmotions (single-label) przygotowane do modelu!")
    print(f"- Rozmiar zbioru treningowego: {len(tokenized['train'])}")
    print(f"- Przykład wejścia:\n{tokenized['train'][0]}")

    return tokenized




def analyze_token_lengths_hf(dataset):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Wyciągamy długości tokenów w zbiorze treningowym
    token_lengths = [len(tokenizer.tokenize(text)) for text in dataset["train"]["text"]]

    # Statystyki
    print("\n📊 Statystyki długości tokenów (GoEmotions):")

    print(f"Średnia: {np.mean(token_lengths):.2f}")
    print(f"Mediana: {np.median(token_lengths):.0f}")
    print(f"Maksymalna długość: {np.max(token_lengths)}")

    print("\n📉 Procent wiadomości krótszych niż:")
    for length in [32, 64, 128, 256, 512]:
        percent = (np.array(token_lengths) < length).mean() * 100
        print(f" - {length} tokenów: {percent:.2f}%")

    # Wykres
    plt.figure(figsize=(10, 5))
    plt.hist(token_lengths, bins=30, color="lightcoral", edgecolor="black")
    plt.axvline(128, color='red', linestyle='--', label="128 tokenów")
    plt.axvline(64, color='green', linestyle='--', label="64 tokeny")
    plt.title("Rozkład długości tokenów w GoEmotions")
    plt.xlabel("Liczba tokenów")
    plt.ylabel("Liczba przykładów")
    plt.legend()
    plt.tight_layout()
    plt.show()



# ====== GŁÓWNA SEKCJA URUCHOMIENIA ======
if __name__ == "__main__":
    dataset, label_names = load_goemotions_dataset()

    # plot_label_frequencies(dataset, label_names)
    # plot_message_lengths(dataset)
    # plot_top_multilabel_combinations(dataset, label_names, top_n=20)
    # print_sample_texts_with_emotion(dataset, label_names, "joy", n=5)
    # analyze_token_lengths_hf(dataset)

    tokenized_datasets = prepare_goemotions_single_label(dataset)
    train_ds = tokenized_datasets["train"]
    val_ds = tokenized_datasets["validation"]


    tokenized_datasets.save_to_disk("data/goemotions_single_tokenized")
    with open("data/goemotions_labels.json", "w") as f:
        json.dump(label_names, f)

    print("✅ Dane zostały zapisane do folderu 'data/'")