# from datasets import load_dataset
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# pip install kagglehub
import kagglehub                        
from kagglehub import KaggleDatasetAdapter 
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
# import torch
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

def load_dataset_kaggle(dataset_handle: str, file_path: str) -> pd.DataFrame:
    """
    Pobiera plik z Kaggle i zwraca go jako DataFrame.
    """
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,      # zwraca pandas DataFrame
        dataset_handle,
        file_path,
        pandas_kwargs={"encoding": "utf-8"}  # ew. dodatkowe parametry
    )
    print("Podgląd danych:")
    print(df.head())
    print("\nInformacje o kolumnach:")
    print(df.info())
    print("\nUnikalne labele emocji:")
    print(df["Emotion"].unique())
    return df


def plot_emotion_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=df["Emotion"], order=df["Emotion"].value_counts().index)
    plt.title("Rozkład klas emocji")
    plt.xticks(rotation=45)
    plt.xlabel("Emocja")
    plt.ylabel("Liczba przykładów")
    plt.tight_layout()
    plt.show()


def plot_text_lengths(df):
    # print(df.columns)
    df["Text_Length"] = df["text"].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Text_Length"], bins=30, kde=True)
    plt.title("Długość wiadomości (liczba słów)")
    plt.xlabel("Liczba słów")
    plt.ylabel("Liczba wiadomości")
    plt.tight_layout()
    plt.show()


def show_examples_for_emotion(df, emotion_label, n=5):
    print(f"\nPrzykładowe wiadomości dla emocji: {emotion_label}\n")
    examples = df[df["Emotion"] == emotion_label]["text"].head(n)
    for i, text in enumerate(examples, 1):
        print(f"{i}. {text}")


def checkMultilabelData(df):
    # Zlicz, ile etykiet znajduje się w każdej komórce
    df["label_count"] = df["Emotion"].apply(lambda x: len(str(x).split(",")))

    # Sprawdź, ile przykładów ma więcej niż 1 etykietę
    multi_label_count = df[df["label_count"] > 1].shape[0]
    total = df.shape[0]

    print(f"Liczba przykładów multi-label: {multi_label_count} z {total} ({(multi_label_count / total) * 100:.2f}%)")


def prepare_data_for_model(df):
    text_col = [col for col in df.columns if "text" in col.lower()][0]
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Emotion"])

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    def tokenize(example):
        return tokenizer(example[text_col], truncation=True, padding="max_length", max_length=128)

    train_dataset = Dataset.from_pandas(train_df[["label", text_col]])
    val_dataset = Dataset.from_pandas(val_df[["label", text_col]])

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = train_dataset.select(range(200000))

    print("\n✅ Dane z Kaggle przygotowane do modelu!")
    print(f"- Liczba klas emocji: {len(label_encoder.classes_)}")
    print(f"- Nazwy klas: {list(label_encoder.classes_)}")
    print(f"- Rozmiar zbioru treningowego: {len(train_dataset)}")
    print(f"- Rozmiar zbioru walidacyjnego: {len(val_dataset)}")
    print(f"- Przykład wejścia:\n{train_dataset[0]}")

    return train_dataset, val_dataset, label_encoder.classes_


def analyze_token_lengths(df):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    text_col = [col for col in df.columns if "text" in col.lower()][0]

    token_lengths = df[text_col].apply(lambda x: len(tokenizer.tokenize(str(x))))

    print("\n📊 Statystyki długości tokenów:")
    print(token_lengths.describe())

    print("\n📉 Procent wiadomości krótszych niż:")
    for length in [32, 64, 128, 256, 512]:
        percent = (token_lengths < length).mean() * 100
        print(f" - {length} tokenów: {percent:.2f}%")

    # Wykres
    plt.figure(figsize=(10, 5))
    plt.hist(token_lengths, bins=30, color="skyblue", edgecolor="black")
    plt.axvline(128, color='red', linestyle='--', label="128 tokenów")
    plt.axvline(64, color='green', linestyle='--', label="64 tokeny")
    plt.title("Rozkład długości tokenów po tokenizacji")
    plt.xlabel("Liczba tokenów")
    plt.ylabel("Liczba wiadomości")
    plt.legend()
    plt.tight_layout()
    plt.show()




# ====== Główna sekcja ======
if __name__ == "__main__":
    DATASET_HANDLE = "simaanjali/emotion-analysis-based-on-text"   # <-- repozytorium Kaggle
    FILE_PATH      = "emotion_sentiment_dataset.csv"               #
    
    df = load_dataset_kaggle(DATASET_HANDLE, FILE_PATH)

    # plot_emotion_distribution(df)
    # plot_text_lengths(df)
    # show_examples_for_emotion(df, emotion_label="hate", n=5)
    # checkMultilabelData(df)
    # analyze_token_lengths(df)

    train_ds, val_ds, class_names = prepare_data_for_model(df)

    train_ds.save_to_disk("data/kaggle_train")
    val_ds.save_to_disk("data/kaggle_val")
    with open("data/kaggle_labels.json", "w") as f:
        json.dump(list(class_names), f)

    print("✅ Dane zapisane do katalogu 'data/'")
