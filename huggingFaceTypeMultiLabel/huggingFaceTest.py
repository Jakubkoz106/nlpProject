import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import json

# Ścieżki do modelu i etykiet
MODEL_PATH = "results_goemotions/checkpoint-4071"  # ← wpisz swój checkpoint
LABELS_PATH = "data/goemotions_labels.json"        # ← ścieżka do listy etykiet

# Wczytaj etykiety
with open(LABELS_PATH, "r") as f:
    label_names = json.load(f)

# Wczytaj tokenizer i model
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# 🔮 Funkcja predykcji
def predict_emotions(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().numpy()
    for i in range(len(probs)):
        print(probs[i],label_names[i], end=", ")

    # Zwróć wszystkie etykiety, których prawdopodobieństwo > threshold
    predicted_labels = [label_names[i] for i, prob in enumerate(probs) if prob > threshold]
    return predicted_labels


texts = [
    "I'm so excited for the weekend!",
    "Why would you say something like that? I'm really upset.",
    "Thanks a lot, I really appreciate your help.",
    "I feel confused and frustrated.",
    "Fuck you motherfucker i will kill you",
"i feel jealous becasue i wanted that kind of love the true connection between two souls and i wanted",
    "i feel like they hated me since then",
    "i feel so comfortable around him",
    "i said without emotion while feeling a freaked out fearful anxiety welling up in my chest",
    "im so excited but feeling kind of shy about it smile",
    "i was just yesterday feeling uncomfortable with highschool sigh",
    "	i am not feeling fearful",
    "i feel like a greedy person for liking two people",
    "	im really feeling good"
]

for text in texts:
    print(f"📝 \"{text}\"")
    print(f"\n🎯 Emocje: {predict_emotions(text)}\n")
