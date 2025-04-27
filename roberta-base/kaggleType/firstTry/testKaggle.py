from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import json

# Wczytaj tokenizer i model z checkpointu
model_path = "results_kaggle/100kLessNeutralAndWith2Epochs"
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Wczytaj etykiety klas
with open("data/kaggle_labels.json", "r") as f:
    class_names = json.load(f)

# 🔮 Funkcja predykcji
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        return class_names[predicted_class_id]

# 🔎 Przykład
texts = [
    "Yesss! I finally got the job!",
"Wow, that's exactly what I needed! Thanks so much!",
"Why did you do that? I'm seriously disappointed.",
"Haha! That cat just slipped off the table!",
"Ugh... this food smells disgusting.",
"Yesss! I finally got the job!",
    "Fuck you motherfucker hi mercedes"
]

for text in texts:
    print(f"📝 \"{text}\" → 🧠 Predykcja: {predict(text)}")
