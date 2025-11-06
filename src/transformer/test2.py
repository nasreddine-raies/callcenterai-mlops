import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Ton modèle hébergé sur Hugging Face
model_id = "nsayer/mon_modele"

# Chargement du tokenizer et du modèle
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Création d’un pipeline de classification
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Exemple de ticket client
texte_ticket = "hardware."

# Prédiction
resultat = classifier(texte_ticket)

print("Résultat :", resultat)
