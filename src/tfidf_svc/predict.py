import joblib

# Charge le modèle et le vectorizer
vectorizer = joblib.load('src/tfidf_svc/tfidf_vectorizer.joblib')
clf = joblib.load('src/tfidf_svc/tfidf_svc_model.joblib')

# Exemple de ticket à prédire
#ticket = "My disk space is full, can you increase it?"
ticket=input("Enter a ticket: ")
# Vectorise le texte
X = vectorizer.transform([ticket])

# Prédit la catégorie
label = clf.predict(X)[0]
proba = clf.predict_proba(X)[0].max()  # Confiance

print(f"Catégorie prédite : {label}")
print(f"Confiance : {proba:.2f}")