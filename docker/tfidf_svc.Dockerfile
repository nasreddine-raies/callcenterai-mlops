# Utilise une image Python officielle
FROM python:3.11-slim

# Crée le répertoire de travail
WORKDIR /app

# Copie le requirements.txt
COPY requirements.txt .

# Installe les dépendances
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copie le code source et les artefacts du modèle
COPY src/tfidf_svc/ src/tfidf_svc/
COPY src/tfidf_svc/tfidf_vectorizer.joblib src/tfidf_svc/tfidf_vectorizer.joblib
COPY src/tfidf_svc/tfidf_svc_model.joblib src/tfidf_svc/tfidf_svc_model.joblib

# Expose le port de l'API
EXPOSE 8000

# Commande pour démarrer le serveur FastAPI
CMD ["uvicorn", "src.tfidf_svc.app:app", "--host", "0.0.0.0", "--port", "8000"]