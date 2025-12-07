# ================================
# Dockerfile complet Transformer Service
# ================================

FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Mettre à jour pip et configurer le timeout
RUN pip install --no-cache-dir --upgrade pip \
    && pip config set global.timeout 200

# Installer dépendances système nécessaires pour torch + transformers
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    libomp-dev \
    git \
    && apt-get clean

# Copy requirements and install all dependencies
COPY src/transformer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout 200

# Créer un dossier pour le cache Hugging Face
RUN mkdir -p /app/cache
ENV HF_HOME=/app/cache

# Pré-télécharger le modèle Hugging Face au build
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
model_id='nsayer/mon_modele'; \
AutoModelForSequenceClassification.from_pretrained(model_id); \
AutoTokenizer.from_pretrained(model_id)"

# Copier le code de l'application
COPY src/transformer/ /app/

# Exposer le port utilisé par FastAPI/Uvicorn
EXPOSE 8000

# Lancer Uvicorn au démarrage du conteneur
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
