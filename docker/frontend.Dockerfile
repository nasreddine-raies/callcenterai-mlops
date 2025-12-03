FROM python:3.10-slim

WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/app.py .

# Streamlit tourne par défaut sur le port 8501
EXPOSE 8501

# Lancer streamlit en écoutant sur toutes les interfaces
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]