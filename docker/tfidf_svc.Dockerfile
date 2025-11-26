# Base image
FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the FastAPI app
COPY src/tfidf_svc/app.py ./app.py

# Copy MLflow data
COPY mlruns/ ./mlruns

# Expose ports
EXPOSE 8000 5000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
