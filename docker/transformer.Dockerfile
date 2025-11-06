# ==============================
# Stage 1: Base image
# ==============================
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Set environment variables (optional)
ENV MODEL_ID=nsayer/mon_modele
ENV TRANSFORMERS_CACHE=/app/cache

# Run the API with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
