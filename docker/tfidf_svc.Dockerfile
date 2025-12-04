# Base image
FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies for numeric packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc gfortran libatlas3-base libopenblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies with longer timeout
COPY src/tfidf_svc/requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip --default-timeout=1000 install --no-cache-dir -r requirements.txt

# Copy the full TFIDF service (including artifacts)
COPY src/tfidf_svc/ .

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
