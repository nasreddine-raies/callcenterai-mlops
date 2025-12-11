# Base image
FROM python:3.11-slim

WORKDIR /app

# Disable pip cache to reduce memory during build
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# Install system-level dependencies in two steps
# This allows memory cleanup between installations
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      gfortran \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libatlas3-base \
      libopenblas-dev \
      liblapack-dev \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy only requirements first (docker cache optimization)
COPY src/tfidf_svc/requirements.txt ./requirements.txt

# Upgrade pip and install Python dependencies with memory optimization
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --prefer-binary -r requirements.txt

# Now copy the full app
COPY src/tfidf_svc/ .

# Expose service port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
