# ================================
# Optimized Transformer Service Dockerfile
# ================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Disable pip cache to reduce memory during build
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# ----------------------------------------------------
# 1) Install system dependencies in separate steps
# This allows memory to be freed between installations
# ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas-dev \
        libomp-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ----------------------------------------------------
# 2) Upgrade pip + setuptools + wheel
# ----------------------------------------------------
RUN pip install --upgrade pip setuptools wheel \
    && pip config set global.timeout 200

# ----------------------------------------------------
# 3) Copy only requirements first (Docker cache layer)
# ----------------------------------------------------
COPY src/transformer/requirements.txt ./requirements.txt

# ----------------------------------------------------
# 4) Install PyTorch separately (large package)
# Use CPU wheels to reduce size
# ----------------------------------------------------
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ----------------------------------------------------
# 5) Install remaining Python dependencies
# ----------------------------------------------------
RUN pip install --prefer-binary -r requirements.txt

# ----------------------------------------------------
# 5) Hugging Face cache directory
# ----------------------------------------------------
RUN mkdir -p /app/cache
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

# ----------------------------------------------------
# 6) Pre-download HF model at build time
# ----------------------------------------------------
RUN python - <<EOF
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_id = "nsayer/mon_modele"
AutoTokenizer.from_pretrained(model_id)
AutoModelForSequenceClassification.from_pretrained(model_id)
EOF

# ----------------------------------------------------
# 7) Copy application source code
# ----------------------------------------------------
COPY src/transformer/ /app/

# Expose FastAPI port
EXPOSE 8000

# ----------------------------------------------------
# 8) Start API Server
# ----------------------------------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
