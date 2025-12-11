from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import os
from typing import Optional
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import REGISTRY

app = FastAPI()

# Prometheus metrics for tracking
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=REGISTRY
)
model_inference_duration_seconds = Histogram(
    'model_inference_duration_seconds',
    'Model inference time in seconds',
    ['model_type'],
    registry=REGISTRY
)
predictions_total = Counter(
    'predictions_total',
    'Total predictions made',
    ['model_type', 'label'],
    registry=REGISTRY
)
predictions_active = Gauge(
    'predictions_active',
    'Currently active predictions',
    registry=REGISTRY
)

# MLflow tracking URI (uses mounted volume in Docker)
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

# Lazy-loaded model
model = None
model_uri: Optional[str] = None

class PredictRequest(BaseModel):
    text: str

def find_latest_model_from_registry() -> str:
    """
    Finds the latest registered model stored in mlruns/1/models/
    """
    registry_path = "mlruns/1/models"

    if not os.path.exists(registry_path):
        raise FileNotFoundError("No registered MLflow models found in mlruns/1/models")

    versions = [
        v for v in os.listdir(registry_path)
        if os.path.isdir(os.path.join(registry_path, v)) and v.startswith("m-")
    ]

    if not versions:
        raise FileNotFoundError("No MLflow model versions found in registry.")

    # Sort by modification time (latest version first)
    versions = sorted(
        versions,
        key=lambda v: os.path.getmtime(os.path.join(registry_path, v)),
        reverse=True
    )

    latest_version = versions[0]

    model_path = f"mlruns/1/models/{latest_version}/artifacts"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifacts not found at: {model_path}")

    return model_path


def load_model():
    global model, model_uri

    if model is None:
        print("🔍 Searching for latest registered MLflow model...")
        model_uri = find_latest_model_from_registry()
        print(f"📦 Loading model from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Model loaded successfully!")

    return model


@app.post("/predict")
def predict(request: PredictRequest):
    start_time = time.time()
    predictions_active.inc()
    
    try:
        pipeline = load_model()

        # Track inference time
        inference_start = time.time()
        prediction = pipeline.predict([request.text])[0]
        inference_duration = time.time() - inference_start
        model_inference_duration_seconds.labels(model_type='tfidf_svc').observe(inference_duration)

        # Try to compute probability
        try:
            proba = pipeline.predict_proba([request.text])[0]
            confidence = float(max(proba))  # best confidence
        except:
            confidence = None  # model doesn't support probas

        # Record custom metrics
        predictions_total.labels(model_type='tfidf_svc', label=prediction).inc()
        http_requests_total.labels(method='POST', endpoint='/predict', status=200).inc()
        http_request_duration_seconds.labels(method='POST', endpoint='/predict').observe(time.time() - start_time)

        return {
            "label": prediction,
            "confidence": confidence
        }
    except Exception as e:
        http_requests_total.labels(method='POST', endpoint='/predict', status=500).inc()
        http_request_duration_seconds.labels(method='POST', endpoint='/predict').observe(time.time() - start_time)
        raise
    finally:
        predictions_active.dec()


@app.get("/")
def root():
    return {"status": "API running", "model_uri": model_uri}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    # Return metrics in the exposition format with correct content type
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/summary")
def metrics_summary():
    """JSON summary of key metrics for visualization"""
    from prometheus_client.core import CollectorRegistry
    from prometheus_client import REGISTRY
    
    # Collect all metrics
    metrics_data = {}
    
    # Extract key metrics
    for collector in REGISTRY._collector_to_names:
        for metric in collector.collect():
            if metric.name == 'http_requests_total':
                metrics_data['http_requests_total'] = [
                    {
                        'labels': dict(sample.labels),
                        'value': sample.value
                    }
                    for sample in metric.samples if sample.name == 'http_requests_total'
                ]
            elif metric.name == 'http_request_duration_seconds':
                metrics_data['http_request_duration_seconds'] = [
                    {
                        'labels': dict(sample.labels),
                        'value': sample.value
                    }
                    for sample in metric.samples if 'bucket' not in sample.name and '_total' not in sample.name and '_created' not in sample.name
                ]
            elif metric.name == 'model_inference_duration_seconds':
                metrics_data['model_inference_duration_seconds'] = [
                    {
                        'labels': dict(sample.labels),
                        'value': sample.value
                    }
                    for sample in metric.samples if 'bucket' not in sample.name and '_total' not in sample.name and '_created' not in sample.name
                ]
            elif metric.name == 'predictions_total':
                metrics_data['predictions_total'] = [
                    {
                        'labels': dict(sample.labels),
                        'value': sample.value
                    }
                    for sample in metric.samples if sample.name == 'predictions_total'
                ]
            elif metric.name == 'predictions_active':
                metrics_data['predictions_active'] = [
                    {
                        'labels': dict(sample.labels),
                        'value': sample.value
                    }
                    for sample in metric.samples if sample.name == 'predictions_active'
                ]
    
    return metrics_data