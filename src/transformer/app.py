# app.py
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import pipeline
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import REGISTRY

# Initialize FastAPI
app = FastAPI(
    title="CallCenterAI Ticket Classifier",
    description="API for classifying IT support tickets using a fine-tuned Transformer model",
    version="1.0"
)

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

# Load your model from Hugging Face Hub
MODEL_ID = "nsayer/mon_modele"
classifier = pipeline("text-classification", model=MODEL_ID)

# Define input data model
class Ticket(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Welcome to CallCenterAI Ticket Classification API!"}


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


@app.post("/predict")
def predict(ticket: Ticket):
    """
    Classify a ticket text into one of the predefined categories.
    """
    start_time = time.time()
    predictions_active.inc()
    
    try:
        # Track inference time
        inference_start = time.time()
        result = classifier(ticket.text)[0]
        inference_duration = time.time() - inference_start
        model_inference_duration_seconds.labels(model_type='transformer').observe(inference_duration)
        
        label = result["label"]
        confidence = round(result["score"], 4)
        
        # Record metrics
        predictions_total.labels(model_type='transformer', label=label).inc()
        http_requests_total.labels(method='POST', endpoint='/predict', status=200).inc()
        http_request_duration_seconds.labels(method='POST', endpoint='/predict').observe(time.time() - start_time)
        
        return {
            "label": label,
            "confidence": confidence
        }
    except Exception as e:
        http_requests_total.labels(method='POST', endpoint='/predict', status=500).inc()
        http_request_duration_seconds.labels(method='POST', endpoint='/predict').observe(time.time() - start_time)
        raise
    finally:
        predictions_active.dec()
