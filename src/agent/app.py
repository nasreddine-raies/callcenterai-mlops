from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
import requests
import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram
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

TFIDF_URL = "http://tfidf_service:8000/predict"
TRANSFORMER_URL = "http://transformer_service:8000/predict"


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def route_prediction(request: PredictRequest):
    start_time = time.time()
    payload = {"text": request.text}

    # Call TF-IDF model
    tfidf_res = requests.post(TFIDF_URL, json=payload).json()

    # Call Transformer model
    transformer_res = requests.post(TRANSFORMER_URL, json=payload).json()

    tfidf_conf = tfidf_res.get("confidence")
    tr_conf = transformer_res.get("confidence")

    # Smart routing logic
    if tr_conf is not None and tr_conf >= 0.75:
        chosen = "transformer"
        result = transformer_res
    elif tfidf_conf is not None and tfidf_conf >= 0.55:
        chosen = "tfidf"
        result = tfidf_res
    else:
        # fallback: choose the highest confidence
        if (tr_conf or 0) >= (tfidf_conf or 0):
            chosen = "transformer"
            result = transformer_res
        else:
            chosen = "tfidf"
            result = tfidf_res

    # Record metrics
    http_requests_total.labels(method='POST', endpoint='/predict', status=200).inc()
    http_request_duration_seconds.labels(method='POST', endpoint='/predict').observe(time.time() - start_time)

    return {
        "chosen_model": chosen,
        "label": result.get("label"),
        "confidence": result.get("confidence"),
        "details": {
            "tfidf": tfidf_res,
            "transformer": transformer_res
        }
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint for router"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/summary")
def metrics_summary():
    """JSON summary of key metrics for visualization"""
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
    
    return metrics_data

