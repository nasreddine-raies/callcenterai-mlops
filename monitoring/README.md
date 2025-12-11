# Monitoring: Prometheus + Grafana

This directory contains the monitoring stack configuration for the CallCenterAI MLOps project.

## Overview

- **Prometheus**: Time-series database for metrics collection (port 9090)
- **Grafana**: Visualization and dashboarding (port 3000)

## Quick Start

Run the entire stack with:
```powershell
docker-compose up --build
```

## Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| Prometheus | http://localhost:9090 | None |
| Grafana | http://localhost:3000 | admin / admin |
| TFIDF API | http://localhost:8000 | - |
| Transformer API | http://localhost:8002 | - |
| Router | http://localhost:9000 | - |
| Frontend | http://localhost:8501 | - |
| MLflow | http://localhost:5000 | - |

## Metrics Collected

### By Service

**TFIDF SVC Service** (`tfidf_service:8000`)
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency (histogram)
- `http_requests_created` - Request creation timestamp
- `model_inference_duration_seconds` - Model prediction time
- `predictions_active` - Active predictions counter

**Transformer API** (`transformer_service:8000`)
- Same as TFIDF SVC (uses same Prometheus client)

**Router Service** (`router_service:8000`)
- Request routing metrics
- Backend service health status

## Dashboards

### Available Dashboards

1. **CallCenterAI MLOps Monitoring** - Main dashboard showing:
   - API Uptime status
   - Request rate (5m average)
   - Error rate (5m)
   - Response time P95 (5m)
   - Model inference time P99 (5m)
   - Active predictions

### Access Dashboards

1. Go to http://localhost:3000
2. Login with `admin / admin`
3. Click "Dashboards" → "Browse"
4. Select "CallCenterAI MLOps Monitoring"

## Alerts

Alert rules are defined in `prometheus/alert_rules.yml`:

- **APIDown** - Fires when any API service is unreachable for 2+ minutes
- **HighErrorRate** - Fires when error rate > 5% for 5 minutes
- **HighLatency** - Fires when P95 latency > 1s for 5 minutes
- **SlowModelInference** - Fires when P99 inference time > 5s for 10 minutes

## Configuration Files

```
monitoring/
├── prometheus/
│   ├── prometheus.yml          # Prometheus scrape config
│   └── alert_rules.yml         # Alert rules
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── prometheus.yml  # Prometheus datasource config
│   │   └── dashboards/
│   │       └── dashboard.yml   # Dashboard provisioning config
│   └── dashboards/
│       └── mlops-dashboard.json # Main monitoring dashboard
└── README.md                    # This file
```

## Adding Custom Metrics

To add custom metrics to your services:

### In Python (using prometheus-client)

```python
from prometheus_client import Counter, Histogram, Gauge

# Create metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['model', 'label'])
inference_time = Histogram('model_inference_duration_seconds', 'Inference time')
active_predictions = Gauge('predictions_active', 'Active predictions')

# Use metrics in your code
prediction_counter.labels(model='tfidf', label='network').inc()
with inference_time.time():
    result = model.predict(text)
active_predictions.inc()
```

### Expose metrics endpoint

Add to your FastAPI app:

```python
from prometheus_client import make_wsgi_app, CollectorRegistry
from prometheus_client import Counter, Histogram
from starlette.middleware.wsgi import WSGIMiddleware

@app.get("/metrics")
def metrics():
    return make_wsgi_app()
```

## Troubleshooting

### Prometheus can't scrape metrics
- Check if services are running: `docker-compose ps`
- Verify service names in `prometheus.yml` match docker-compose service names
- Check service logs: `docker-compose logs tfidf_service`

### Grafana not showing data
- Ensure Prometheus datasource is configured correctly
- Wait 15-30 seconds for Prometheus to collect first metrics
- Check Prometheus targets: http://localhost:9090/targets

### Alerts not firing
- Check alert rules are loaded: http://localhost:9090/alerts
- Verify rule expressions in `alert_rules.yml` match actual metrics
- Check evaluation: `evaluate_interval` in prometheus.yml

## Next Steps

1. **Customize dashboards** in Grafana UI (changes persist in `monitoring/grafana/dashboards/`)
2. **Add more alert rules** in `monitoring/prometheus/alert_rules.yml`
3. **Set up notification channels** in Grafana (Slack, Email, PagerDuty)
4. **Export dashboards** from Grafana for version control

## Performance Tuning

- **Scrape interval**: Adjust in `prometheus.yml` (default: 15s for global, 5s for APIs)
- **Retention**: Add `--storage.tsdb.retention.time=30d` to Prometheus command in docker-compose
- **Disk usage**: Monitor `/prometheus` and `/var/lib/grafana` volumes

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/grafana/latest/)
- [Prometheus Client Libraries](https://prometheus.io/docs/instrumenting/clientlibs/)
