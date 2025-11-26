
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
from prometheus_fastapi_instrumentator import Instrumentator
import time
import os

app = FastAPI(title="TFIDF SVC API - TEST")
Instrumentator().instrument(app).expose(app)

# Don't load model at startup - we'll test connectivity first
mlflow_tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(mlflow_tracking_uri)

class TicketRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.get("/")
def read_root():
    return {"message": "TFIDF SVC API TEST is running"}

@app.get("/health")
def health_check():
    """Basic health check without model loading"""
    try:
        # Just test MLflow connectivity
        import requests
        response = requests.get(f"{mlflow_tracking_uri}/health", timeout=5)
        return {
            "status": "healthy", 
            "mlflow_connected": response.status_code == 200,
            "model_loaded": False,
            "message": "API is running but model not loaded"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/test-mlflow")
def test_mlflow():
    """Test MLflow connectivity specifically"""
    try:
        import requests
        response = requests.get(f"{mlflow_tracking_uri}/health", timeout=5)
        return {
            "mlflow_status": "connected" if response.status_code == 200 else "failed",
            "response": response.text,
            "status_code": response.status_code
        }
    except Exception as e:
        return {"mlflow_status": "error", "error": str(e)}

@app.post("/predict")
def predict_ticket(req: TicketRequest):
    return {
        "error": "Model not loaded", 
        "message": "Model artifacts are missing. Please retrain the model.",
        "suggested_fix": "Run: python src/tfidf_svc/train.py"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)