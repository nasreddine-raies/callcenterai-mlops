from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import os
from typing import Optional

app = FastAPI()

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
    pipeline = load_model()
    prediction = pipeline.predict([request.text])[0]
    return {"prediction": prediction}


@app.get("/")
def root():
    return {"status": "API running", "model_uri": model_uri}