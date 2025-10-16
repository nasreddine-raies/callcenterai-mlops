from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
from prometheus_fastapi_instrumentator import Instrumentator

# Charge le modèle et le vectorizer
vectorizer = joblib.load("src/tfidf_svc/tfidf_vectorizer.joblib")
clf = joblib.load("src/tfidf_svc/tfidf_svc_model.joblib")

app = FastAPI(title="TFIDF SVC API")

# Instrumentation Prometheus
Instrumentator().instrument(app).expose(app)

class TicketRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
def predict_ticket(req: TicketRequest):
    X = vectorizer.transform([req.text])
    label = clf.predict(X)[0]
    confidence = clf.predict_proba(X)[0].max()
    return PredictionResponse(label=label, confidence=confidence)

@app.get("/")
def read_root():
    return {"message": "TFIDF SVC API is running"}