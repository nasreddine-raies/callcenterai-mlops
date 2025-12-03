from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

TFIDF_URL = "http://tfidf_service:8000/predict"
TRANSFORMER_URL = "http://transformer_service:8000/predict"


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def route_prediction(request: PredictRequest):
    payload = {"text": request.text}

    # Call TF-IDF model
    tfidf_res = requests.post(TFIDF_URL, json=payload).json()

    # Call Transformer model
    transformer_res = requests.post(TRANSFORMER_URL, json=payload).json()

    tfidf_conf = tfidf_res["confidence"]
    tr_conf = transformer_res["confidence"]

    # Smart routing logic
    if tr_conf >= 0.75:
        chosen = "transformer"
        result = transformer_res
    elif tfidf_conf >= 0.55:
        chosen = "tfidf"
        result = tfidf_res
    else:
        # fallback: choose the highest confidence
        if tr_conf >= tfidf_conf:
            chosen = "transformer"
            result = transformer_res
        else:
            chosen = "tfidf"
            result = tfidf_res

    return {
        "chosen_model": chosen,
        "label": result["label"],
        "confidence": result["confidence"],
        "details": {
            "tfidf": tfidf_res,
            "transformer": transformer_res
        }
    }
