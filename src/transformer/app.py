# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI
app = FastAPI(
    title="CallCenterAI Ticket Classifier",
    description="API for classifying IT support tickets using a fine-tuned Transformer model",
    version="1.0"
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

@app.post("/predict")
def predict(ticket: Ticket):
    """
    Classify a ticket text into one of the predefined categories.
    """
    result = classifier(ticket.text)[0]
    return {
        "label": result["label"],
        "confidence": round(result["score"], 4)
    }
