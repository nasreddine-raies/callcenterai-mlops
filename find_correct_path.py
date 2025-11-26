import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteTicketClassifier(mlflow.pyfunc.PythonModel):
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier
    
    def predict(self, context, model_input):
        """
        Predict topics for input texts
        model_input: DataFrame with 'Document' column or list of strings
        """
        # Handle different input types
        if isinstance(model_input, pd.DataFrame):
            texts = model_input['Document'].tolist()
        elif isinstance(model_input, list):
            texts = model_input
        else:
            texts = [str(model_input)]
        
        # Transform text to TF-IDF features
        X_tfidf = self.vectorizer.transform(texts)
        
        # Make predictions
        predictions = self.classifier.predict(X_tfidf)
        
        return predictions

def create_complete_model():
    """Create a complete model with vectorizer and classifier"""
    
    # Load your data
    df = pd.read_csv("./data/data_final.csv")
    logger.info(f"Dataset shape: {df.shape}")
    
    # Use a sample for quick training
    df_sample = df.head(2000)
    X = df_sample['Document']
    y = df_sample['Topic_group']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("file:///home/nasreddine/Desktop/callcenterai/mlruns")
    mlflow.set_experiment("Complete_Ticket_Classifier")
    
    with mlflow.start_run() as run:
        logger.info(f"Starting run: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_param("model_type", "TFIDF_SVM_Complete")
        mlflow.log_param("dataset_size", len(df_sample))
        mlflow.log_param("test_size", 0.2)
        
        # Create and train TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        logger.info("Fitting TF-IDF vectorizer...")
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train SVM classifier
        logger.info("Training SVM classifier...")
        classifier = SVC(kernel='linear', probability=True, random_state=42)
        classifier.fit(X_train_tfidf, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Create and log the complete model
        complete_model = CompleteTicketClassifier(vectorizer, classifier)
        
        # Log as pyfunc model (includes both vectorizer and classifier)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=complete_model,
            registered_model_name="complete-ticket-classifier"
        )
        
        logger.info(f"✅ Complete model created! Accuracy: {accuracy:.4f}")
        logger.info(f"✅ Model registered as: complete-ticket-classifier")
        logger.info(f"✅ Run ID: {run.info.run_id}")
        
        # Test the model
        test_texts = [
            "computer hardware issue",
            "need HR support",
            "software problem"
        ]
        
        predictions = complete_model.predict(None, test_texts)
        for text, pred in zip(test_texts, predictions):
            logger.info(f"   Test: '{text}' -> '{pred}'")
        
        return run.info.run_id

if __name__ == "__main__":
    run_id = create_complete_model()
    print(f"\n🎉 Complete model created successfully! Run ID: {run_id}")