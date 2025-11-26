import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# MLflow setup
# -----------------------------
# Ensure mlruns folder exists
MLFLOW_DIR = os.path.abspath("mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)

# Use SQLite DB inside mlruns/
MLFLOW_DB_PATH = os.path.join(MLFLOW_DIR, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("Simple_Ticket_Classifier")


def train_with_mlflow():
    """Train a TFIDF + SVC model and log everything to MLflow"""
    
    # -----------------------------
    # Load data
    # -----------------------------
    data_path = os.path.abspath("data/data_final.csv")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Dataset loaded. Shape: {df.shape}")

    # Sample for faster training (optional)
    df_sample = df.head(5000)
    X = df_sample['Document']
    y = df_sample['Topic_group']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Start MLflow run
    # -----------------------------
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", "TFIDF_SVC")
        mlflow.log_param("dataset_size", len(df_sample))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_param("tfidf_ngram_range", (1, 2))
        mlflow.log_param("svc_kernel", "linear")

        # -----------------------------
        # Create pipeline
        # -----------------------------
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.8,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('svc', SVC(kernel='linear', probability=True, random_state=42))
        ])

        # Train model
        logger.info("Training pipeline...")
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Training completed. Accuracy: {accuracy:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log the model to MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Save model locally for API/production use
        models_dir = os.path.abspath("models")
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(pipeline, os.path.join(models_dir, "pipeline_model.joblib"))
        logger.info(f"✅ Model saved locally to {models_dir}/pipeline_model.joblib")

        return accuracy


if __name__ == "__main__":
    accuracy = train_with_mlflow()
    if accuracy is not None:
        print(f"\n🎉 Training successful! Accuracy: {accuracy:.4f}")
