#!/usr/bin/env python3
"""
Register the latest run from the `Simple_Ticket_Classifier` experiment
to the MLflow Model Registry and promote it to a given stage.

This script expects the MLflow Tracking URI to be a SQLite DB at
`mlruns/mlflow.db` (the same DB created by `src/tfidf_svc/train.py`).

It requires `mlflow` to be installed in the environment.
"""
import mlflow
from mlflow.tracking import MlflowClient
import sys
import time


def main(model_name: str = "tfidf-svc-call-center", stage: str = "Staging"):
    tracking_db = "sqlite:///mlruns/mlflow.db"
    print(f"Using tracking DB: {tracking_db}")
    mlflow.set_tracking_uri(tracking_db)
    client = MlflowClient()

    exp = client.get_experiment_by_name("Simple_Ticket_Classifier")
    if exp is None:
        print("Experiment 'Simple_Ticket_Classifier' not found. Exiting.")
        sys.exit(1)

    exp_id = exp.experiment_id
    print(f"Found experiment id: {exp_id}")

    runs = client.search_runs([exp_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        print("No runs found for the experiment. Exiting.")
        sys.exit(1)

    latest = runs[0]
    run_id = latest.info.run_id
    print(f"Latest run id: {run_id}")

    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from URI: {model_uri} as '{model_name}'")

    # Attempt registration (requires MLflow server to be reachable if using REST-based registry)
    try:
        result = mlflow.register_model(model_uri, model_name)
        version = result.version
        print(f"Model registered as version {version}")

        # Wait briefly for the registry to accept the model
        time.sleep(2)

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,
        )
        print(f"Model version {version} transitioned to {stage}")
    except Exception as e:
        print("Registration failed:", e)
        sys.exit(2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register latest MLflow run to Model Registry")
    parser.add_argument("--model-name", default="tfidf-svc-call-center")
    parser.add_argument("--stage", default="Staging")
    args = parser.parse_args()
    main(args.model_name, args.stage)
