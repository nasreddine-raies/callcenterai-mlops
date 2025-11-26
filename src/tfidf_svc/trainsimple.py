# fix_registration.py
import mlflow
import os
from mlflow.tracking import MlflowClient

def fix_model_registration():
    mlflow.set_tracking_uri("file:///home/nasreddine/Desktop/callcenterai/mlruns")
    client = MlflowClient()
    
    print("🔧 Fixing model registration...")
    
    # Find the latest run
    runs = client.search_runs(experiment_ids=["0"])  # Default experiment
    if not runs:
        print("❌ No runs found")
        return False
    
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    print(f"Found latest run: {run_id}")
    
    # Check if the run has model artifacts
    artifacts = client.list_artifacts(run_id)
    model_artifacts = [art for art in artifacts if "model" in art.path]
    
    if not model_artifacts:
        print("❌ No model artifacts found in run")
        return False
    
    print(f"✅ Model artifacts found: {[art.path for art in model_artifacts]}")
    
    # Delete the broken registration if it exists
    try:
        client.delete_registered_model("ticket-classifier")
        print("🗑️ Deleted broken model registration")
    except Exception as e:
        print(f"ℹ️ Could not delete model (may not exist): {e}")
    
    # Register the model properly
    try:
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, "ticket-classifier")
        print(f"✅ Model registered successfully: {model_uri}")
        return True
    except Exception as e:
        print(f"❌ Failed to register model: {e}")
        return False

if __name__ == "__main__":
    if fix_model_registration():
        print("\n🎉 Model registration fixed!")
    else:
        print("\n💥 Failed to fix registration")