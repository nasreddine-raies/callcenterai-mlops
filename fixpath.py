import mlflow
import os
from mlflow.tracking import MlflowClient

def fix_model_paths():
    # Use file URI for container path
    mlflow.set_tracking_uri("file:///app/mlruns")
    client = MlflowClient()
    
    print("🔧 Fixing model paths for container...")
    
    # Find all experiments and runs
    experiments = mlflow.search_experiments()
    
    for exp in experiments:
        print(f"\nExperiment: {exp.name}")
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        
        for run in runs:
            run_id = run.info.run_id
            # Check if this run has a model that works in container
            try:
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.pyfunc.load_model(model_uri)
                print(f"✅ Run {run_id} works in container!")
                
                # Register this model with container-compatible paths
                try:
                    # Delete old registrations
                    try:
                        client.delete_registered_model("container-ticket-classifier")
                    except:
                        pass
                    
                    # Register new one
                    mlflow.register_model(model_uri, "container-ticket-classifier")
                    print(f"✅ Registered: container-ticket-classifier from {run_id}")
                    return True
                    
                except Exception as e:
                    print(f"❌ Registration failed: {e}")
                    
            except Exception as e:
                continue
    
    return False

if __name__ == "__main__":
    if fix_model_paths():
        print("\n🎉 Model paths fixed for container!")
    else:
        print("\n💥 No working models found")