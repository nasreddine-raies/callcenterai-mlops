import mlflow
from mlflow.tracking import MlflowClient
import argparse

def register_model(run_id, model_name, stage="Production"):
    """
    Register a model from a specific run to the model registry
    and transition it to a specific stage
    """
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    client = MlflowClient()
    
    # Get the model URI from the run
    model_uri = f"runs:/{run_id}/model"
    
    print(f"Registering model from run: {run_id}")
    
    # Register the model
    result = mlflow.register_model(model_uri, model_name)
    
    version = result.version
    print(f"Model registered as version {version}")
    
    # Transition to specified stage
    if stage:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True
        )
        print(f"Model version {version} transitioned to {stage}")
    
    return version

def list_models():
    """List all registered models"""
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    models = client.search_registered_models()
    
    if not models:
        print("No registered models found")
        return
    
    for model in models:
        print(f"\nModel: {model.name}")
        print(f"Description: {model.description}")
        
        versions = client.search_model_versions(f"name='{model.name}'")
        for version in versions:
            print(f"  Version {version.version}: Stage={version.current_stage}, Run ID={version.run_id}")

def promote_model(model_name, version, stage):
    """Promote a specific model version to a stage"""
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True
    )
    print(f"Model {model_name} version {version} promoted to {stage}")

def main():
    parser = argparse.ArgumentParser(description="Manage MLflow model registry")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a model from a run')
    register_parser.add_argument('--run-id', required=True, help='MLflow run ID')
    register_parser.add_argument('--model-name', default='tfidf-svc-call-center', help='Model name')
    register_parser.add_argument('--stage', default='Production', help='Stage to transition to')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all registered models')
    
    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote a model version to a stage')
    promote_parser.add_argument('--model-name', required=True, help='Model name')
    promote_parser.add_argument('--version', required=True, help='Model version')
    promote_parser.add_argument('--stage', required=True, help='Target stage (Staging/Production)')
    
    args = parser.parse_args()
    
    if args.command == 'register':
        register_model(args.run_id, args.model_name, args.stage)
    elif args.command == 'list':
        list_models()
    elif args.command == 'promote':
        promote_model(args.model_name, args.version, args.stage)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()