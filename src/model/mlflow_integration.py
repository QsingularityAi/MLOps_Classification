import mlflow
import mlflow.sklearn
import os
from datetime import datetime
import numpy as np

def setup_mlflow(experiment_name="classification-pipeline"):
    """Set up MLflow tracking"""
    # Use local directory for tracking
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    
    return mlflow.start_run()

def log_model_training(run, model, params, metrics, X_train, feature_importances=None):
    """Log model training details to MLflow"""
    # Log parameters
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)
    
    # Log metrics - handle lists/arrays
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (list, np.ndarray)):
            # For arrays, log each element separately
            for i, value in enumerate(metric_value):
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{metric_name}_{i}", float(value))
        else:
            # For regular numeric values
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)
    
    # Log the model
    mlflow.sklearn.log_model(model, "model")
    
    return run.info.run_id

def register_model(run_id, model_name="stroke-prediction-model"):
    """Register the model in MLflow Model Registry"""
    model_uri = f"runs:/{run_id}/model"
    
    result = mlflow.register_model(model_uri, model_name)
    
    # Add a version tag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag(
        name=model_name,
        version=result.version,
        key="timestamp",
        value=timestamp
    )
    
    return result.version