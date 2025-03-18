import argparse
import json
import os
import logging
import pandas as pd
from datetime import datetime

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_processing.ingest import (load_data, check_data_quality, 
                                      preprocess_data, split_data, save_splits)
from src.model.train import load_config, train_model, save_model
from src.evaluation.evaluate import evaluate_model, save_evaluation_results, plot_evaluation_results
from src.model.mlflow_integration import setup_mlflow, log_model_training, register_model

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(config_path):
    """Run the complete training pipeline"""
    start_time = datetime.now()
    logger.info(f"Starting pipeline at {start_time}")
    
    # Set up MLflow
    mlflow_run = setup_mlflow()
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Extract configuration parameters
    data_path = config.get("data_path", "data/raw/Brain.csv")
    processed_data_dir = config.get("processed_data_dir", "data/processed")
    model_dir = config.get("model_dir", "models")
    results_dir = config.get("results_dir", "results")
    plot_output_dir = os.path.join(results_dir, "plots") # ADDED LINE
    model_params = config.get("model_params", None)
    test_size = config.get("test_size", 0.2)
    val_size = config.get("val_size", 0.1)
    random_state = config.get("random_state", 42)
    
    # Ensure directories exist
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plot_output_dir, exist_ok=True) # ADDED LINE
    
    # 1. Load data
    logger.info(f"Loading data from {data_path}")
    data = load_data(data_path)
    
    # 2. Check data quality
    logger.info("Checking data quality")
    is_valid, issues = check_data_quality(data)
    if not is_valid:
        logger.error(f"Data quality issues: {issues}")
        raise ValueError("Data quality check failed. See logs for details.")
    
    # 3. Preprocess data
    logger.info("Preprocessing data")
    processed_data = preprocess_data(data)
    
    # 4. Split data
    logger.info("Splitting data into train/val/test sets")
    if val_size > 0:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            processed_data, test_size=test_size, val_size=val_size, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = split_data(
            processed_data, test_size=test_size, val_size=0, random_state=random_state)
        X_val, y_val = None, None
    
    # 5. Save splits
    logger.info("Saving data splits")
    save_splits(X_train, X_test, y_train, y_test, X_val, y_val, output_dir=processed_data_dir)
    
    # 6. Train model
    logger.info("Training model")
    best_model, best_params, best_score = train_model(
        X_train, y_train, model_params=model_params, grid_search=True, random_state=random_state)
    
    # 7. Save model
    logger.info("Saving model")
    model_path = save_model(best_model, model_dir=model_dir)
    
    # 8. Evaluate model
    logger.info("Evaluating model")
    evaluation = evaluate_model(best_model, X_test, y_test)
    
    # 9. Log to MLflow
    logger.info("Logging to MLflow")
    run_id = log_model_training(
        mlflow_run,
        best_model,
        best_params,
        evaluation,
        X_train
    )
    
    # 10. Register model
    logger.info("Registering model")
    register_model(run_id)
    
    # 11. Save evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")
    save_evaluation_results(evaluation, results_path)
    
    # 12. Plot evaluation results (optional) # UNCOMMENTED
    plot_evaluation_results(evaluation, plot_output_dir) # UNCOMMENTED
    
    # Calculate elapsed time
    elapsed_time = datetime.now() - start_time
    logger.info(f"Pipeline completed in {elapsed_time}")
    
    return {
        "model_path": model_path,
        "results_path": results_path,
        "best_params": best_params,
        "best_score": best_score,
        "evaluation": evaluation,
        "mlflow_run_id": run_id
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the training pipeline')
    parser.add_argument('--config', type=str, default='configs/model_config.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    run_pipeline(args.config)
