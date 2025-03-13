import os
import time
import logging
import datetime
import pandas as pd
import requests
import json
import argparse
from sklearn.metrics import roc_curve, auc
import joblib
import numpy as np
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("monitoring-pipeline")

def load_latest_model():
    """Load the latest model from the models directory"""
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        raise ValueError(f"Models directory {models_dir} does not exist")
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    if not model_files:
        raise ValueError("No model files found in the models directory.")
    
    # Sort by modification time (newest first)
    latest_model_file = sorted(
        model_files,
        key=lambda x: os.path.getmtime(os.path.join(models_dir, x)),
        reverse=True
    )[0]
    
    model_path = os.path.join(models_dir, latest_model_file)
    logger.info(f"Loading model from {model_path}")
    
    return joblib.load(model_path), model_path

def get_latest_test_data():
    """Get the latest test data"""
    data_dir = "data/processed"
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    test_files = [f for f in os.listdir(data_dir) if "test" in f and f.endswith('.csv')]
    
    if not test_files:
        raise ValueError("No test data files found in the data directory.")
    
    # Sort by modification time (newest first)
    latest_test_file = sorted(
        test_files,
        key=lambda x: os.path.getmtime(os.path.join(data_dir, x)),
        reverse=True
    )[0]
    
    test_data_path = os.path.join(data_dir, latest_test_file)
    logger.info(f"Loading test data from {test_data_path}")
    
    return pd.read_csv(test_data_path)

def generate_roc_curve_data(model, X_test, y_test):
    """Generate ROC curve data for Grafana"""
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # Create a dataframe with ROC curve data
    roc_data = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'threshold': thresholds
    })
    
    # Save to CSV for Grafana to use
    os.makedirs('data/visualization', exist_ok=True)
    roc_csv_path = "data/visualization/roc_curve_data.csv"
    roc_data.to_csv(roc_csv_path, index=False)
    logger.info(f"ROC curve data saved to {roc_csv_path} (AUC: {roc_auc:.4f})")
    
    return roc_auc

def push_to_prometheus(metrics, endpoint="http://localhost:9091/metrics/job/stroke-prediction"):
    """Push custom metrics directly to Prometheus"""
    prometheus_text = ""
    timestamp = int(time.time() * 1000)
    
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            prometheus_text += f"{metric_name} {value} {timestamp}\n"
    
    try:
        response = requests.post(endpoint, data=prometheus_text)
        if response.status_code == 200:
            logger.info("Metrics successfully pushed to Prometheus")
        else:
            logger.error(f"Failed to push metrics to Prometheus: {response.status_code}")
    except Exception as e:
        logger.error(f"Error pushing metrics to Prometheus: {str(e)}")

def create_confusion_matrix_panel(y_test, y_pred):
    """Create confusion matrix data for Grafana panel"""
    conf_matrix = pd.crosstab(y_test, y_pred, 
                             rownames=['Actual'], 
                             colnames=['Predicted'],
                             normalize='index')
    
    # Convert to long format for Grafana
    conf_df = conf_matrix.reset_index().melt(id_vars=['Actual'], 
                                            var_name='Predicted', 
                                            value_name='Percentage')
    
    # Save to CSV for Grafana
    os.makedirs('data/visualization', exist_ok=True)
    conf_matrix_path = "data/visualization/confusion_matrix.csv"
    conf_df.to_csv(conf_matrix_path, index=False)
    logger.info(f"Confusion matrix data saved to {conf_matrix_path}")
    
    # Also save raw counts (not normalized)
    raw_conf_matrix = pd.crosstab(y_test, y_pred, 
                                 rownames=['Actual'], 
                                 colnames=['Predicted'])
    raw_conf_df = raw_conf_matrix.reset_index().melt(id_vars=['Actual'], 
                                                    var_name='Predicted', 
                                                    value_name='Count')
    raw_conf_matrix_path = "data/visualization/confusion_matrix_raw.csv"
    raw_conf_df.to_csv(raw_conf_matrix_path, index=False)
    
    # Calculate key metrics from confusion matrix for Prometheus
    tn = raw_conf_matrix.loc[0, 0] if 0 in raw_conf_matrix.index and 0 in raw_conf_matrix.columns else 0
    fp = raw_conf_matrix.loc[0, 1] if 0 in raw_conf_matrix.index and 1 in raw_conf_matrix.columns else 0
    fn = raw_conf_matrix.loc[1, 0] if 1 in raw_conf_matrix.index and 0 in raw_conf_matrix.columns else 0
    tp = raw_conf_matrix.loc[1, 1] if 1 in raw_conf_matrix.index and 1 in raw_conf_matrix.columns else 0
    
    metrics = {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
    
    if (tp + fp) > 0:
        metrics['precision'] = tp / (tp + fp)
    if (tp + fn) > 0:
        metrics['recall'] = tp / (tp + fn)
    if metrics.get('precision', 0) + metrics.get('recall', 0) > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    
    return metrics

def check_data_drift(reference_data, current_data, numerical_columns=None):
    """Check for data drift between reference and current datasets"""
    if numerical_columns is None:
        numerical_columns = ['age', 'avg_glucose_level', 'bmi']
    
    drift_metrics = {}
    for column in numerical_columns:
        if column in reference_data.columns and column in current_data.columns:
            # Calculate KS statistic
            from scipy.stats import ks_2samp
            ks_stat, p_value = ks_2samp(
                reference_data[column].dropna(),
                current_data[column].dropna()
            )
            
            drift_metrics[f"drift_{column}_ks"] = ks_stat
            drift_metrics[f"drift_{column}_pvalue"] = p_value
            
            # Calculate distribution statistics
            ref_mean = reference_data[column].mean()
            current_mean = current_data[column].mean()
            ref_std = reference_data[column].std()
            current_std = current_data[column].std()
            
            drift_metrics[f"ref_{column}_mean"] = ref_mean
            drift_metrics[f"current_{column}_mean"] = current_mean
            drift_metrics[f"ref_{column}_std"] = ref_std
            drift_metrics[f"current_{column}_std"] = current_std
            
            # Normalized difference
            if abs(ref_mean) > 0:
                drift_metrics[f"drift_{column}_mean_diff"] = abs(ref_mean - current_mean) / abs(ref_mean)
            
            # Overall drift score (average of KS statistics)
            if 'overall_drift_score' not in drift_metrics:
                drift_metrics['overall_drift_score'] = 0
            drift_metrics['overall_drift_score'] += ks_stat
    
    # Average the drift score
    if len(numerical_columns) > 0:
        drift_metrics['overall_drift_score'] /= len(numerical_columns)
    
    # Save drift metrics to CSV for Grafana
    os.makedirs('data/visualization', exist_ok=True)
    drift_df = pd.DataFrame([drift_metrics])
    drift_df_path = "data/visualization/drift_metrics.csv"
    drift_df.to_csv(drift_df_path, index=False)
    logger.info(f"Drift metrics saved to {drift_df_path}")
    
    return drift_metrics

def generate_predictions_dashboard_data(model, X_test, bins=10):
    """Generate prediction distribution data for dashboard"""
    # Get prediction probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Create histogram data
    hist, bin_edges = np.histogram(y_probs, bins=bins, range=(0, 1))
    
    # Create dataframe
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pred_df = pd.DataFrame({
        'probability_bin': bin_centers,
        'count': hist
    })
    
    # Save to CSV for Grafana
    os.makedirs('data/visualization', exist_ok=True)
    pred_dist_path = "data/visualization/prediction_distribution.csv"
    pred_df.to_csv(pred_dist_path, index=False)
    logger.info(f"Prediction distribution saved to {pred_dist_path}")

def run_monitoring_pipeline(args):
    """Run the complete monitoring pipeline"""
    try:
        # Create necessary directories
        os.makedirs('data/visualization', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Load the model and test data
        model, model_path = load_latest_model()
        test_data = get_latest_test_data()
        
        # Get reference data (either from a fixed file or the original training data)
        if os.path.exists(args.reference_data):
            reference_data = pd.read_csv(args.reference_data)
            logger.info(f"Loaded reference data from {args.reference_data}")
        else:
            logger.warning(f"No reference data found at {args.reference_data}. Using test data as reference.")
            reference_data = test_data.copy()
        
        # Prepare features and target
        X_test = test_data.drop('stroke', axis=1) if 'stroke' in test_data.columns else test_data
        if 'stroke' in test_data.columns:
            y_test = test_data['stroke']
            
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Generate ROC curve data
            roc_auc = generate_roc_curve_data(model, X_test, y_test)
            
            # Create confusion matrix data
            conf_metrics = create_confusion_matrix_panel(y_test, y_pred)
            
            # Generate metrics for Prometheus
            all_metrics = {
                'model_accuracy': np.mean(y_pred == y_test),
                'model_roc_auc': roc_auc,
                **conf_metrics
            }
            
            # Generate prediction distribution data
            generate_predictions_dashboard_data(model, X_test)
            
        else:
            logger.warning("No 'stroke' column found in test data. Skipping prediction metrics.")
            all_metrics = {}
        
        # Check for data drift
        numerical_columns = [col for col in X_test.columns if X_test[col].dtype in [np.float64, np.int64]]
        drift_metrics = check_data_drift(reference_data, test_data, numerical_columns)
        
        # Combine all metrics
        all_metrics.update(drift_metrics)
        
        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_names = X_test.columns
            for feature, importance in zip(feature_names, model.feature_importances_):
                all_metrics[f'feature_importance_{feature}'] = importance
        
        # Add model metadata
        model_version = os.path.basename(model_path).split('.')[0]
        all_metrics['model_version'] = model_version
        all_metrics['monitoring_timestamp'] = time.time()
        
        # Push metrics to Prometheus
        if args.push_to_prometheus:
            push_to_prometheus(all_metrics)
        
        # Save all metrics to JSON for reference
        os.makedirs('data/metrics', exist_ok=True)
        metrics_json_path = f"data/metrics/metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_json_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"All metrics saved to {metrics_json_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in monitoring pipeline: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the monitoring pipeline for stroke prediction model')
    parser.add_argument('--reference-data', type=str, default='data/processed/reference_data.csv',
                       help='Path to reference data for drift detection')
    parser.add_argument('--push-to-prometheus', action='store_true',
                       help='Push metrics to Prometheus')
    parser.add_argument('--interval', type=int, default=0,
                       help='Run monitoring at regular intervals (in seconds)')
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    if args.interval > 0:
        logger.info(f"Starting monitoring pipeline with {args.interval} second intervals")
        while True:
            run_monitoring_pipeline(args)
            time.sleep(args.interval)
    else:
        run_monitoring_pipeline(args)