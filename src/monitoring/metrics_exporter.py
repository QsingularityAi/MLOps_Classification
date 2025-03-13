from flask import Flask, Response
import pandas as pd
import numpy as np
import time
import threading
import joblib
import json
import os
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import CollectorRegistry
import logging

# Set

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Set up Prometheus metrics
REGISTRY = CollectorRegistry()
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy score', ['model_version'], registry=REGISTRY)
MODEL_F1 = Gauge('model_f1_score', 'Model F1 score', ['model_version', 'class'], registry=REGISTRY)
MODEL_DRIFT_SCORE = Gauge('model_drift_score', 'Data drift score', registry=REGISTRY)
FEATURE_IMPORTANCE = Gauge('feature_importance', 'Feature importance values', ['feature'], registry=REGISTRY)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds', 
    'Prediction latency in seconds',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0),
    registry=REGISTRY
)
PERFORMANCE_COUNTER = Counter(
    'model_performance_checks_total', 
    'Number of model performance checks',
    registry=REGISTRY
)

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'models/latest_model.joblib')
REFERENCE_DATA_PATH = os.getenv('REFERENCE_DATA_PATH', 'data/processed/reference_data.csv')
TEST_DATA_PATH = os.getenv('TEST_DATA_PATH', 'data/processed/test_data.csv')
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '300'))  # Default: 5 minutes

def update_metrics():
    """Update metrics at regular intervals"""
    while True:
        try:
            # Load model and data
            model = joblib.load(MODEL_PATH)
            
            # Get model version from filename
            model_version = os.path.basename(MODEL_PATH).split('_')[-1].replace('.joblib', '')
            logger.info(f"Updating metrics for model version {model_version}")
            
            # Try to load reference data
            try:
                reference_data = pd.read_csv(REFERENCE_DATA_PATH)
                logger.info(f"Loaded reference data from {REFERENCE_DATA_PATH}")
            except Exception as e:
                logger.warning(f"Error loading reference data: {str(e)}")
                reference_data = None
            
            # Try to load test data
            try:
                test_data = pd.read_csv(TEST_DATA_PATH)
                logger.info(f"Loaded test data from {TEST_DATA_PATH}")
            except Exception as e:
                logger.warning(f"Error loading test data: {str(e)}")
                test_data = None
            
            # Update metrics only if we have test data
            if test_data is not None:
                # Measure prediction latency
                X_test = test_data.drop('stroke', axis=1) if 'stroke' in test_data.columns else test_data
                sample_size = min(100, len(X_test))
                X_sample = X_test.iloc[:sample_size]
                
                start_time = time.time()
                model.predict(X_sample)
                latency = (time.time() - start_time) / sample_size
                PREDICTION_LATENCY.observe(latency)
                logger.info(f"Prediction latency: {latency:.6f} seconds per sample")
                
                # If test data has labels, calculate accuracy and F1 score
                if 'stroke' in test_data.columns:
                    y_test = test_data['stroke']
                    y_pred = model.predict(X_test)
                    
                    # Calculate accuracy
                    accuracy = np.mean(y_pred == y_test)
                    MODEL_ACCURACY.labels(model_version=model_version).set(accuracy)
                    logger.info(f"Model accuracy: {accuracy:.4f}")
                    
                    # Calculate F1 score (simplified)
                    tp = np.sum((y_test == 1) & (y_pred == 1))
                    fp = np.sum((y_test == 0) & (y_pred == 1))
                    fn = np.sum((y_test == 1) & (y_pred == 0))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    MODEL_F1.labels(model_version=model_version, class='stroke').set(f1)
                    logger.info(f"Model F1 score: {f1:.4f}")
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_') and X_test is not None:
                feature_names = X_test.columns
                for feature, importance in zip(feature_names, model.feature_importances_):
                    FEATURE_IMPORTANCE.labels(feature=feature).set(importance)
                logger.info("Updated feature importance metrics")
            
            # Detect data drift if we have both reference and test data
            if reference_data is not None and test_data is not None:
                drift_score = 0
                drift_features = ['age', 'avg_glucose_level', 'bmi']
                
                for col in drift_features:
                    if col in reference_data.columns and col in test_data.columns:
                        ref_mean = reference_data[col].mean()
                        test_mean = test_data[col].mean()
                        if ref_mean != 0:
                            # Normalized absolute difference
                            feature_drift = abs(ref_mean - test_mean) / abs(ref_mean)
                            drift_score += feature_drift
                            logger.info(f"Drift for {col}: {feature_drift:.4f}")
                
                if len(drift_features) > 0:
                    drift_score = drift_score / len(drift_features)  # Average across features
                    MODEL_DRIFT_SCORE.set(drift_score)
                    logger.info(f"Overall drift score: {drift_score:.4f}")
            
            PERFORMANCE_COUNTER.inc()
            logger.info(f"Metrics updated successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}", exc_info=True)
        
        time.sleep(CHECK_INTERVAL)

@app.route('/model-metrics')
def metrics():
    """Expose metrics for Prometheus"""
    return Response(generate_latest(REGISTRY), mimetype='text/plain')

@app.route('/health')
def health():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == '__main__':
    # Start metrics update thread
    metrics_thread = threading.Thread(target=update_metrics, daemon=True)
    metrics_thread.start()
    
    # Get port from environment or use default
    port = int(os.getenv('PORT', '8001'))
    
    # Start the Flask app
    logger.info(f"Starting metrics exporter on port {port}")
    app.run(host='0.0.0.0', port=port)