from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import time
import glob
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import logging
import os
import sys

# Set up logging with more detailed information
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Stroke Prediction API")

# Set up metrics endpoint for Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Set up Prometheus metrics
REQUEST_COUNT = Counter(
    "stroke_prediction_requests_total", 
    "Total number of prediction requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "stroke_prediction_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)
PREDICTION_PROBABILITY = Histogram(
    "stroke_prediction_probability_distribution",
    "Distribution of prediction probabilities",
    ["prediction"],
    buckets=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)
MODEL_VERSION = Gauge(
    "stroke_prediction_model_version",
    "Current model version timestamp"
)

# Debug environment
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")
if os.path.exists('models'):
    logger.info(f"Models directory exists and contains: {os.listdir('models')}")
else:
    logger.error(f"Models directory does not exist in {os.getcwd()}")

# Load the model - find the latest model file with enhanced error checking
def find_latest_model(model_dir="models"):
    logger.info(f"Searching for model files in directory: {model_dir}")
    
    # Check if the directory exists
    if not os.path.exists(model_dir):
        logger.error(f"Model directory {model_dir} does not exist")
        # Try to find the models directory
        for root, dirs, files in os.walk("."):
            for d in dirs:
                if d.lower() == "models":
                    logger.info(f"Found potential models directory at: {os.path.join(root, d)}")
                    # List contents of the found directory
                    model_path = os.path.join(root, d)
                    logger.info(f"Contents of {model_path}: {os.listdir(model_path)}")
        return None
    
    # List all .joblib files in the model directory
    model_files = glob.glob(os.path.join(model_dir, "*.joblib"))
    
    if not model_files:
        logger.error(f"No model files found in {model_dir}")
        # List all files in the directory to see what's there
        logger.info(f"Contents of {model_dir}: {os.listdir(model_dir)}")
        return None
    
    # Find the most recent model file based on filename (which includes timestamp)
    latest_model = sorted(model_files)[-1]
    logger.info(f"Selected latest model: {latest_model}")
    return latest_model

# Try to load the model from environment variable first, then fall back to finding latest
model_path = os.getenv("MODEL_PATH", "models/latest_model.joblib")
logger.info(f"Model path from environment variable: {model_path}")

try:
    # First try direct load
    logger.info(f"Attempting to load model from: {model_path}")
    model = joblib.load(model_path)
    model_version = os.path.basename(model_path).split("_")[-1].replace(".joblib", "")
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.warning(f"Could not load model from {model_path}: {str(e)}")
    
    # Try with absolute path
    try:
        abs_path = os.path.abspath(model_path)
        logger.info(f"Trying with absolute path: {abs_path}")
        model = joblib.load(abs_path)
        model_version = os.path.basename(abs_path).split("_")[-1].replace(".joblib", "")
        logger.info(f"Model loaded successfully with absolute path from {abs_path}")
    except Exception as e:
        logger.warning(f"Could not load model with absolute path: {str(e)}")
        
        # Now try to find the latest model
        logger.info("Attempting to find latest model file...")
        try:
            latest_model_path = find_latest_model()
            if latest_model_path:
                logger.info(f"Found latest model at: {latest_model_path}")
                model = joblib.load(latest_model_path)
                model_version = os.path.basename(latest_model_path).split("_")[-1].replace(".joblib", "")
                logger.info(f"Model loaded successfully from {latest_model_path}")
            else:
                logger.error("No model files found")
                # Try looking in other locations
                for potential_dir in ["../models", "/app/models", "./models"]:
                    if os.path.exists(potential_dir):
                        logger.info(f"Found potential model directory: {potential_dir}")
                        logger.info(f"Contents: {os.listdir(potential_dir)}")
                        # Try the first joblib file if any
                        joblib_files = [f for f in os.listdir(potential_dir) if f.endswith('.joblib')]
                        if joblib_files:
                            try_path = os.path.join(potential_dir, joblib_files[0])
                            logger.info(f"Trying to load model from {try_path}")
                            try:
                                model = joblib.load(try_path)
                                model_version = os.path.basename(try_path).split("_")[-1].replace(".joblib", "")
                                logger.info(f"Model loaded successfully from {try_path}")
                                break
                            except Exception as e:
                                logger.warning(f"Failed to load from {try_path}: {str(e)}")
                else:
                    model = None
                    model_version = "unknown"
        except Exception as e:
            logger.error(f"Error searching for models: {str(e)}")
            model = None
            model_version = "unknown"

# Set model version in metric if model loaded successfully
if model is not None:
    MODEL_VERSION.set(1)
    logger.info("Model loaded successfully and metrics updated")
else:
    logger.error("Failed to load model after all attempts")

class PatientData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

class PredictionResponse(BaseModel):
    stroke_probability: float
    stroke_prediction: bool
    model_version: str
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the Stroke Prediction API", "model_version": model_version}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": model_version}

@app.get("/debug")
def debug_info():
    """Debug endpoint to return information about the environment and file system"""
    debug_data = {
        "cwd": os.getcwd(),
        "model_path": model_path,
        "model_loaded": model is not None,
        "model_version": model_version,
        "directory_contents": {
            "root": os.listdir("."),
        }
    }
    
    # Check various directories
    for dir_path in ["models", "../models", "/app/models"]:
        if os.path.exists(dir_path):
            debug_data["directory_contents"][dir_path] = os.listdir(dir_path)
    
    return debug_data

@app.post("/predict", response_model=PredictionResponse)
def predict_stroke(patient: PatientData):
    """Endpoint to predict stroke risk"""
    start_time = time.time()
    
    if model is None:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert patient data to DataFrame
        patient_dict = patient.dict()
        df = pd.DataFrame([patient_dict])
        
        # Preprocess input (match the preprocessing done during training)
        # One-hot encoding
        df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 
                                         'residence_type', 'smoking_status'])
        
        # Ensure all expected columns are present (fill missing with 0)
        expected_columns = model.feature_names_in_
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Keep only the columns used during model training
        df = df[expected_columns]
        
        # Make prediction
        stroke_prob = model.predict_proba(df)[0][1]
        stroke_pred = bool(model.predict(df)[0])
        
        # Update metrics
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        PREDICTION_PROBABILITY.labels(prediction="stroke" if stroke_pred else "no_stroke").observe(stroke_prob)
        
        return {
            "stroke_probability": float(stroke_prob),
            "stroke_prediction": stroke_pred,
            "model_version": model_version
        }
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")