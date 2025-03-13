from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import time
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import logging

# Set up logging
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

# Load the model
model_path = os.getenv("MODEL_PATH", "models/latest_model.joblib")
try:
    model = joblib.load(model_path)
    model_version = os.path.basename(model_path).split("_")[-1].replace(".joblib", "")
    # Set model version in metric
    MODEL_VERSION.set(1)
    logger.info(f"Model loaded from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    model_version = "unknown"

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